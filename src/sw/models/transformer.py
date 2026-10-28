"""Band-token transformer scaffold.

``BandTransformer.forward`` is wiring only. It runs the tokenizer, then
spectral blocks, a pool over band groups, a 2D spatial position
embedding, spatial blocks, and a segmentation head.

``SpectralBlock``, ``SpectralPool``, ``SpatialPE``, ``SpatialBlock``, and
``SegHead`` raise ``NotImplementedError``. The steps are in each forward
docstring. Fill those in; do not rewrite the composition in
``BandTransformer.forward``.

Attention inside the blocks has to be ``sw.attention`` (our ``nn.Linear``
and softmax). Do not call nn.MultiheadAttention or drop in a library ViT
block. Suggested owners and the order to fill the bodies are in
``docs/TRANSFORMER-TODO.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sw.models.tokenizer import BandTokenizer
from sw.attention import ResidualSelfAttentionBlock


def _check_block_args(dim: int, heads: int, mlp_ratio: float) -> None:
    """Shared width check. ``sw.attention`` requires ``dim % heads == 0``."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if heads < 1:
        raise ValueError(f"heads must be >= 1, got {heads}")
    if dim % heads != 0:
        raise ValueError(f"dim={dim} must be divisible by heads={heads}")
    if mlp_ratio <= 0:
        raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}")


class SpectralBlock(nn.Module):
    """Pre-norm self-attention across the band-group axis.

    One sequence per spatial token. Input and output are ``(B, N, G, D)``:
    ``B`` batch, ``N = h * w`` spatial tokens, ``G`` band groups, ``D`` the
    width (``dim``).

    The body is a TODO. Suggested owner: Tim.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        _check_block_args(dim, heads, mlp_ratio)
        if mlp_ratio != 4.0:
            raise NotImplementedError(
                "SpectralBlock only supports mlp_ratio=4.0 for now. Use FeedForward for other ratios."
            )
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.attn_block = ResidualSelfAttentionBlock(dim, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, G, D) -> (B, N, G, D)``.

        Hint: reshape to ``(B*N, G, D)`` so each spatial token is its own
        sequence. Then LayerNorm, attention from ``sw.attention`` across
        ``G``, residual, LayerNorm, MLP (hidden width ``dim * mlp_ratio``),
        residual, and reshape back to ``(B, N, G, D)``.

        ``MultiHeadSelfAttention`` returns ``(output, weights)``. Return the
        output tensor only, so the next block can take it.
        ``ResidualSelfAttentionBlock`` is that same pre-norm pattern with
        the MLP fixed at ``4 * dim``. Any other ``mlp_ratio`` needs
        ``FeedForward`` with the hidden width above.
        """
        b, n, g, d = x.shape
        x = x.reshape(b * n, g, d)
        x, _ = self.attn_block(x)
        x = x.reshape(b, n, g, d)
        return x


class SpectralPool(nn.Module):
    """Reduce the band-group axis to one vector per spatial token.

    Input ``(B, N, G, D)``. Output ``(B, N, D)``.

    FOR NOW, the pool is a mean over ``G``.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, G, D) -> (B, N, D)``.

        Hint: mean over the group axis, or pool with a learned token of
        width ``dim`` (a cross-attention query, or a learned weighted sum
        over ``G``). The output has no ``G`` axis.
        """
        return x.mean(dim=2) 

class SpatialPE(nn.Module):
    """Add a 2D position embedding on an ``h`` by ``w`` token grid.

    Input ``(B, N, D)`` with ``N = h * w``, plus the grid size. Output
    ``(B, N, D)``, the same tokens plus the embedding.

    The embedding is a TODO. Suggested owner: Scott. ``sw.pe`` is the
    wavelength table already added inside the tokenizer. This one is spatial.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        # TODO(scott): fixed 2D sin-cos (recipe default, 0 parameters) or a learnable table.

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(B, N, D) + pe(h, w) -> (B, N, D)``, where ``N = h * w``.

        Hint: build an ``(h, w, D)`` embedding, flatten it in row-major
        order to ``(N, D)``, and add it to ``x``. Take ``h`` and ``w`` from
        the arguments. A 256 input with patch 8 is a 32 by 32 grid (1024
        tokens). ``--downsample 2`` makes that 16 by 16 (256 tokens).
        """
        raise NotImplementedError(
            "TODO(scott): SpatialPE. Add a 2D position embedding."
        )


class SpatialBlock(nn.Module):
    """Pre-norm self-attention across spatial tokens.

    Input and output are ``(B, N, D)``: ``B`` batch, ``N = h * w`` spatial
    tokens, ``D`` the width. The group axis is already gone.

    The body is a TODO. Suggested owner: Scott.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        _check_block_args(dim, heads, mlp_ratio)
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, D) -> (B, N, D)``.

        Hint: LayerNorm, attention from ``sw.attention`` across ``N``,
        residual, LayerNorm, MLP (hidden width ``dim * mlp_ratio``),
        residual. The tensor is already ``(B, N, D)``, which is the layout
        ``MultiHeadSelfAttention`` takes.

        Return the output tensor only, not ``(output, weights)``.
        ``ResidualSelfAttentionBlock`` matches this pattern when
        ``mlp_ratio`` is 4. Any other ratio needs ``FeedForward`` with the
        hidden width above.
        """
        raise NotImplementedError(
            "TODO(scott): SpatialBlock. Attention across the N spatial tokens."
        )


class SegHead(nn.Module):
    """Map spatial tokens to full-resolution class logits.

    Input tokens ``(B, N, D)`` with ``N = h * w``, the grid ``(h, w)``, and
    the image size ``out_size = (H, W)``. Output ``(B, num_classes, H, W)``
    with ``num_classes`` 7.

    ``patch`` is the tokenizer stride (default 8). When ``H`` and ``W`` are
    multiples of ``patch``, ``H = h * patch`` and ``W = w * patch``. A crop
    that is not a multiple (30 by 40 yields a 3 by 5 grid) still has to
    land on ``out_size``.

    The body is a TODO. Suggested owner: Scott. The recipe's heavier head
    (a 1x1 to 32 channels, two bilinear-times-2 plus 3x3 stages, and an
    optional full-resolution spectral skip) can replace a single projection
    and upsample later.
    """

    def __init__(self, dim: int, num_classes: int = 7, patch: int = 8):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        self.dim = dim
        self.num_classes = num_classes
        self.patch = patch

    def forward(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        """``(B, N, D) -> (B, num_classes, H, W)``.

        Hint: reshape to ``(B, h, w, D)``, move ``D`` onto the channel axis
        so the map is ``(B, D, h, w)``, project channels to ``num_classes``,
        and upsample to ``out_size`` ``(H, W)``.
        """
        raise NotImplementedError(
            "TODO(scott): SegHead. Map tokens to full-resolution logits."
        )


class BandTransformer(nn.Module):
    """Spectral-then-spatial band-token segmenter.

    The forward below is composition. It is already written so the order
    is visible. The five modules it calls are TODOs, so a run raises
    ``NotImplementedError`` until those bodies exist.

    Args:
        depth_spectral: number of ``SpectralBlock`` s. ``0`` skips spectral
            mixing. ``2`` is the recipe default.
        depth_spatial: number of ``SpatialBlock`` s. ``0`` skips spatial
            mixing. ``6`` is the recipe default.
        dim: width ``D``. Default 64.
        heads: attention heads. Default 4, the recipe value. ``dim`` must
            be divisible by ``heads``.
        mlp_ratio: MLP hidden width is ``dim * mlp_ratio``. Default 4.
        num_classes: 7 (background plus six object classes).
        n_bands: spectral channels. Real cubes have 224.
        group: bands per tokenizer group. Default 16, so ``G = 14``.
        patch: tokenizer kernel and stride, and the head's nominal scale.
            Default 8.

    Input ``(B, n_bands, H, W)`` returns logits ``(B, num_classes, H, W)``.
    """

    def __init__(
        self,
        depth_spectral: int,
        depth_spatial: int,
        dim: int = 64,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 7,
        n_bands: int = 224,
        group: int = 16,
        patch: int = 8,
    ):
        super().__init__()
        if depth_spectral < 0 or depth_spatial < 0:
            raise ValueError(
                f"depths must be >= 0, got spectral {depth_spectral}, spatial {depth_spatial}"
            )
        _check_block_args(dim, heads, mlp_ratio)
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        if group < 1:
            raise ValueError(f"group must be >= 1, got {group}")

        self.depth_spectral = depth_spectral
        self.depth_spatial = depth_spatial
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.patch = patch

        self.tokenizer = BandTokenizer(
            n_bands=n_bands, d_model=dim, group=group, patch=patch
        )
        self.spectral = nn.ModuleList(
            [SpectralBlock(dim, heads, mlp_ratio) for _ in range(depth_spectral)]
        )
        self.pool = SpectralPool(dim)
        self.pos = SpatialPE(dim)
        self.spatial = nn.ModuleList(
            [SpatialBlock(dim, heads, mlp_ratio) for _ in range(depth_spatial)]
        )
        self.head = SegHead(dim, num_classes=num_classes, patch=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W) -> (B, num_classes, H, W)``.

        At a 256 input and patch 8 the tokenizer returns
        ``(B, 1024, G, D)`` and grid ``(32, 32)``. ``--downsample 2`` feeds
        a 128 map, and the grid is ``(16, 16)`` (256 tokens).
        """
        tokens, (h, w) = self.tokenizer(x)  # (B, N, G, D), N = h * w
        for block in self.spectral:
            tokens = block(tokens)  # (B, N, G, D)
        tokens = self.pool(tokens)  # (B, N, D)
        tokens = self.pos(tokens, h, w)  # (B, N, D)
        for block in self.spatial:
            tokens = block(tokens)  # (B, N, D)
        height, width = x.shape[-2:]
        return self.head(tokens, h, w, out_size=(int(height), int(width)))
