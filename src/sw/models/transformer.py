"""Band-token transformer scaffold.

``BandTransformer.forward`` is wiring only. It runs the tokenizer, then
spectral blocks, a pool over band groups, a 2D spatial position
embedding, spatial blocks, and a segmentation head.

Status: ``SpectralBlock`` and ``SpectralPool`` are filled in (PR #12) and
so is ``SpatialPE``. ``SpatialBlock`` and ``SegHead`` still raise
``NotImplementedError``; the steps are in each forward docstring. Fill
those in; do not rewrite the composition in ``BandTransformer.forward``.

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

    Filled in by Tim (PR #12): it wraps ``sw.attention.ResidualSelfAttentionBlock``.
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
    """Add a fixed 2D sin-cos position embedding on an ``h`` by ``w`` grid.

    Input ``(B, N, D)`` with ``N = h * w``, plus the grid size. Output
    ``(B, N, D)``: the same tokens plus one embedding vector per grid site,
    shared across the batch.

    Why it is needed: ``SpectralPool`` leaves one token per image patch, and
    self-attention in ``SpatialBlock`` treats its input as an unordered set.
    Without this term the spatial blocks cannot tell neighbouring patches
    from distant ones.

    The formula is the fixed 2D sin-cos table from MAE and later ViT
    recipes (the original ViT learned its table instead). MAE needs
    ``D % 4 == 0``; this version also allows ``D % 4 == 2``. ``D`` must be
    even and at least 4.
    With ``F = ceil(D / 4)`` frequencies ``omega_k = temperature ** (-k / F)``
    for ``k = 0 .. F - 1`` (so ``omega_0 = 1``), the code for row ``i`` and
    column ``j`` is::

        row(i) = [sin(i * omega_0), ..., sin(i * omega_{F-1}),
                  cos(i * omega_0), ..., cos(i * omega_{F-1})]
        col(j) = the same with j
        code(i, j) = row(i)[:D // 2] followed by col(j)[:D // 2]

    When ``D % 4 == 0`` each half is exactly ``2F = D // 2`` values and
    nothing is cut. When ``D % 4 == 2`` each half keeps all ``F`` sines and
    the first ``F - 1`` cosines. Either way each axis has sines and cosines.

    At ``D = 64``: 16 frequencies, channels 0-31 encode the row and 32-63
    encode the column. At ``D = 8``: ``omega = (1, 0.01)``, and site
    ``(i, j)`` is ``[sin i, sin 0.01i, cos i, cos 0.01i, sin j, sin 0.01j,
    cos j, cos 0.01j]``. Low ``k`` changes quickly from site to site and
    high ``k`` slowly. At ``D = 64`` on the 32 by 32 grid, every site's
    closest codes belong to its 8 grid neighbours, and distance between
    codes grows roughly (not strictly) with grid distance. At small ``D``
    the few frequencies repeat sooner (at ``D = 8`` a site 6 columns away
    can look closer than a neighbour), which is one reason the model uses
    ``D = 64``.

    Design choices:

    - No parameters and nothing in the state dict. The table is rebuilt
      on each call (``h * w * D`` values, cheap next to attention), so one
      checkpoint runs on a 16 by 16 grid (``--downsample 2``) and a 32 by 32
      grid.
    - Positions are integer grid indices, not fractions of the image. A
      site keeps its code when the grid grows: site ``(3, 4)`` has the same
      code on 16 by 16 and 32 by 32. On the coarser grid each site covers
      twice the pixels per side (four times the area), so the code means
      "grid index", not "image location".
    - Row-major flattening, the tokenizer's order: token ``n`` is grid site
      ``(n // w, n % w)``.

    ``sw.pe`` is a different table, the wavelength encoding inside the
    tokenizer. This one is spatial.
    """

    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        # Even, so the row and column halves are the same size; at least 4,
        # so each half has a sine and a cosine. Below 4, (sin i, sin j)
        # alone nearly repeats on a 32 by 32 grid (sin 8 is close to sin 14).
        if dim < 4 or dim % 2:
            raise ValueError(f"dim must be even and >= 4, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.dim = dim
        # The longest wavelength scale. 10000 is the ViT and MAE default.
        self.temperature = temperature

    def table(self, h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """The ``(h * w, D)`` embedding, row-major (see the class docstring).

        ``dtype`` is float32 or float64. Half precision is refused: bf16
        cannot hold every integer position above 256, and fp16 or bf16
        angles are too coarse. ``forward`` casts a float32 table instead.
        """
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"table dtype must be float32 or float64, got {dtype}")
        if h < 1 or w < 1:
            raise ValueError(f"grid must be at least 1x1, got {h}x{w}")
        n_freq = -(-self.dim // 4)  # F = ceil(D / 4) frequencies per axis
        k = torch.arange(n_freq, device=device, dtype=dtype)
        omega = self.temperature ** (-k / n_freq)  # (F,): omega_0 = 1, then smaller (1.8e-4 at D = 64)
        # Angle at each (position, frequency) pair: an outer product.
        ys = torch.arange(h, device=device, dtype=dtype)[:, None] * omega  # (h, F)
        xs = torch.arange(w, device=device, dtype=dtype)[:, None] * omega  # (w, F)
        row = torch.cat([ys.sin(), ys.cos()], dim=-1)  # (h, 2F): all sines, then all cosines
        col = torch.cat([xs.sin(), xs.cos()], dim=-1)  # (w, 2F)
        # Split the D channels evenly between the axes. When D % 4 == 0
        # each half is all 2F values. When D % 4 == 2 the last cosine of
        # each half is cut (2F = D / 2 + 1).
        n_row = self.dim // 2
        n_col = self.dim - n_row
        # Broadcast so every site (i, j) holds row(i) next to col(j).
        grid = torch.cat(
            [
                row[:, None, :n_row].expand(h, w, n_row),  # constant along a row
                col[None, :, :n_col].expand(h, w, n_col),  # constant along a column
            ],
            dim=-1,
        )  # (h, w, D)
        # Row-major flatten: index i * w + j, the order the tokenizer uses.
        return grid.reshape(h * w, self.dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(B, N, D) + pe(h, w) -> (B, N, D)``, where ``N = h * w``.

        Raises ``ValueError`` if ``x`` is not ``(B, h * w, D)``. That check
        cannot catch ``h`` and ``w`` passed in swapped order (the product is
        the same), so take both from the tokenizer, as
        ``BandTransformer.forward`` does.
        """
        if x.ndim != 3 or x.shape[1] != h * w or x.shape[2] != self.dim:
            raise ValueError(
                f"expected (B, {h * w}, {self.dim}) for a {h}x{w} grid, got {tuple(x.shape)}"
            )
        # Built on the tokens' device. Half-precision tokens get a float32
        # table cast down: a table built in bf16 is off by up to 0.067 at
        # D = 64 on 32 by 32, against 0.002 for the cast float32 table. Float64 tokens get a
        # float64 table, so their values are exact to float64. Broadcasts
        # over the batch. Out of place, so the caller's tensor is not
        # modified.
        table_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        return x + self.table(h, w, device=x.device, dtype=table_dtype).to(x.dtype)


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
    is visible. ``SpatialBlock`` and ``SegHead`` are still TODOs, so a run
    raises ``NotImplementedError`` until those bodies exist.

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
