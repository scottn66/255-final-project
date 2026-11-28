"""Band-token transformer scaffold.

``BandTransformer.forward`` is wiring only. It runs the tokenizer, then
spectral blocks, a pool over band groups, a 2D spatial position
embedding, spatial blocks, and a segmentation head.

Status: every module is filled in: ``SpectralBlock`` and ``SpectralPool``
(PR #12), then ``SpatialPE``, ``SpatialBlock`` and ``SegHead``. Change the
modules, not the composition in ``BandTransformer.forward``.

Attention inside the blocks has to be ``sw.attention`` (our ``nn.Linear``
and softmax). Do not call nn.MultiheadAttention or drop in a library ViT
block. Suggested owners and the order to fill the bodies are in
``docs/TRANSFORMER-TODO.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sw.models.tokenizer import BandTokenizer, CollapsedTokenizer
from sw.attention import FeedForward, MultiHeadSelfAttention, ResidualSelfAttentionBlock
from sw.data import HYPER_LT_TRAIN_PIXEL_COUNTS


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


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch for some samples.

    In training, each sample in the batch keeps the branch with probability
    ``1 - p`` and loses it (the branch output becomes zero) with probability
    ``p``. A kept branch is divided by ``1 - p``, so its expected value
    equals the branch itself, which is what evaluation uses. In eval mode,
    and whenever ``p == 0``, this is the identity (the same tensor object
    is returned). No parameters.

    Used as ``x = x + drop_path(branch(x))``. A dropped branch leaves
    ``x`` unchanged for that sample, so the block acts as if it were
    shallower. That regularises deep stacks (Huang et al., 2016; DeiT).

    The mask has one entry per item of the first axis and is shared by
    every other axis. ``SpatialBlock`` passes ``(B, N, D)``, so a drop
    there is per image. A caller that folds tokens into the first axis
    (as ``SpectralBlock`` does with ``(B * N, G, D)``) would drop per token
    instead, so such a caller must reshape back to ``(B, ...)`` first.
    ``p`` must be in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop path must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # Shape (B, 1, 1, ...): one Bernoulli draw per sample, broadcast
        # over every other axis.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep
        return x * mask.to(x.dtype) / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


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
    """Pre-norm self-attention across the spatial tokens, then an MLP.

    Input and output are ``(B, N, D)``: ``B`` batch, ``N = h * w`` spatial
    tokens (one per image patch, 1024 at full resolution), ``D`` the width.
    The band-group axis is already gone (``SpectralPool``), and the
    position embedding is already added (``SpatialPE``).

    The computation, with ``LN`` a LayerNorm and ``MHSA`` the from-scratch
    ``sw.attention.MultiHeadSelfAttention``::

        x = x + drop_path(MHSA(LN1(x)))   # each patch reads from all patches
        x = x + drop_path(MLP(LN2(x)))    # each patch on its own, width D * mlp_ratio

    Pre-norm (normalise before each branch, add the raw residual) is the
    order ``ResidualSelfAttentionBlock`` and ``SpectralBlock`` use, and it
    trains more stably from scratch than post-norm (Xiong et al., 2020).

    Why this is written out instead of wrapping
    ``ResidualSelfAttentionBlock``, as ``SpectralBlock`` does: that block
    fixes the MLP width at ``4 * D`` and has no place for stochastic depth.
    The layers and their attribute names (``ln1``, ``attn``, ``ln2``,
    ``ffn``) are the same, so at ``mlp_ratio`` 4 this block has exactly the
    parameters of a ``SpectralBlock`` (49,984 at ``D = 64``), and its
    weights load with ``strict=True`` into a ``ResidualSelfAttentionBlock``
    (or into a ``SpectralBlock``'s ``attn_block``). ``drop_path`` adds no
    parameters.

    Cost: attention builds ``(B, heads, N, N)`` tensors, so memory grows
    with ``N ** 2``. At ``N = 1024``, 4 heads and float32, autograd keeps
    two of them for the backward pass (the softmax output and the
    ``nan_to_num`` copy ``sw.attention`` makes of it): 16 MiB each. With the
    smaller tensors that is about 36 MiB saved per image per block, about
    1.7 GiB for six blocks at batch 8. Measured on CPU, a forward and
    backward of those six blocks raises peak memory by about 2 GiB. At
    ``--downsample 2`` (``N = 256``) the ``N ** 2`` tensors are 16 times
    smaller.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        _check_block_args(dim, heads, mlp_ratio)
        # MLP hidden width, rounded to the nearest integer (Python's round,
        # so an exact .5 goes to the even width: 12 * 0.625 = 7.5 gives 8).
        hidden = int(round(dim * mlp_ratio))
        if hidden < 1:
            raise ValueError(f"dim * mlp_ratio must round to >= 1, got {hidden}")
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)  # nn.Linear Q, K, V, O and a softmax
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden=hidden)  # Linear -> GELU -> Linear
        # One module serves both branches. It holds no state, and each call
        # draws a fresh mask, so the two branches drop independently.
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, D) -> (B, N, D)``.

        Returns the tokens only. The attention weights that
        ``MultiHeadSelfAttention`` also returns are discarded here.
        """
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected (B, N, {self.dim}), got {tuple(x.shape)}")
        attn_out, _weights = self.attn(self.ln1(x))  # (B, N, D); weights (B, H, N, N) unused
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


class SegHead(nn.Module):
    """Map spatial tokens back to full-resolution class logits.

    Input tokens ``(B, N, D)`` with ``N = h * w``, the grid ``(h, w)``, and
    the image size ``out_size = (H, W)``. Output ``(B, num_classes, H, W)``.
    Each token stands for one ``patch`` by ``patch`` square of the image, so
    the head has to turn a 32 by 32 grid of 64-wide vectors into a 256 by
    256 map of 7 logits.

    Steps (the recipe head), with shapes at full resolution (``h = w = 32``,
    ``patch = 8``, ``hidden = 32``)::

        LayerNorm, tokens -> (B, D, h, w)        (B, 64, 32, 32)
        1x1 conv + GELU                          (B, 32, 32, 32)
        bilinear x2, 3x3 conv + GELU             (B, 32, 64, 64)
        bilinear x2, 3x3 conv + GELU             (B, 32, 128, 128)
        bilinear to (h * patch, w * patch)       (B, 32, 256, 256)
        replicate-pad to out_size                (unchanged here)
        [skip only] concat 1x1 conv + GELU of    (B, 32 + 16, 256, 256)
            the input image, 224 -> 16
        1x1 conv + GELU, Dropout2d, 1x1 conv     (B, 7, 256, 256)

    Why each step:

    - LayerNorm first: the pre-norm blocks add their branches to an
      un-normalised residual stream, so the last block's output is not
      normalised.
    - The reshape is row-major (token ``n`` is grid site ``(n // w, n % w)``),
      the tokenizer's order, so each token's map cell sits over its own patch.
    - Two learned upsampling stages, not one jump: the 3x3 convs let
      neighbouring tokens shape a smooth boundary between patches. There are
      ``min(2, floor(log2(patch)))`` of them, so a small patch gets fewer.
      The last step to full size is plain bilinear. As a result a token
      also reaches a little past its own patch. Measured at ``patch = 8``:
      at most 13 pixels when ``H`` and ``W`` are multiples of the patch,
      and up to ``2 * patch - 1 = 15`` when there is a remainder (the
      replicate padding extends the last row and column). Never 2 patches.
    - A limit to know: learned spatial detail stops at stride 2 (the last
      3x3 conv runs at half resolution) and the final x2 is plain
      bilinear, so class boundaries come out about a pixel blurred, and
      1-2 pixel objects (filament, video tape) are hard for this head
      alone. The full-resolution spectral skip below is the per-pixel path.
    - The skip (``skip_channels > 0``, off by default): a 1x1 conv and a
      GELU on the image the tokens came from, ``n_bands`` to
      ``skip_channels`` channels at ``out_size``. It is concatenated with
      the padded map, and the next 1x1 conv (``fuse``) takes
      ``hidden + skip_channels`` channels. A 1x1 conv reads one pixel, so
      this path gives each pixel its own spectrum at full resolution, with
      no neighbours and no tokens. It also means the head can label a
      pixel from its spectrum alone, without the transformer.
      ``sw.skip_probe`` scores a trained model with either path zeroed.
    - Pad, never stretch: the tokenizer drops the pixels past the last full
      patch (a 30 by 40 input gives a 3 by 5 grid over 24 by 40). The map is
      resized to exactly the covered 24 by 40, and the leftover rows and
      columns repeat the nearest covered ones. Stretching 24 rows over 30
      would shift every token off its own pixels.
    - GELU after every conv except the classifier, as in the blocks. No
      BatchNorm or GroupNorm, so a pixel's logits never depend on the other
      images in the batch or on image statistics.

    ``out_size`` must be the image the tokens came from:
    ``h * patch <= H < (h + 1) * patch``, and the same for ``W``. Anything
    else raises, which also catches ``h`` and ``w`` passed in swapped order
    on a non-square image.

    At ``D = 64``, ``hidden = 32``, 7 classes and ``patch = 8``, the head has
    21,991 parameters (the tests derive this from the layer sizes). The
    skip at 224 bands and 16 channels adds 4,112: 3,600 for its 1x1 conv
    and 512 more weights in ``fuse``. That gives 26,103.

    With the skip off the head is the one it was before the skip existed:
    the same modules, state-dict keys and random draws. The skip conv is
    built last, after the classifier, so with the skip on the layers up to
    ``up`` still draw the weights they drew before (``fuse`` has a new
    shape, so it and the classifier draw new ones).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 7,
        patch: int = 8,
        hidden: int = 32,
        dropout: float = 0.0,
        skip_channels: int = 0,
        n_bands: int = 224,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if skip_channels < 0:
            raise ValueError(f"skip_channels must be >= 0, got {skip_channels}")
        if skip_channels and n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {n_bands}")
        self.dim = dim
        self.num_classes = num_classes
        self.patch = patch
        self.hidden = hidden
        self.skip_channels = skip_channels
        # The image bands the skip reads. Not used when the skip is off.
        self.n_bands = n_bands

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Conv2d(dim, hidden, kernel_size=1), nn.GELU())
        # Number of learned x2 stages: at most 2, and never more than the
        # patch allows (2 ** n_up <= patch), so the map is not upsampled past
        # full resolution. patch 1 -> 0, 2 or 3 -> 1, 4 or more -> 2.
        n_up = 0
        while n_up < 2 and 2 ** (n_up + 1) <= patch:
            n_up += 1
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.GELU())
                for _ in range(n_up)
            ]
        )
        # With the skip on, fuse also takes the skip's channels.
        self.fuse = nn.Sequential(nn.Conv2d(hidden + skip_channels, hidden, kernel_size=1), nn.GELU())
        # Dropout2d zeroes whole channels (the same channels at every pixel of
        # an image) in training. Off by default.
        self.drop = nn.Dropout2d(dropout)
        # Conv2d bias defaults to a small uniform draw, not class priors. Use
        # :func:`init_seg_head_prior_bias` (``--head-prior-bias``) for log priors.
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)
        # The full-resolution spectral skip. Built last, and only when it is
        # on: with it off, no module, state-dict key or random draw is added.
        self.skip = None
        if skip_channels:
            self.skip = nn.Sequential(nn.Conv2d(n_bands, skip_channels, kernel_size=1), nn.GELU())

    def forward(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        out_size: tuple[int, int],
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, N, D) -> (B, num_classes, H, W)``; see the class docstring.

        ``image`` is the ``(B, n_bands, H, W)`` input the tokens came from.
        The skip needs it, and a head without the skip refuses it.
        """
        if tokens.ndim != 3 or tokens.shape[1] != h * w or tokens.shape[2] != self.dim:
            raise ValueError(
                f"expected (B, {h * w}, {self.dim}) for a {h}x{w} grid, got {tuple(tokens.shape)}"
            )
        b, _, d = tokens.shape
        height, width = int(out_size[0]), int(out_size[1])
        p = self.patch
        if not (h * p <= height < (h + 1) * p and w * p <= width < (w + 1) * p):
            raise ValueError(
                f"out_size {height}x{width} does not match a {h}x{w} grid of {p}-pixel patches "
                f"(need {h * p}..{(h + 1) * p - 1} by {w * p}..{(w + 1) * p - 1})"
            )
        # getattr: a head object pickled before the skip existed has no ``skip``.
        skip = getattr(self, "skip", None)
        if skip is None:
            if image is not None:
                raise ValueError("this head has no skip (skip_channels=0), so it takes no image")
        elif image is None:
            raise ValueError("the skip needs the input image: pass image=(B, n_bands, H, W)")
        elif image.ndim != 4 or tuple(image.shape) != (b, self.n_bands, height, width):
            raise ValueError(
                f"expected an image of shape ({b}, {self.n_bands}, {height}, {width}) for the skip, "
                f"got {tuple(image.shape)}"
            )
        # (B, N, D) -> (B, D, N) -> (B, D, h, w). The transpose matters: a
        # plain reshape of (B, N, D) would scramble tokens and channels.
        y = self.norm(tokens).transpose(1, 2).reshape(b, d, h, w)
        y = self.proj(y)  # (B, hidden, h, w)
        for stage in self.up:
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = stage(y)
        # Exactly the pixels the patches cover, then pad the remainder.
        cover = (h * p, w * p)
        if tuple(y.shape[-2:]) != cover:
            y = F.interpolate(y, size=cover, mode="bilinear", align_corners=False)
        pad_h, pad_w = height - cover[0], width - cover[1]
        if pad_h or pad_w:
            # F.pad takes (left, right, top, bottom) for the last two axes.
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate")
        if skip is not None:
            # One 1x1 conv on the image itself: each pixel's own spectrum.
            y = torch.cat([y, skip(image)], dim=1)  # (B, hidden + skip_channels, H, W)
        y = self.fuse(y)  # (B, hidden, H, W)
        return self.classifier(self.drop(y))  # (B, num_classes, H, W)


def init_seg_head_prior_bias(
    head: SegHead,
    counts=HYPER_LT_TRAIN_PIXEL_COUNTS,
) -> torch.Tensor:
    """Set ``head.classifier.bias`` to ``log`` class priors from pixel counts.

    Returns the bias vector that was written. Counts must be positive and
    length ``head.num_classes``. Default counts are the hyperspectral train
    pixel totals in ``sw.data.HYPER_LT_TRAIN_PIXEL_COUNTS``.
    """
    if head.classifier.bias is None:
        raise ValueError("SegHead classifier has no bias")
    counts_t = torch.as_tensor(list(counts), dtype=torch.float32)
    if counts_t.numel() != head.num_classes:
        raise ValueError(
            f"expected {head.num_classes} class counts, got {counts_t.numel()}"
        )
    if torch.any(counts_t <= 0):
        raise ValueError("class counts must be positive")
    prior = counts_t / counts_t.sum()
    bias = prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
    with torch.no_grad():
        head.classifier.bias.copy_(bias.to(device=head.classifier.bias.device))
    return bias


# Channels of the head's full-resolution spectral skip (``head_skip=True``).
# 16 is the recipe value: "1x1 spectral skip (224 -> 16)" in docs/RECIPE.md.
HEAD_SKIP_CHANNELS = 16


class BandTransformer(nn.Module):
    """Spectral-then-spatial band-token segmenter.

    The forward below is composition. It is already written so the order
    is visible. Every module it calls is filled in.

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
        pe_init_std: scale of the band tokenizer's group embedding
            (``IndexBandPE``) after its ``N(0, 1)`` draw. Default 1.0.
        tokenizer: ``"band"`` (the default) is ``BandTokenizer``, ``G``
            band groups. ``"collapsed"`` is ``CollapsedTokenizer``: a 1x1
            conv mixes all the bands before the patch embedding, so
            ``G = 1``. Any other value raises ``ValueError``.
        head_skip: ``True`` turns on the head's full-resolution spectral
            skip (``SegHead(skip_channels=16)``): a 1x1 conv from the input
            ``x``, ``n_bands`` to 16 channels, joined to the decoder map
            just before the head's last two convs. Default ``False``, which
            is the model as it was. Works with both tokenizers.

    The skip reads the same tensor the tokenizer reads, so with
    ``--standardize`` it sees standardized bands. It adds 4,112 parameters
    at 224 bands: 492,599 for the fused model and 495,303 for the control.
    With the skip, a pixel's logits depend on that pixel's own spectrum
    through a path that does not pass through any token or attention.
    ``sw.skip_probe`` scores a trained model with either path zeroed.

    The band-collapsed control is ``BandTransformer(0, 8,
    tokenizer="collapsed")``: "the same backbone with the band axis
    collapsed first, at matched parameters". Matched means the total of
    trainable parameters of the whole model: 491,191 against 488,487 for the
    fused ``BandTransformer(2, 6)``, 0.55% more. FLOPs and step time are not
    matched. With ``tokenizer="collapsed"``:

    - ``depth_spectral`` must be 0. With one group, attention has a single
      token to attend to (its weights are all 1), so a spectral block is a
      per-token map, not attention over bands. The parameter total would
      not show it: both block types have 49,984 parameters, so 2+6 would
      also count 491,191 while only 6 blocks do attention.
    - ``group`` and ``pe_init_std`` have no effect: there are no groups and
      no ``IndexBandPE``. Passing them is not an error (their range checks
      still run).
    - ``forward`` is the same code. The tokens are ``(B, N, 1, D)``, and the
      mean over one group is the identity.

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
        pe_init_std: float = 1.0,
        tokenizer: str = "band",
        head_skip: bool = False,
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
        if pe_init_std <= 0:
            raise ValueError(f"pe_init_std must be > 0, got {pe_init_std}")
        if tokenizer not in ("band", "collapsed"):
            raise ValueError(f"tokenizer must be 'band' or 'collapsed', got {tokenizer!r}")
        if tokenizer == "collapsed" and depth_spectral > 0:
            raise ValueError(
                f"tokenizer='collapsed' needs depth_spectral=0, got {depth_spectral}: with one "
                "band group a spectral block is a per-token MLP, not attention over bands. "
                "The parameter-matched control is depth_spectral=0, depth_spatial=8 "
                "(matched against the default 2+6 model)"
            )

        self.depth_spectral = depth_spectral
        self.depth_spatial = depth_spatial
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.patch = patch
        self.pe_init_std = float(pe_init_std)
        self.tokenizer_kind = tokenizer
        self.head_skip = bool(head_skip)

        # The "band" branch is the constructor call this class always made,
        # in the same place, so a seed still gives the same weights.
        if tokenizer == "band":
            self.tokenizer = BandTokenizer(
                n_bands=n_bands,
                d_model=dim,
                group=group,
                patch=patch,
                pe_init_std=pe_init_std,
            )
        else:
            # reduce stays at CollapsedTokenizer's default of 16: that is
            # what matches the parameter count (see its docstring).
            self.tokenizer = CollapsedTokenizer(n_bands=n_bands, d_model=dim, patch=patch)
        self.spectral = nn.ModuleList(
            [SpectralBlock(dim, heads, mlp_ratio) for _ in range(depth_spectral)]
        )
        self.pool = SpectralPool(dim)
        self.pos = SpatialPE(dim)
        self.spatial = nn.ModuleList(
            [SpatialBlock(dim, heads, mlp_ratio) for _ in range(depth_spatial)]
        )
        # skip_channels 0 builds the head exactly as before the skip existed.
        self.head = SegHead(
            dim,
            num_classes=num_classes,
            patch=patch,
            skip_channels=HEAD_SKIP_CHANNELS if head_skip else 0,
            n_bands=n_bands,
        )

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
        if getattr(self.head, "skip", None) is not None:
            # The skip reads x itself, at the size the model was given.
            return self.head(tokens, h, w, out_size=(int(height), int(width)), image=x)
        return self.head(tokens, h, w, out_size=(int(height), int(width)))
