# Transformer plan: SpatialPE, SpatialBlock x6, SegHead

Written Nov 13, 2026. Everything marked "measured" was run on CPU (4 threads, torch 2.15.0, fp32) on a prototype of `main` at `c794745`. The prototype is saved as [`transformer-prototype.patch`](transformer-prototype.patch) (apply with `git apply docs/transformer-prototype.patch` on a branch from `main`); none of it is on `main`. No GPU and no dataset were available, so there are no CUDA or real-data numbers here.

## Context

- **Merged:** `BandTokenizer` (66,624 parameters), `SpectralBlock` (wraps `ResidualSelfAttentionBlock`), `SpectralPool` (mean over groups), and `BandTransformer.forward` (wiring only).
- **Open:** `SpatialPE.forward`, `SpatialBlock.forward` and `SegHead.forward` raise `NotImplementedError`; five tests are `xfail`.
- **Hypothesis the model serves** (`docs/CHECKUP-QA.md`): attending over 14 band groups before attending over space lets a hyperspectral-only model of about 0.49M parameters be competitive with 52.8 and 54.3 FG mIoU.
- **It is false if**, at full resolution, 200 epochs and the CNN's loss, the fused model is (1) more than 3 points below 52.8, or (2) less than 2 points above the same backbone with the band axis collapsed first, at matched parameters.
- **State of this plan:** all three bodies are built in the prototype patch. The full suite is `64 passed`, and `--model transformer --smoke` runs at 488,487 parameters.

## Decisions

| # | Choice | Why | Evidence | Rejected alternative |
|---|---|---|---|---|
| 1 | `SpatialPE` is a fixed 2D sin-cos table, built on each call, row-major, any width >= 2 | 0 parameters and no state-dict keys, so one checkpoint runs on 16x16 and 32x32 grids | Measured: 0 parameters, 0 keys; 1024 unique rows on 32x32, minimum pairwise distance 1.17; a 3x5 table equals the corner of the 6x9 table | Design 2's table, which raises for widths not divisible by 4; a learnable table |
| 2 | `SpatialBlock` is composed from `sw.attention.MultiHeadSelfAttention` and `FeedForward`, with a `DropPath` module on each branch | `mlp_ratio` works for any value; stochastic depth needs access to the branches; attention stays `nn.Linear` plus softmax | Measured: 49,984 parameters, equal to `SpectralBlock`; same output as `ResidualSelfAttentionBlock` with shared weights (1e-6); float64 `gradcheck` passes | Wrapping `ResidualSelfAttentionBlock` (MLP fixed at 4x, no branch access) |
| 3 | `sw.attention` is not changed in this work | Memory fits without it, and it is a behaviour change to a merged file | Measured, batch 8 at 1024 tokens: peak RSS increase 3,451 MB as merged, 2,546 MB with `nan_to_num` only under a mask (monkeypatched in a scratch script) | Design 3's bundled attention edit, whose forced float32 softmax broke float64 `gradcheck` (Critique 1) |
| 4 | `SegHead` is the recipe head from Design 2: token LayerNorm, 1x1 to 32, two bilinear-x2 + 3x3 stages, cover-then-replicate-pad, optional skip, 1x1 fuse, dropout, classifier | The only head that keeps tokens over their own pixels on non-multiple sizes, uses every token at any patch size, and never drops the skip silently | Measured: rows 0-23 of the 30x40 output equal the 24x40 output; all tokens get gradient at patch 1, 2, 4, 6, 8, 16; a one-pixel poke moves one pixel's logits | Design 1's 583-parameter head (cannot draw 1 px structure); Design 3's head (stretches, drops tokens when patch < 4, no token LayerNorm) |
| 5 | The skip is off by default, in the model and on the CLI (`--head-skip` turns it on) | The skip is a per-pixel classifier over all 224 bands; on by default it confounds hypothesis condition 2 | Measured on synthetic spectra: tokenizer + head with the skip and no attention blocks reaches train FG mIoU 1.000 | Skip on by default (Designs 2 and 3) |
| 6 | The skip reaches the head through one keyword on the last line of `BandTransformer.forward` (`image=x`), in its own PR | The required with/without-skip ablation cannot run otherwise; the composition order is unchanged | Measured: with that line reverted, 62 tests pass and the 2 that fail are the skip tests | A forward pre-hook that stashes the cube on the head (hidden state) |
| 7 | The band-collapsed control is a new `CollapsedTokenizer` (1x1 conv 224->16, then the same 16->64 k8 s8 conv and LayerNorm), run with `--tokenizer collapsed --depth-spectral 0 --depth-spatial 8` | `--depth-spectral 0` on the band tokenizer is a mean over groups and cannot see band order, so it is a strawman | Measured: shuffling the 14 groups changes the band-tokenizer 0+N model by 1.5e-08 and the collapsed model by 4.4e-03. Parameters 491,191 against 488,487 (+2,704, 0.55%) | Using band-tokenizer 0+8 as the control (all three designs) |
| 8 | Stochastic depth is one linear schedule over all blocks, spectral first; `SpectralBlock.forward` calls the parts of its `attn_block` | Fused 2+6 and control 0+8 get the same regularisation | Measured: both give rates `0.1 * i / 7` for i = 0..7; `SpectralBlock` at rate 0 equals `attn_block(x)[0]`; its state-dict keys are unchanged | Spatial blocks only (Designs 2 and 3: rate sum 0.30 against 0.40) |
| 9 | `--drop-path` and `--head-dropout` default to 0; recipe runs pass `0.1` explicitly | The overfit gate and the comparison to `cnn_ds2` need regularisation off, and every run should state its settings | Critique 2 measured a weaker overfit result with defaults on | Harness default 0.1 (Designs 2 and 3) |
| 10 | First Colab runs are fp32. `--amp` comes later and only with `GradScaler` (which the harness always uses) | Without loss scaling, fp16 backbone gradients underflow | Measured, CPU fp16 autocast, batch 4 at 128x128: unscaled, tokenizer gradient cosine to fp32 is 0.004 with 99% zeros; at scale 65536 every part is 1.000 | `--amp` on the first real run (Design 3) |
| 11 | Harness gates added: `--no-aug`, train FG mIoU and per-class train IoU in `metrics.jsonl`, batch 1 allowed for the transformer, `--crop-size` must be a multiple of the patch, `--smoke` honours `--batch-size` | The recipe's overfit check ("no aug, train FG mIoU above 95%") was not observable | Measured: the overfit command below reaches train FG mIoU 0.996 on a synthetic root | Running the check by hand in a notebook |
| 12 | The smoke test passes `--downsample 1` on random 32x32 cubes and asserts the spatial blocks saw 16 tokens | At the old fixture the grid was 1x1, so spatial attention, the PE and the upsampling were not exercised | Measured with a forward hook inside the test | Leaving the fixture as is (all three designs) |

Mutation checks I ran on copies of the final code (each must fail the suite):

| Mutation | Result |
|---|---|
| Head reshape without `.transpose(1, 2)` | 1 failed (`test_seg_head_keeps_each_token_over_its_own_patch`) |
| PE flattened column-major | 1 failed (`test_spatial_pe_is_fixed_row_major_and_grid_stable`) |
| `DropPath` turned into a no-op | 2 failed |
| `DropPath` without the `1 / keep` rescale | 1 failed |
| Head stretches instead of padding | 1 failed |
| `SpectralBlock` drops per token instead of per image | 1 failed |
| Skip output multiplied by zero | 2 failed |

## The three modules

All code below is in the prototype patch as written. Imports at the top of `src/sw/models/transformer.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from sw.attention import FeedForward, MultiHeadSelfAttention, ResidualSelfAttentionBlock
from sw.models.tokenizer import BandTokenizer, CollapsedTokenizer
```

### SpatialPE

```python
class SpatialPE(nn.Module):
    """Add a fixed 2D sin-cos position embedding on an ``h`` by ``w`` grid.

    Input ``(B, N, D)`` with ``N = h * w``, plus the grid size. Output
    ``(B, N, D)``, the same tokens plus the embedding.

    No parameters and nothing in the state dict, so one checkpoint runs on
    a 16 by 16 grid (``--downsample 2``) and a 32 by 32 grid. The first
    ``ceil(D / 2)`` channels encode the row and the rest the column, each as
    sin then cos. Positions are grid indices, so a site keeps its code when
    the grid grows. ``sw.pe`` is the wavelength table inside the tokenizer.
    This one is spatial.
    """

    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (one row and one column channel), got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.dim = dim
        self.temperature = temperature

    def table(self, h: int, w: int, device=None) -> torch.Tensor:
        """The ``(h * w, D)`` float32 embedding in row-major order.

        Row-major is the tokenizer's order: token ``n`` is grid site
        ``(n // w, n % w)``.
        """
        if h < 1 or w < 1:
            raise ValueError(f"grid must be at least 1x1, got {h}x{w}")
        n_freq = -(-self.dim // 4)  # ceil(D / 4) frequencies per axis
        i = torch.arange(n_freq, device=device, dtype=torch.float32)
        omega = self.temperature ** (-i / n_freq)  # (n_freq,)
        ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega
        xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega
        row = torch.cat([ys.sin(), ys.cos()], dim=-1)  # (h, 2 * n_freq)
        col = torch.cat([xs.sin(), xs.cos()], dim=-1)  # (w, 2 * n_freq)
        n_row = -(-self.dim // 2)  # equals 2 * n_freq when D % 4 == 0
        n_col = self.dim - n_row
        grid = torch.cat(
            [
                row[:, None, :n_row].expand(h, w, n_row),
                col[None, :, :n_col].expand(h, w, n_col),
            ],
            dim=-1,
        )  # (h, w, D)
        return grid.reshape(h * w, self.dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(B, N, D) + pe(h, w) -> (B, N, D)``, where ``N = h * w``."""
        if x.ndim != 3 or x.shape[1] != h * w or x.shape[2] != self.dim:
            raise ValueError(
                f"expected (B, {h * w}, {self.dim}) for a {h}x{w} grid, got {tuple(x.shape)}"
            )
        # Built in float32 on the token device, then cast to the token dtype.
        return x + self.table(h, w, device=x.device).to(x.dtype)
```

### DropPath and SpatialBlock

```python
class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch per sample.

    Training only. The mask has one entry per item of the first axis. A
    kept branch is divided by the keep probability, so the expected value
    matches evaluation. ``p = 0`` and eval mode are the identity. No
    parameters.
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
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep
        return x * mask.to(x.dtype) / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


class SpatialBlock(nn.Module):
    """Pre-norm self-attention across spatial tokens.

    Input and output are ``(B, N, D)``: ``B`` batch, ``N = h * w`` spatial
    tokens, ``D`` the width. The group axis is already gone.

    Same layers and attribute names as
    ``sw.attention.ResidualSelfAttentionBlock`` (``ln1``, ``attn``, ``ln2``,
    ``ffn``), so the parameter count equals a ``SpectralBlock`` at
    ``mlp_ratio`` 4. It is written out for two reasons: the MLP width
    follows ``mlp_ratio``, and stochastic depth sits on each residual
    branch. ``drop_path`` adds no parameters.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        _check_block_args(dim, heads, mlp_ratio)
        hidden = int(round(dim * mlp_ratio))
        if hidden < 1:
            raise ValueError(f"dim * mlp_ratio must round to >= 1, got {hidden}")
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden=hidden)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, D) -> (B, N, D)``. The attention weights are dropped."""
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected (B, N, {self.dim}), got {tuple(x.shape)}")
        attn_out, _ = self.attn(self.ln1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x
```

### SegHead

```python
class SegHead(nn.Module):
    """Map spatial tokens to full-resolution class logits.

    Input tokens ``(B, N, D)`` with ``N = h * w``, the grid ``(h, w)``, and
    the image size ``out_size = (H, W)``. Output ``(B, num_classes, H, W)``.

    Steps (the recipe head):

    1. LayerNorm on the tokens (the pre-norm stack ends un-normalised),
       reshape to ``(B, D, h, w)`` in row-major order, 1x1 conv to
       ``hidden`` channels.
    2. Up to two stages of bilinear x2 then a 3x3 conv. The count is
       ``min(2, floor(log2(patch)))``, so the map never goes below the
       token grid.
    3. Bilinear to the pixels the patches cover, ``(h * patch, w * patch)``,
       then replicate-pad the bottom and right to ``out_size``. The
       tokenizer drops a remainder there (30 by 40 gives a 3 by 5 grid over
       24 by 40), so padding keeps each token over its own pixels.
    4. Optional full-resolution spectral skip: a 1x1 conv on the input cube
       (``skip_bands`` to ``skip_dim``), concatenated. It is the only path
       that sees single pixels. ``skip_bands = 0`` builds no skip.
    5. 1x1 fuse to ``hidden``, ``Dropout2d``, 1x1 classifier.

    GELU follows every conv except the classifier. There is no BatchNorm
    or GroupNorm, so a pixel's logits do not depend on image statistics.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 7,
        patch: int = 8,
        hidden: int = 32,
        skip_bands: int = 0,
        skip_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        if hidden < 1 or skip_dim < 1:
            raise ValueError(f"hidden and skip_dim must be >= 1, got {hidden}, {skip_dim}")
        if skip_bands < 0:
            raise ValueError(f"skip_bands must be >= 0, got {skip_bands}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dim = dim
        self.num_classes = num_classes
        self.patch = patch
        self.skip_bands = skip_bands

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Conv2d(dim, hidden, kernel_size=1), nn.GELU())
        n_up = 0
        while n_up < 2 and 2 ** (n_up + 1) <= patch:
            n_up += 1
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.GELU())
                for _ in range(n_up)
            ]
        )
        if skip_bands > 0:
            self.skip = nn.Sequential(nn.Conv2d(skip_bands, skip_dim, kernel_size=1), nn.GELU())
            fuse_in = hidden + skip_dim
        else:
            self.skip = None
            fuse_in = hidden
        self.fuse = nn.Sequential(nn.Conv2d(fuse_in, hidden, kernel_size=1), nn.GELU())
        self.drop = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        out_size: tuple[int, int],
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, N, D) -> (B, num_classes, H, W)``.

        ``image`` is the input cube ``(B, skip_bands, H, W)``. A head built
        with a skip raises without it. A head built without a skip ignores it.
        """
        b, n, d = tokens.shape
        if n != h * w or d != self.dim:
            raise ValueError(
                f"expected (B, {h * w}, {self.dim}) for a {h}x{w} grid, got {tuple(tokens.shape)}"
            )
        height, width = int(out_size[0]), int(out_size[1])
        y = self.norm(tokens).transpose(1, 2).reshape(b, d, h, w)  # (B, D, h, w)
        y = self.proj(y)
        for stage in self.up:
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = stage(y)
        # The pixels the patches cover. Pad the remainder, do not stretch.
        cover = (min(h * self.patch, height), min(w * self.patch, width))
        if tuple(y.shape[-2:]) != cover:
            y = F.interpolate(y, size=cover, mode="bilinear", align_corners=False)
        pad_h, pad_w = height - cover[0], width - cover[1]
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate")
        if self.skip is not None:
            if image is None:
                raise ValueError(
                    "this head was built with a spectral skip; pass image=(B, C, H, W)"
                )
            if tuple(image.shape) != (b, self.skip_bands, height, width):
                raise ValueError(
                    f"image must be ({b}, {self.skip_bands}, {height}, {width}), "
                    f"got {tuple(image.shape)}"
                )
            y = torch.cat([y, self.skip(image)], dim=1)
        y = self.fuse(y)
        return self.classifier(self.drop(y))
```

### Other changes

**`SpectralBlock`** (Tim's class; constructor gains `drop_path: float = 0.0` and `self.drop_path = DropPath(drop_path)`; the `mlp_ratio != 4` guard and the `attn_block` keys are unchanged):

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, g, d = x.shape
        blk = self.attn_block
        attn_out, _ = blk.attn(blk.ln1(x.reshape(b * n, g, d)))
        x = x + self.drop_path(attn_out.reshape(b, n, g, d))
        x = x + self.drop_path(blk.ffn(blk.ln2(x)))
        return x
```

**`BandTransformer.__init__`** gains `drop_path: float = 0.0, head_skip: bool = False, head_dropout: float = 0.0, tokenizer: str = "band"`:

```python
        if tokenizer == "band":
            self.tokenizer = BandTokenizer(n_bands=n_bands, d_model=dim, group=group, patch=patch)
        else:
            self.tokenizer = CollapsedTokenizer(n_bands=n_bands, d_model=dim, reduce=group, patch=patch)
        # One linear schedule over every block, spectral first.
        depth = depth_spectral + depth_spatial
        rates = [self.drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.spectral = nn.ModuleList(
            [SpectralBlock(dim, heads, mlp_ratio, drop_path=rate) for rate in rates[:depth_spectral]]
        )
        self.pool = SpectralPool(dim)
        self.pos = SpatialPE(dim)
        self.spatial = nn.ModuleList(
            [SpatialBlock(dim, heads, mlp_ratio, drop_path=rate) for rate in rates[depth_spectral:]]
        )
        self.head = SegHead(
            dim, num_classes=num_classes, patch=patch,
            skip_bands=n_bands if head_skip else 0, dropout=head_dropout,
        )
```

**`BandTransformer.forward`**: one keyword added to the last line; nothing else touched. This is the change that needs explicit sign-off (PR 4).

```python
        # image=x feeds the head's optional full-resolution spectral skip.
        return self.head(tokens, h, w, out_size=(int(height), int(width)), image=x)
```

**`src/sw/models/tokenizer.py`**: new class, `BandTokenizer` untouched.

```python
class CollapsedTokenizer(nn.Module):
    """Control tokenizer: mix all bands first, then patch-embed. G = 1."""

    def __init__(self, n_bands=224, d_model=64, reduce=16, patch=8):
        super().__init__()
        if reduce < 1:
            raise ValueError(f"reduce must be >= 1, got {reduce}")
        self.n_bands = n_bands
        self.group = n_bands
        self.n_groups = 1
        self.reduce = nn.Conv2d(n_bands, reduce, kernel_size=1)
        self.proj = nn.Conv2d(reduce, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.shape[1] != self.n_bands:
            raise ValueError(f"Input has {x.shape[1]} bands but tokenizer expects {self.n_bands}")
        t = self.proj(self.reduce(x))  # (B, D, h, w)
        h, w = t.shape[-2:]
        t = self.norm(t.flatten(2).transpose(1, 2))  # (B, h*w, D)
        return t.unsqueeze(2), (h, w)  # (B, h*w, 1, D)
```

**`src/sw/train.py`**
- `TrainConfig` gains `drop_path=0.0`, `head_skip=False`, `head_dropout=0.0`, `tokenizer="band"`, `augment=True`; `build_model` passes the first four through.
- New flags: `--drop-path`, `--head-skip`, `--head-dropout`, `--tokenizer {band,collapsed}`, `--no-aug`.
- `_check_cfg`: batch size 1 is rejected for the CNN only; range checks for the new fields; `--crop-size` must be a multiple of `--patch` for the transformer.
- The "at least 2 images" and drop-last rules apply to the CNN only.
- Each `metrics.jsonl` row gains `train_fg_miou`, `train_per_class_iou` and `epoch_seconds` (train plus validation). The epoch print line gains `train_fg_miou`. `metrics.csv` columns are unchanged, so resuming an old run still appends cleanly.
- A startup line prints the transformer options.
- `--batch-size` defaults to 4, or 2 with `--smoke`; an explicit value is now honoured under `--smoke`.
- The CNN path and its tests are unchanged (`tests/test_train.py` and `tests/test_cnn.py` pass).

**`tests/test_transformer.py`**: the `xfail` marker is gone. 22 tests (was 10):
- PE: order on a non-square grid, grid stability, odd widths.
- Blocks: parameter parity, `mlp_ratio` 2, float64 `gradcheck`, `DropPath` statistics, per-image drop.
- Head: token-to-patch localisation, padding, every token used at six patch sizes, skip flag.
- Model: parameter totals, drop-path schedule, band-order sensitivity of each row, CLI flags, and three smoke variants on a 4x4 grid.

**`tests/test_tokenizer.py`**: one test for `CollapsedTokenizer` (shape `(2, 15, 1, 64)`, 69,328 parameters).

**`src/sw/attention.py`**: not changed. The optional later change (PR 9) is this tail in `scaled_dot_product_attention`. I measured it by monkeypatching in a scratch script, not by editing the file:

```python
    weights = torch.softmax(scores, dim=-1)
    if key_mask is not None:
        # A fully masked row is 0/0. Without a mask there is no such row.
        weights = torch.nan_to_num(weights, nan=0.0)
```

With it, unmasked NaNs propagate instead of being zeroed, so it needs the attention owner's review and a float64 `gradcheck` test. Do not add a forced float32 softmax.

**Not done:** `docs/TRANSFORMER-TODO.md`, `README.md` and `docs/RESULTS.md` are not updated.

## Work breakdown

One branch for WP6, small PRs in this order. Each is a subset of the prototype patch. The last column says whether the code exists in the prototype patch.

| PR | Owner | Files | Test command that must pass | Done when | Built |
|---|---|---|---|---|---|
| 1. SpatialPE | Scott | `transformer.py` (`SpatialPE`), `tests/test_transformer.py` | `PYTHONPATH=src python3 -m pytest -q tests/test_transformer.py -k spatial_pe` | 2 tests pass; 0 parameters, 0 state-dict keys | yes |
| 2. DropPath + SpatialBlock | Scott | `transformer.py`, tests | `PYTHONPATH=src python3 -m pytest -q tests/test_transformer.py -k "spatial_block or drop_path or reject_heads"` | Block counts 49,984; `gradcheck` passes | yes |
| 3. SegHead without the skip path wired; remove all `xfail` marks; new smoke fixture | Scott | `transformer.py` (`SegHead`), tests | `PYTHONPATH=src python3 -m pytest -q .` | No `xfail` left; `BandTransformer.forward` byte-identical to `main`; smoke prints 488,487 | yes |
| 4. Skip flag | Scott, with explicit OK on the `forward` line from the reviewer | `transformer.py` (`head_skip`, `image=x`), `train.py` (`--head-skip`), tests | `PYTHONPATH=src python3 -m pytest -q tests/test_transformer.py -k "skip or ablation or smoke"` | `--head-skip` prints 492,599; a skip head without a cube raises | yes |
| 5. Stochastic depth over all blocks | Tim (touches `SpectralBlock`) | `transformer.py`, `train.py` (`--drop-path`, `--head-dropout`), tests | `PYTHONPATH=src python3 -m pytest -q tests/test_transformer.py -k "drop_path or param_totals or spectral_block"` | Same rate list for 2+6 and 0+8; rate 0 reproduces the merged block | yes |
| 6. Band-collapsed control | Tim | `tokenizer.py`, `transformer.py` (`tokenizer=`), `train.py` (`--tokenizer`), `tests/test_tokenizer.py`, tests | `PYTHONPATH=src python3 -m pytest -q tests/test_tokenizer.py tests/test_transformer.py -k "collapsed or ablation or cli_flags or param_totals"` | Control prints 491,191; the band-order test passes | yes |
| 7. Harness gates | Scott | `train.py`, tests | `PYTHONPATH=src python3 -m pytest -q tests/test_train.py tests/test_transformer.py` | `--no-aug` and `train_fg_miou` exist; batch 1 runs for the transformer | yes |
| 8. Optimiser recipe (WP7) | Scott | `train.py`, `tests/test_train.py` | `PYTHONPATH=src python3 -m pytest -q tests/test_train.py` | Opt-in flags for warmup (5 epochs, per iteration), cosine floor 1e-5, weight decay 0.05 with no decay on norms, biases and the group embedding, `unscale_` then `clip_grad_norm_(1.0)`, Dice term; CNN defaults unchanged | no |
| 9. Attention memory (optional) | Tim | `attention.py`, `tests/test_attention.py` | `PYTHONPATH=src python3 -m pytest -q tests/test_attention.py` | Only if the T4 reading in run 1 below exceeds 12 GiB at batch 8; float64 `gradcheck` test added | no |
| 10. Docs | Scott | `docs/TRANSFORMER-PLAN.md`, `docs/TRANSFORMER-TODO.md`, `README.md`, `docs/RESULTS.md`, `docs/RECIPE.md` | `grep -n "xfail\|NotImplementedError" docs/TRANSFORMER-TODO.md src/sw/models/transformer.py src/sw/train.py` returns only the `SpectralBlock` `mlp_ratio` guard | Measured counts replace "about 0.5M"; RECIPE's "4 MB per image per layer" is corrected (table below) | no |

PRs 1 to 3 give a model that trains end to end with `forward` untouched. PR 8 must land before any run is reported as a result; until then every run is a wiring check or a screening row. Lifting the `mlp_ratio != 4` guard in `SpectralBlock` is a later PR for Tim, needed only for the ratio sweep.

## Measured numbers

### Parameters (counted on built modules)

| Module | Count |
|---|---|
| `BandTokenizer` | 66,624 |
| `CollapsedTokenizer` | 69,328 |
| `SpectralBlock` (each) | 49,984 |
| `SpectralPool` | 0 |
| `SpatialPE` | 0 |
| `SpatialBlock` (each) | 49,984 (33,472 at `mlp_ratio` 2) |
| `SegHead` without skip | 21,991 |
| `SegHead` with skip | 26,103 (norm 128, proj 2,080, up 18,496, skip 3,600, fuse 1,568, classifier 231) |
| **Fused 2+6, no skip (default)** | **488,487** |
| Fused 2+6, skip | 492,599 |
| Collapsed 0+8, no skip | 491,191 |
| Collapsed 0+8, skip | 495,303 |
| Band tokenizer 0+8 (mean-pooled groups) | 488,487 |
| Spectral-only 2+0 | 188,583 |
| No blocks 0+0, no skip / skip | 88,615 / 92,727 |
| `MiniNetSeg` (CNN control) | 571,879 |

### Attention weight memory

Computed from shapes, one fp32 tensor of `B x 4 heads x S x S`. Halve for fp16. The merged `sw.attention` keeps two such tensors per block for backward (softmax output and `nan_to_num` output).

| Tokens S | Batch | One spatial tensor | Retained, 6 spatial blocks | One spectral tensor (`B*S` sequences of 14) | Retained, 2 spectral blocks |
|---|---|---|---|---|---|
| 256 | 4 | 4 MiB | 48 MiB | 3.1 MiB | 12 MiB |
| 256 | 8 | 8 MiB | 96 MiB | 6.1 MiB | 25 MiB |
| 1024 | 4 | 64 MiB | 768 MiB | 12.3 MiB | 49 MiB |
| 1024 | 8 | 128 MiB | 1,536 MiB | 24.5 MiB | 98 MiB |

### One training step on CPU

Measured: second of two forward + backward passes, train mode, cross-entropy on random labels, one process per row. "RSS increase" is peak process RSS minus RSS before the first step; the input batch is not included.

| Input | Tokens | Batch | Model | Forward | Backward | RSS increase | Peak RSS |
|---|---|---|---|---|---|---|---|
| 128x128 | 256 | 4 | fused 2+6 | 0.09 s | 0.11 s | 363 MB | 942 MB |
| 128x128 | 256 | 8 | fused 2+6 | 0.17 s | 0.18 s | 686 MB | 1,321 MB |
| 256x256 | 1024 | 1 | fused 2+6 | 0.13 s | 0.15 s | 595 MB | 1,174 MB |
| 256x256 | 1024 | 4 | fused 2+6 | 0.65 s | 1.04 s | 1,778 MB | 2,527 MB |
| 256x256 | 1024 | 8 | fused 2+6 | 1.40 s | 2.02 s | 3,451 MB | 4,425 MB |
| 256x256 | 1024 | 8 | fused 2+6, skip, drop-path 0.1, dropout 0.1 | 1.62 s | 2.69 s | 3,899 MB | 4,873 MB |
| 256x256 | 1024 | 8 | collapsed 0+8 | 1.34 s | 2.37 s | 2,892 MB | 3,866 MB |
| 256x256 | 1024 | 8 | band 0+8 | 1.24 s | 2.51 s | 2,945 MB | 3,919 MB |
| 256x256 | 1024 | 4 | fused 2+6, `nan_to_num` only under a mask | 0.60 s | 0.77 s | 1,364 MB | 2,112 MB |
| 256x256 | 1024 | 8 | fused 2+6, `nan_to_num` only under a mask | 1.26 s | 1.60 s | 2,546 MB | 3,520 MB |

Computed, not measured: at 3.4 s per step and 65 steps (514 images at batch 8), model compute is about 3.7 minutes per full-resolution epoch on this CPU. GPU step time and tiff read time are unknown.

### Checks

- **Suite:** `PYTHONPATH=src python3 -m pytest -q .` printed `46 passed, 5 xfailed` before, and `64 passed in 4.27s` now.
- **Sizes:** 30x40, 33x47, 29x37, 8x8 and 256x256 all return logits at exactly the input size.
- **PE against tokens:** PE row norm 5.66; pooled token row norm at random init 7.12 on a random cube.
- **Band order at random init:** shuffling the 14 groups changes logits by 1.5e-08 (band 0+2), 4.4e-03 (collapsed 0+2), 1.2e-04 (fused 1+1 and 2+6), 5.0e-02 (band 0+2 with skip).
- **fp16 autocast on CPU** (batch 4, 128x128): gradient cosine to fp32 without loss scaling is 0.004 (tokenizer), 0.000 (spectral), 0.699 (spatial), 0.994 (head). At scale 65536 all four are 1.000. This is a CPU indication, not a CUDA result.

### Smoke command, real output

`SW_ROOT=<synthetic root> PYTHONPATH=src python3 -m sw.train --model transformer --smoke --device cpu --out-dir <scratch>/runs/smoke`, on 4 train and 2 val synthetic cubes of (224, 256, 256). At `--downsample 4` this is an 8x8 grid, 64 tokens. Wall time 3.8 s.

```
BandTransformer parameters: 488,487 (0.488M)
device cpu  epochs 1-1  batch 2  lr 0.001  weight_decay 0.0001  schedule cosine  amp off
train images 4  val images 2  downsample 4  crop None  out <scratch>/runs/smoke
transformer  tokenizer band  blocks 2+6  drop_path 0.0  head_skip off  head_dropout 0.0  augment on
class weights (paper): 0.676 0.937 1.000 1.207 1.125 1.516 0.900
Limited run: the val FG mIoU below checks the wiring. It is not a score to report.
epoch 01  steps    2  lr 0.001  train_loss 1.9675  val_fg_miou 0.0404  train_fg_miou 0.0404
saved <scratch>/runs/smoke/best.pt  (epoch 1, val FG mIoU 0.0404)
saved <scratch>/runs/smoke/last.pt  (epoch 1)
done  best val FG mIoU 0.0404  epoch 1  log <scratch>/runs/smoke/metrics.jsonl
```

The same smoke also ran with `--downsample 1` (1024 tokens) and for every ablation row; the parameter lines matched the table above.

### Overfit one batch (synthetic, not evidence about real data)

Through the CLI, with the command from the next section on CPU: 4 synthetic cubes at `--downsample 2`, one spectrum per class plus noise, lines 1 px wide after downsampling, 200 steps.

| Run | Train loss | Train FG mIoU at step 100 / 150 / 200 | Thin lines (per-class IoU at the end) | Large classes | Wall |
|---|---|---|---|---|---|
| Fused, `--head-skip` | 0.0047 | 0.764 / 0.963 / 0.996 | 1.000 and 0.988 | 0.997 to 1.000 | 111 s |
| Fused, no skip | 0.1236 | 0.481 / 0.605 / 0.624 | 0.000 and 0.000 | 0.806 to 0.993 | 102 s |

In a script (2 cubes of 64x64, 300 AdamW steps at 1e-3, same kind of data):

| Model | Train FG mIoU | 1 px row / 1 px column / 2 px bar |
|---|---|---|
| Fused 2+6, skip | 1.000 | 1.000 / 1.000 / 1.000 |
| No blocks 0+0, skip | 1.000 | 1.000 / 1.000 / 1.000 |
| Fused 2+6, no skip | 0.555 | 0.008 / 0.000 / 0.464 |
| Collapsed 0+8, no skip | 0.660 | 0.071 / 0.000 / 0.982 |
| Band 0+8, no skip | 0.630 | 0.000 / 0.000 / 0.882 |
| No blocks 0+0, no skip | 0.475 | 0.000 / 0.000 / 0.018 |

On pure-noise cubes the fused model did not fit in 300 steps (FG mIoU 0.549 without skip, 0.479 with). Design 3's "IoU 1.000 on noise cubes" did not carry over to this data, so noise cubes are not used as a gate.

## First Colab runs

Set up as in `docs/COLAB.md` (`SW_ROOT` on local disk, checkpoints on Drive). All runs are fp32: no `--amp`. `$RUNS` is `/content/drive/MyDrive/sw_runs`. None of these were run by me.

**0. Smoke.**

```bash
PYTHONPATH=src python -m sw.train --model transformer --smoke --device cuda --out-dir /content/sw_smoke
```

Stop and debug if the parameter line is not `488,487` or the run raises.

**1. Memory and step time at full resolution, batch 8** (in a notebook cell, so the peak is readable):

```python
import sys, torch; sys.path.insert(0, "src")
from sw.train import main
torch.cuda.reset_peak_memory_stats()
main(["--model", "transformer", "--device", "cuda", "--batch-size", "8", "--epochs", "1",
      "--max-train-steps", "5", "--max-train-images", "40", "--max-val-images", "8",
      "--out-dir", "/content/sw_memcheck"])
print(torch.cuda.max_memory_allocated() / 2**30, "GiB")
```

Log the GiB figure and `epoch_seconds`. If it runs out of memory, use batch 4. If the peak is above 12 GiB on a T4, do PR 9 before long runs.

**2. Overfit one batch** (RECIPE section 8 item 1). Regularisation is off by default.

```bash
PYTHONPATH=src python -m sw.train --model transformer --downsample 2 --head-skip --no-aug --no-cosine \
  --max-train-images 4 --max-val-images 4 --batch-size 4 --epochs 200 --num-workers 2 \
  --device cuda --out-dir $RUNS/tf_overfit_skip
```

- **Pass:** `train_fg_miou` above 0.95. If it is still rising at epoch 200, rerun with `--epochs 400` before calling it a failure.
- **Stop and debug if** the loss is flat, or `train_fg_miou` stays under 0.95 at 400 steps. Check labels, token order and the loss first. No 30-epoch run before this passes.
- Then repeat without `--head-skip` (`--out-dir $RUNS/tf_overfit_noskip`). Expected: `train_per_class_iou` in `metrics.jsonl` above 0.9 for film, basket, cardboard and bag, and thin classes near 0. That is the head's resolution limit, not a wiring bug.

**3. Half resolution, same flags as `cnn_ds2`** (`docs/COLAB.md`), fused without and with the skip:

```bash
PYTHONPATH=src python -m sw.train --model transformer --epochs 30 --batch-size 4 --num-workers 2 \
  --device cuda --patience 8 --downsample 2 --out-dir $RUNS/tf_ds2
PYTHONPATH=src python -m sw.train --model transformer --head-skip --epochs 30 --batch-size 4 --num-workers 2 \
  --device cuda --patience 8 --downsample 2 --out-dir $RUNS/tf_ds2_skip
```

- Compare with 0.391 (film 0.552, bag 0.436). These are screening rows: 30 epochs, one seed, CNN optimiser defaults.
- **Stop and debug if** the loss is NaN, or val FG mIoU is under 0.10 at epoch 10 with every foreground class at 0.
- A transformer that trains but trails the CNN is not a bug; go to the decision tree in RECIPE section 7.

**4. Full resolution.** First a CNN row at the same settings (none exists in `docs/RESULTS.md`), then the transformer:

```bash
PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --num-workers 2 --device cuda --patience 8 \
  --out-dir $RUNS/cnn_full
PYTHONPATH=src python -m sw.train --model transformer --epochs 30 --batch-size 4 --num-workers 2 \
  --device cuda --patience 8 --out-dir $RUNS/tf_full
PYTHONPATH=src python -m sw.train --model transformer --head-skip --epochs 30 --batch-size 4 --num-workers 2 \
  --device cuda --patience 8 --out-dir $RUNS/tf_full_skip
```

**What to log for every run:**
- the parameter line and the `transformer ...` options line;
- per epoch: `train_loss`, `train_fg_miou`, `val_fg_miou` and all seven per-class IoUs;
- `epoch_seconds`, and peak GPU memory from run 1;
- one row in `docs/RESULTS.md` with the flags in the notes column.

**The runs the hypothesis is judged on** come after PR 8: full resolution, 200 epochs, fused and collapsed with identical flags, the CNN with the same loss, then three seeds for the reported rows. Budget them from the measured `epoch_seconds`: hours = 200 x epoch_seconds / 3600 per run, times the number of runs. If that does not fit before Dec 2, keep three seeds for the fused and collapsed rows, run the other rows once, and say so on the slide. Half-resolution and 30-epoch rows are screening only.

## Phase 2 ablation rows

All rows share data, resolution, loss, epochs, seed, `--drop-path` and `--head-dropout`. Flags are added to `--model transformer`.

| Row | Flags | Parameters | What it tests |
|---|---|---|---|
| A. Fused | (defaults: `--depth-spectral 2 --depth-spatial 6`) | 488,487 | The model |
| B. Band-collapsed control | `--tokenizer collapsed --depth-spectral 0 --depth-spatial 8` | 491,191 | **Hypothesis condition 2 is A minus B**, no skip in either |
| C. Fused + skip | `--head-skip` | 492,599 | Required head ablation: C minus A |
| D. Collapsed + skip | `--tokenizer collapsed --depth-spectral 0 --depth-spatial 8 --head-skip` | 495,303 | Whether the A-B gap survives the skip (C minus D) |
| E. No spectral attention, mean-pooled groups | `--depth-spectral 0 --depth-spatial 8` | 488,487 | Order-blind lower bound; not the control |
| F. Spectral-only | `--depth-spectral 2 --depth-spatial 0` | 188,583 | RECIPE section 8 item 8 |
| G. No attention, with skip | `--depth-spectral 0 --depth-spatial 0 --head-skip` | 92,727 | How much the per-pixel skip carries without any attention |
| H. No attention, no skip | `--depth-spectral 0 --depth-spatial 0` | 88,615 | Floor for G |
| CNN | `--model cnn` | 571,879 | Second reference |

- The control is matched to 0.55% in parameters and gets the same drop-path schedule. It is not matched in compute: on CPU at batch 8 and 1024 tokens a step took 3.42 s for A and 3.71 s for B. I did not compute GFLOPs.
- The control's bottleneck is 16 channels so its patch conv has the same shape as the band tokenizer's. RECIPE names 224->32; that tokenizer would be about 138k parameters and unmatched.
- Report A, B, C and D together. If C minus D is under 2 points while A minus B is over, say that the skip closes the gap.

## Risks and open questions

1. **The `forward` edit needs a yes or no.** If refused, rows C, D and G cannot run and the required skip ablation is blocked.
2. **Thin classes without the skip.** Rows A and B cannot draw 1 px structure (measured IoU 0.000 on synthetic lines). Filament and video_tape are read only from skip rows.
3. **The fused model is weakly sensitive to band order at random init** (1.2e-04 against 4.4e-03 for the control). Band order reaches it only through the group embedding inside spectral attention, and then a mean pool. If A does not beat B, a learned pool in `SpectralPool` is the first thing to try. Open question for Tim.
4. **The harness optimiser is not the recipe** (lr 1e-3 flat into cosine, weight decay 1e-4 on everything, no warmup, no clip, no Dice). PR 8 is not built.
5. **GPU behaviour is unmeasured:** T4 memory, step time, AMP on CUDA. The CPU figures suggest batch 8 at full resolution fits with room; run 1 confirms or refutes that.
6. **The skip sees the raw cube in [0, 1].** If per-band z-score is added to the loader, the skip input changes scale and the skip rows must be rerun.
7. **`train_fg_miou` is taken in train mode**, before each step's update. With drop-path or dropout on it reads low; the overfit gate runs with both off.
8. **`SpectralBlock` still rejects `mlp_ratio != 4`**, so `--mlp-ratio 2` only builds with `--depth-spectral 0`.
9. **PE scale:** PE row norm 5.66 against pooled tokens at 7.12 at init. Not tuned.
10. **Default PyTorch init** (no truncated-normal 0.02, no zero-init classifier). It trained on synthetic data; untested on real data.
11. **Time.** 19 days to Dec 2, and the 200-epoch, three-seed runs have no hour estimate until run 1 gives `epoch_seconds`.
12. **Stale docs:** `docs/TRANSFORMER-TODO.md` and `README.md` still describe TODO bodies. `docs/CHECKUP-QA.md` lives on `claude/great-cannon-wdl2oo`, not on `main`.

## Verification checklist

- [ ] `PYTHONPATH=src python3 -m pytest -q .` prints `64 passed` (no `xfail`, no `xpass`).
- [ ] `grep -c "nn.MultiheadAttention(\|nn.TransformerEncoder\|import timm" src/sw/models/transformer.py` prints `0`.
- [ ] `git diff main -- src/sw/attention.py` is empty.
- [ ] The diff of `BandTransformer.forward` is one keyword on the last line, approved by the reviewer.
- [ ] Smoke prints `BandTransformer parameters: 488,487 (0.488M)`; with `--head-skip`, `492,599`; the control, `491,191`.
- [ ] Parameter counts on the slides are these measured numbers, not "about 0.49M, designed".
- [ ] Colab run 1 recorded: peak GiB and `epoch_seconds` at batch 8, full resolution.
- [ ] Colab run 2 passed: `train_fg_miou` above 0.95 with `--head-skip --no-aug --no-cosine`.
- [ ] No `--amp` on any run until one epoch of fp32 and AMP train loss have been compared.
- [ ] Every results row lists resolution, epochs, seed, `--drop-path`, `--head-dropout`, skip on or off, and the tokenizer.
- [ ] Hypothesis condition 2 is read from rows A and B at full resolution, 200 epochs, after PR 8.
- [ ] No data, tiffs, zips or checkpoints in any commit.

Files changed by the prototype patch: `src/sw/models/transformer.py`, `src/sw/models/tokenizer.py`, `src/sw/train.py`, `tests/test_transformer.py`, `tests/test_tokenizer.py`.
