# Band-token transformer: fill-in order

`src/sw/models/transformer.py` wires the model. Five forwards raise `NotImplementedError`. `BandTransformer.forward` is already written and only calls those pieces. Attention inside the blocks is `src/sw/attention.py` (`nn.Linear` and softmax). Do not call `nn.MultiheadAttention`.

The names below are suggestions. `docs/PLAN.md` still has Scott on the critical path. The split is so two people can fill different classes.

While a body still raises, its test is an expected failure (`xfail`). A wrong shape is a normal failure as soon as the body exists. A matching shape shows as `XPASS`, and the suite stays green. Delete that test's `pytest.mark.xfail` line once it XPASSes if you want the report to say passed.

Run from the repo root. `pytest.ini` already puts `src` on the path.

## Order

### 1. SpectralBlock — suggestion: Tim

Pre-norm attention across the G band groups. `(B, N, G, D) -> (B, N, G, D)`.

```bash
pytest -q tests/test_transformer.py::test_spectral_block_shape
```

### 2. SpectralPool — suggestion: Tim

Drop G. `(B, N, G, D) -> (B, N, D)`. The default written on the class is a mean over G (no parameters). A learned token is the other option.

```bash
pytest -q tests/test_transformer.py::test_spectral_pool_shape
```

### 3. SpatialPE — suggestion: Scott

Add a 2D embedding on the h by w grid. `(B, N, D) -> (B, N, D)` with `N = h * w`. The default written on the class is fixed 2D sin-cos (no parameters). `src/sw/pe.py` is the wavelength table inside the tokenizer, not this grid.

```bash
pytest -q tests/test_transformer.py::test_spatial_pe_shape
```

### 4. SpatialBlock — suggestion: Scott

Pre-norm attention across the N spatial tokens. `(B, N, D) -> (B, N, D)`.

```bash
pytest -q tests/test_transformer.py::test_spatial_block_shape
```

### 5. SegHead — suggestion: Scott

`(B, N, D)` reshaped to `(B, D, h, w)`, then logits upsampled to `(B, 7, H, W)`. `out_size` is the image the tokenizer saw, including sizes that are not a multiple of the patch.

```bash
pytest -q tests/test_transformer.py::test_seg_head_shape
```

### 6. End-to-end shape

Needs all five. A 30 by 40 input is not a multiple of patch 8, so the logits have to come back as `(2, 7, 30, 40)`.

```bash
pytest -q tests/test_transformer.py::test_band_transformer_shape
```

### 7. Smoke

Needs all five, then the harness.

```bash
pytest -q tests/test_transformer.py::test_transformer_smoke
PYTHONPATH=src python -m sw.train --model transformer --smoke --device cpu
```

The CLI reads `SW_ROOT` (or `--data-root`): a folder with `hyper/` and `labels_hyper_lt/`. `--smoke` is one epoch, batch 2, two optimizer steps, four train images, two val images, downsample 4. The number it prints is a wiring check, not a score. `--model` defaults to `cnn`, and that path is unchanged.

### 8. Colab

Run this only after smoke is a real pass. Half resolution, same idea as the CNN checkup run. A 256 cube with `--downsample 2` is 128, patch 8 is a 16 by 16 grid (256 tokens).

```bash
PYTHONPATH=src python -m sw.train --model transformer --downsample 2 \
  --epochs 30 --batch-size 4 --num-workers 2 --device cuda \
  --out-dir runs/transformer
```

Defaults are the recipe's small model: 2 spectral blocks, 6 spatial blocks, D=64, 4 heads, MLP ratio 4, group 16, patch 8. The harness prints the parameter count at startup. The recipe's formula, once the bodies match a pre-norm block, is about 50k parameters per block and about 0.5M overall. Re-measure. Do not treat that formula as the test.

## Counts the tests lock

`test_band_transformer_wiring_and_tokenizer_param_count` expects the tokenizer submodule at 66,624 parameters (conv 16→64, k8 s8, plus LayerNorm, plus the 14 by 64 group embedding). Pool, position, block, and head counts stay unlocked.

## Interface notes

- The tokenizer returns `(B, N, G, D)` and `(h, w)`. Its class docstring says `(B*h*w, G, D)`. The code and `tests/test_tokenizer.py` are the contract this file uses.
- `MultiHeadSelfAttention` and `ResidualSelfAttentionBlock` return `(output, weights)`. Spectral and spatial blocks return the tensor only.
- `ResidualSelfAttentionBlock` fixes the MLP at `4 * dim`. Another `mlp_ratio` needs `FeedForward` with hidden width `dim * mlp_ratio`.
- `dim` must be divisible by `heads`. The default pair is 64 and 4.
