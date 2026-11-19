# Input-blind transformer: diagnosis and gates

## Symptom

On 16+ training cubes at `lr 1e-3`, `BandTransformer` (2 spectral / 6 spatial,
488,487 params) stays background-only. Training loss sits at the input-blind
class-prior loss (~0.79 on 16 cubes, ~0.81 on 64). Four foreground-rich cubes
learn by ~150–216 steps. At `lr 3e-4` on 64 cubes the model only escapes after
~2.3–3.4k steps. Warmup + grad clipping at `1e-3` did not help; `1e-3` also
produced late collapses. The size-matched CNN (`MiniNetSeg`, BatchNorm) learns
on the same raw `[0, 1]` input (val FG mIoU 0.391).

## Diagnosis (code review)

Confirmed in `BandTokenizer` / `IndexBandPE` / `SegHead` / harness:

1. **Shared per-band offset + per-token LayerNorm (most likely).** Cubes are
   `uint16 / 65535` only (`sw.data`). Real train cubes sit near mean ~0.23 with
   spatial std ~0.14. The shared `Conv2d(16→64, k8 s8)` turns that offset into
   a near-constant direction per group; `LayerNorm` over `D` keeps the
   direction and equalises token norms, so spatial variation is a small
   fraction of RMS (~0.18 after tokenizer+PE, ~0.30 at head input; logits
   spatial std ~2% of magnitude).
2. **IndexBandPE at `N(0, 1)` (high).** PE is added *after* LayerNorm, is
   position-independent across the spatial grid, and has RMS comparable to the
   normalised tokens, which further dilutes the spatial residual.
3. **SegHead classifier bias ≈ no class prior (medium-high).** `Conv2d` bias
   starts as a small uniform draw, not `log` priors. With ~81% background
   pixels, “always background” stays the cheapest early CE fit when features
   barely vary in space.
4. **Mild paper class weights (medium).** Median-frequency `**0.12` keeps
   weights near 1, so the class prior remains an easy attractor.
5. **Spectral mean-pool (lower).** Averages group tokens before spatial
   blocks; secondary to (1)–(2).
6. **Augmentation / PE add order / SpatialPE (low).** Same aug as the CNN;
   fixed 2D sin-cos SpatialPE is not the early failure mode.

CNN BatchNorm removes the shared offset; the transformer’s LayerNorm does not.

Nothing in the review contradicted the measured ratios. Contradictions to
watch for in the A/B: if `--standardize` does not move first-step train FG
mIoU while PE scale does, prefer hypothesis (2); if neither moves loss off
the prior, revisit loss/weights and lr.

## Flags (all default to today’s behaviour)

| Flag | Default | Effect |
|------|---------|--------|
| `--standardize` | off | Per-band mean/std from **train only**, cached (default `{out-dir}/band_stats.pt`), applied to train/val/test. Errors if stats would be labeled val/test. |
| `--stats-path PATH` | `{out-dir}/band_stats.pt` | Cache location (keep out of git). |
| `--pe-init-std S` | `1.0` | Scale `IndexBandPE` after its `N(0,1)` draw. Try `0.02` with standardise. |
| `--head-prior-bias` | off | SegHead final conv bias ← `log` train class priors. |
| `--lr` | `1e-3` | **Unchanged.** For the transformer on raw cubes, prefer `3e-4`. |

Diagnostic (no dataset required):

```bash
PYTHONPATH=src python scripts/diag_input_blind.py
PYTHONPATH=src python scripts/diag_input_blind.py --standardize --pe-init-std 0.02
```

## Local A/B (Scott, 64 train / 8 val)

Protocol: 4000 steps, `--downsample 2`, batch 4, aug on, **constant lr**
(`--no-cosine`), **no early stop**, `--model transformer` unless noted.
Metric: first step where train FG mIoU ≥ 0.05; final train loss vs input-blind
prior loss.

Grid:

- lr `{1e-3, 3e-4}` × `{raw, --standardize, --standardize --pe-init-std 0.02}`
- Optional: `--standardize --lr 3e-4` plus warmup **if** `--warmup-steps` exists
  (it does **not** on this branch; skip until the optimiser-recipe PR lands)
- CNN control: `--model cnn --standardize` at the CNN’s usual lr

Example:

```bash
PYTHONPATH=src python -m sw.train --model transformer --downsample 2 \
  --batch-size 4 --max-train-steps 4000 --no-cosine \
  --lr 3e-4 --standardize --pe-init-std 0.02 \
  --out-dir runs/tf_std_pe002_lr3e4
```

## Reporting note

Per-band standardisation **deviates from the paper’s `[0, 1]` scaling**. Any
result table that uses it must say so, and both compared rows in a pair must
use the same preprocessing.
