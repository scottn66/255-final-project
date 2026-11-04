# Nov 4 checkup: source card

Every number on the deck, where it comes from, and the condition to say next to it. Slide numbers are the 10-slide order in the deck copy. The six links are deck TODO item 11: the paper link goes on each baseline card on slide 4, the other five in that slide's footer. This table is also the content of the backup slide if Jeong asks where a number came from.

## The six links

| # | Link | Backs |
|---|---|---|
| L1 | Paper: <https://arxiv.org/abs/2403.18033> (Casao et al., IROS 2024) | Table IV (every baseline number, per-class IoU, params, GFLOPs). Table I (instance counts). Table II (79.6). Sec III-A (Specim FX17, 224 bands, 900 to 1700 nm, raw 600 by 640). Sec IV-A (resize to 256 by 256). Sec IV-C (label transfer). Sec V-A (metric, 200 epochs, optimizers, LR 1e-3, poly schedule, median-frequency weights, rotation plus flips). |
| L2 | Code: <https://github.com/ferpb/spectralwaste-segmentation> | `scripts/train_model.py`: batch 12, class weights `median_frequency_exp(counts, 0.12)` from RGB-label counts, no pretrained weights loaded, best-val checkpoint. `scripts/dim_reduction.py` for HYPER3. |
| L3 | Dataset: <https://doi.org/10.5281/zenodo.10880544> | `spectralwaste_segmentation.zip`, 23 GB, md5 `ae2c268bb385345f12e55b4389d2d3b9`, CC BY 4.0. `hyper/` cubes 224 by 256 by 256 uint16, `labels_hyper_lt/` masks. The 514 / 167 / 171 split, counted on disk. |
| L4 | SegFormer: <https://arxiv.org/abs/2105.15203> | The SegFormer-B0 card: the paper's transformer baseline. |
| L5 | MiniNet-v2: <https://doi.org/10.1109/TRO.2020.2974099> | The MiniNet-v2 card: the family MiniNetSeg is drawn from. |
| L6 | CMX: <https://arxiv.org/abs/2203.04838> | The CMX-B0 card: RGB plus HSI, context only. |

Our own numbers come from the repo, `main` at 7ea4556: `docs/RESULTS.md` (the run), `src/sw/data.py` (pixel counts, palette), `src/sw/metrics.py` (FG mIoU), `src/sw/train.py` (downsample applies to val masks too), `src/sw/models/tokenizer.py` with `tests/test_transformer.py` (66,624), and `src/sw/models/transformer.py` (spectral block and pool, merged in PR #12).

## Slide by slide

| Slide | On the slide | Source | Say with it |
|---|---|---|---|
| 3 Domain | 224 bands, 900 to 1700 nm | L1 Sec III-A | |
| 3 | 514 training images | L3, counted on disk; not in the paper | official split |
| 3 | under 0.1% of pixels are filament | `src/sw/data.py`: 33,101 / 33,685,504 = 0.098% | of train pixels |
| 3 | six object classes plus background | L1 Table I lists the six; `meta.json` ids 0 to 6 | |
| 4 Prior work | MiniNet-v2 52.8, 0.585M | L1 Table IV, HYPER row | 256 by 256, 200 epochs, batch 12 (L2); split not stated |
| 4 | SegFormer-B0 54.3, about 4.1M | L1 Table IV (4.062M) | same |
| 4 | CMX-B0 58.2 | L1 Table IV, RGB-HYPER hybrid row (11.539M) | RGB plus HSI, context only |
| 4 | filament about 30 to 33 | L1 Table IV: 30.5 MiniNet-v2, 33.1 SegFormer-B0 | name the model |
| 4 | tape weak, if added | L1 Table IV: 28.8 / 21.9 | |
| 4 | about 0.5M from-scratch, untested | our design | designed, not measured; untested on SpectralWaste segmentation |
| 4 | no published INT8 drop | L1 has no INT8 row | |
| 5 Hypothesis | 52.8 / 54.3, about 0.49M | L1 Table IV; hand count under slide 8 | the margin is ours to state |
| 6 Data + metric | 224 by 256 by 256, uint16, band first | L3 files; `src/sw/data.py` | the paper's raw cubes are 600 by 640 (Sec III-A), resized to 256 (Sec IV-A) |
| 6 | 900 to 1700 nm, scaled to [0, 1] | L1 Sec III-A; the loader divides by 65535 | |
| 6 | 514 / 167 / 171 | L3, counted on disk | official split only |
| 6 | background about 82% of train pixels | 27,612,233 / 33,685,504 = 81.97% | of train pixels |
| 6 | filament 33,101 pixels, 0.098% | `src/sw/data.py` | of train pixels |
| 6 | FG mIoU = mean IoU of classes 1 to 6 | `src/sw/metrics.py`; L1 Sec V-A "their mean"; each Table IV row averages to its printed mIoU | same as Table IV; IoU = TP / (TP + FP + FN) over all val pixels |
| 7 Baseline | MiniNetSeg 571,879 params | `python -m sw.models.cnn` | |
| 7 | 39.1, best val FG mIoU, epoch 29 | `docs/RESULTS.md`, row `cnn_ds2` | val; half resolution for cube and mask; 30 epochs; batch 4; one run |
| 7 | per class 55 / 50 / 58 / 29 / 44 / 0 | same row: .552 .495 .578 .286 .436 .000 | background .868 is not in the mean |
| 7 | paper bars full resolution, about 200 epochs | L1 Sec V-A: exactly 200 | "200", not "about 200" |
| 7 | class-weight sentence | L2 hard-codes RGB counts; we use `labels_hyper_lt` counts; under 1% difference at exponent 0.12 (`docs/RECIPE.md` section 6) | |
| 8 Approach | 14 by 16 band tokens, patch 8 | `src/sw/models/tokenizer.py` | 66,624 params, merged (PR #9) |
| 8 | 2 spectral blocks, 6 spatial blocks, D = 64, 4 heads | `BandTransformer` defaults in `src/sw/models/transformer.py` | spectral block and pool merged (PR #12); spatial PE, spatial blocks, head are stubs |
| 8 | about 0.5M parameters | hand count: 66,624 + 8 x 49,984 = 466,496, plus the head | say about 0.49M everywhere, and "designed" |
| 8 | 1024 or 256 tokens | 32 by 32 grid at 256 input; 16 by 16 at 128 | |
| 8 | from-scratch QKV | `src/sw/attention.py`; the test forbids built-in attention | |
| 9 Status | tokenizer 14 groups, 66,624 | `tests/test_transformer.py` locks it | |
| 9 | "five modules still open" | outdated: tokenizer, spectral block and pool merged; three stubs | match the slide 8 wording |
| 10 Next | Phase 1, 2, 3; Dec 2 | `docs/PLAN.md` | INT8 is "planned" |
| B Backup | 79.6; 81 instances; 20 images; MA 56.3 | L1 Table II | agreement between LT and manual labels; "ceiling" is our reading |

## Not in the paper, so label it as ours

- 514 / 167 / 171: counted in the Zenodo zip. Cross-check: 514 x 65,536 = 33,685,504, which equals the sum of the train pixel counts.
- Which split Table IV is scored on: not stated. Ours is val.
- Background excluded from mIoU: inferred from the Table IV arithmetic.
- Batch 12: the official code, not the paper.
- SegFormer-B0 trained from scratch: inferred from the released code (the `init_weights` loader is commented out).
- Nominal wavelengths: `meta.json` ships none; 900 + i x 800 / 223 nm assumes even spacing.
