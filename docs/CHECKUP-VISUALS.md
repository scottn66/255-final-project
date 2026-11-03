# Nov 4 checkup: visual artifacts plan

Written Tue Nov 3, 2026 for the Wed Nov 4, 6 PM progress checkup (5 of 20 points, Canvas 7916136). Owner: Scott, with the optional teammate asks from `docs/PLAN.md`. This is a build plan for the figures and prep sheets. It is not the deck.

## What the slides have to carry

`docs/PLAN.md` says the checkup wants three things: the CNN score, evidence the transformer trained on real data, and the Phase 2/3 plan. The six proposed artifacts cover the first and third well. None of them is the second. Artifact G below closes that gap, and it is the only one that depends on code that is not merged yet.

What exists today, so nothing below is invented:

| Fact | Where it lives |
|---|---|
| CNN run `cnn_ds2`: best val FG mIoU 0.391 at epoch 29, per-class IoU for all 7 classes | `docs/RESULTS.md` |
| Per-epoch metrics for that run | Only on Scott's Drive (`sw_runs/cnn_ds2/metrics.csv`). `notebooks/03_train_cnn.ipynb` on `main` is an empty badge cell, so GitHub has no per-epoch evidence yet. |
| Band tokenizer, 66,624 params, `(B, 1024, 14, 64)` at 256 input | `src/sw/models/tokenizer.py`, merged (PR #9, Tim) |
| `SpectralBlock` and `SpectralPool` (mean over groups) | Open PR #12 (Tim, Oct 28), branch `AddingSpectralBlockPool` |
| `SpatialPE`, `SpatialBlock`, `SegHead` | Still raise `NotImplementedError` (Scott) |
| One-cube figures: false colour, mask, legend, per-class spectrum, pixel counts | `notebooks/01_look_at_the_data.ipynb`, executed Oct 6 |
| Train pixel counts per class, class palette | `src/sw/data.py` |

Two constraints shape the order. The data is 23 GB and is not on any machine right now, so anything that reads cubes needs a Colab session and a download first. Everything else can be built from numbers already in the repo, with no data and no GPU.

## Build order

| # | Artifact | Covers | Needs data | Build where | Time | Cut line |
|---|---|---|---|---|---|---|
| A | Per-class IoU bar chart, ours vs paper | CNN score, next steps | No | matplotlib anywhere, or a Slides chart | 30 min | Keep |
| B | Architecture diagram, current vs goal | Approach, "what have you built" | No | Slides shapes (editable), SVG fallback | 90 min | Keep, simplify to one row if short |
| E | Q&A prep sheet | The 1:30 of questions | No | `docs/CHECKUP-QA.md` | 60 min | Keep |
| F | Source card | Any number Jeong asks about | No | `docs/CHECKUP-SOURCES.md` + last backup slide | 30 min | Keep |
| G | Transformer run evidence (curve + startup printout) | Transformer trained on real data | Yes, plus PR #12 and the three open modules | Colab, then matplotlib | 20 min once a run exists | Keep; any run length counts |
| C | Data example figure (false colour, mask, spectra) | Dataset and metrics | Yes | Colab, one new cell in notebook 01 | 30 min after download | Fall back to the Oct 6 notebook outputs |
| D | Spectral signature chart, 6 classes over 224 bands | Dataset, why band tokens | Yes | Colab, one loop over train | 45 min after download | First to drop |

Build A, F, E tonight with no data. Start the Colab download in the background at the same time. B and G are Nov 4 morning. C and D only if the Colab session is up by noon.

## A. Per-class IoU bar chart for the 0.391 run

One chart that shows the score and the rare-class problem at once.

- **Series 1 (filled bars, class colours):** our `cnn_ds2` val IoU per object class: film 0.552, basket 0.495, cardboard 0.578, video_tape 0.286, filament 0.000, bag 0.436.
- **Series 2 (hollow grey bars beside each):** the paper's MiniNet-v2 HYPER row, Table IV, as fractions: 0.612, 0.610, 0.788, 0.288, 0.305, 0.563.
- **Background** as a seventh bar on the far left, grey and hatched, 0.868, labelled "not in the mean".
- **Two dashed lines:** 0.391 (ours) and 0.528 (paper), each labelled at the right edge.
- **Annotation on filament:** "0.000, 0.098% of train pixels".
- **Caption, verbatim on the slide:** "Ours: official val split, half resolution (128 by 128), 30 epochs, 571,879 params, one run. Paper: its reported numbers, 256 by 256, 200 epochs, 0.585M params. Mean over the 6 object classes, background excluded, as in the paper."
- Y axis "IoU", 0 to 1. Class order is id order (1 to 6) so it matches the mask palette and the legend in artifact C.

Why this chart answers the gap question: the six per-class gaps are film 6.0, basket 11.5, cardboard 21.0, video_tape 0.2, filament 30.5, bag 12.7 points. Filament alone is 37% of the 13.7-point gap and cardboard is 26%. If filament matched the paper's 30.5, the mean would be 0.442. Those three sentences go in the speaker notes, not on the chart.

Build: a 40-line matplotlib script. The numbers are all in `docs/RESULTS.md` and the paper's Table IV. Save as `docs/figures/per_class_iou.png` at dpi 200, 10 by 4.5 inches. If the chart is drawn as a native Slides chart instead, keep the same two series and the same caption.

## B. Architecture diagram, current vs goal

Draw it from the module contracts in `src/sw/models/transformer.py` and `src/sw/models/cnn.py`. Two rows with a shared input on the left and a shared loss and metric on the right. A colour legend for status: merged on `main`, in an open PR, still open, Phase 2/3 goal (dashed).

**Row 1, CNN control (has a score).** Input cube (224, 256, 256) → 1x1 band mix 224→48 → separable 3x3 → four stride-2 stages at widths 64, 96, 128, 176, each with a multi-dilation module (rates 1, 2, 4, 8) → bottleneck module → three decoder stages with skips → 1x1 to 7 → logits (7, 256, 256). Label: `MiniNetSeg`, 571,879 params, val FG mIoU 0.391 (half res). Status: merged.

**Row 2, band-token transformer (being built).**

| Box | Shape out | Contract | Status |
|---|---|---|---|
| Tokenizer: 14 groups of 16 bands, one shared Conv2d 16→64 with kernel 8 stride 8, LayerNorm, plus learned group embedding (14 by 64) | (B, 1024, 14, 64), grid 32 by 32 | `BandTokenizer`, 66,624 params, locked by test | merged (PR #9) |
| SpectralBlock x2: pre-norm attention over the 14 groups inside each spatial token | (B, 1024, 14, 64) | `sw.attention` MHSA, 4 heads, MLP 4x | open PR #12 |
| SpectralPool: mean over groups | (B, 1024, 64) | no params | open PR #12 |
| SpatialPE: fixed 2D sin-cos on the 32 by 32 grid | (B, 1024, 64) | no params | open (Scott) |
| SpatialBlock x6: pre-norm attention over the 1024 spatial tokens | (B, 1024, 64) | `sw.attention` MHSA, 4 heads, MLP 4x | open (Scott) |
| SegHead: reshape to (64, 32, 32), 1x1 to 7, bilinear up to the input size | (B, 7, 256, 256) | | open (Scott) |

Width D = 64, 4 heads, recipe estimate about 0.5M params total. Say "estimate" until `python -m sw.train --model transformer --smoke` prints the real count. For the half-resolution run the grid is 16 by 16, so 256 tokens.

**Shared right side.** Cross-entropy with the paper's median-frequency weights computed on the hyperspectral train counts (filament 1.516), then FG mIoU over classes 1 to 6.

**Goal boxes, dashed.** Phase 2: three rows in one table, spectral-only (0 spatial blocks), spatial-only, fused; the cross-attention block for fusion already exists in `src/sw/attention.py`. Phase 3: INT8 export and the mIoU drop as one more column.

**Lecture tie-ins as small side labels.** Tokenizer = patch embedding, group embedding = position embedding (Lecture 1 ViT components). Attention = `nn.Linear` Q, K, V and softmax; the test in `tests/test_transformer.py` fails if `nn.MultiheadAttention` or `nn.TransformerEncoder` appears. Seg head and mIoU = Lecture 9.

Build in Slides with shapes so it can be edited during the dry run. If an SVG is easier to iterate, draw it first as `docs/figures/architecture.svg` and paste the export in; the SVG needs no data and no torch.

## C. Data example figure

One cube as three panels: false colour, mask with the class legend, mean spectrum of each class present.

Notebook 01 already makes all three pieces, but for `train[0]`, which contains only film, bag, and background. For the slide, pick a cube with more classes. Scanning the mask PNGs costs nothing (no cube reads): loop over `labels_hyper_lt/train`, count classes per mask, and take the one with the most classes, preferring one that has filament. Print its stem on the slide so the number is traceable.

Spec:

- Panel 1: false colour from bands 56, 112, 168 (the notebook's `channels // 4` picks), each stretched to its 2nd to 98th percentile. Title "bands 56 / 112 / 168, about 1101 / 1302 / 1503 nm nominal".
- Panel 2: mask in the dataset palette (`PALETTE` in `src/sw/data.py`), legend with id and name.
- Panel 3: mean spectrum per class present, line colour = class colour, x = band index with a second axis in nominal nm, y = reflectance in [0, 1].
- Caption: "One train cube, 224 by 256 by 256, uint16 scaled to [0, 1]. Masks are label-transfer masks (`labels_hyper_lt`), 79.6 mIoU against manual labels in the paper's Table II."

Build: add one cell at the end of notebook 01 that composes `plt.subplots(1, 3, figsize=(15, 4.5))` from the variables already in scope and calls `savefig("docs/figures/data_example.png", dpi=200, bbox_inches="tight")`. Commit the PNG (the `.gitignore` keeps notebook and docs images committable), never the cube. Owner: Yash (his optional task is to run notebook 01) or Surya (data slide), Scott otherwise. Fallback if the Colab session is late: the Oct 6 outputs already in the committed notebook, with "one image, two object classes" said out loud.

## D. Spectral signature chart, 224 bands and 6 materials

Mean spectrum per class over many train cubes, so every class has a curve.

- One line per object class (class colours), mean reflectance over all pixels of that class across N train cubes, with a faint ±1 sd band. Background as a thin grey line.
- X axis: band index 0 to 223, with a second axis in nominal nm. `meta.json` ships no wavelengths, so use 900 + i × 800/223 (about 3.59 nm per band) and write "nominal FX17 grid" on the axis.
- Shade the 14 token groups (16 bands, about 57 nm each) alternately, and mark the three false-colour bands. That makes this one figure carry both "what the data looks like" and "why we tokenize bands in groups".
- Inset or companion bar: class pixel share on a log axis from `HYPER_LT_TRAIN_PIXEL_COUNTS` (background 82.0%, filament 0.098%).

Build: on Colab, loop over `SpectralWasteDataset(split="train")`, and for each cube accumulate per-class sum, sum of squares, and pixel count as (7, 224) arrays. N = 50 sorted stems is about 1.5 GB of reads and a couple of minutes; the full 514 is about 15 GB and roughly ten minutes from local disk. Say N on the figure. Save `docs/figures/band_signatures.png`. This is the first artifact to drop if time runs out, because artifact C's third panel already shows spectra for one image.

## E. Q&A prep sheet

A page with the six to eight questions Jeong is most likely to ask. For each: the facts with their source, the trap to avoid, and a blank line where Scott writes the spoken answer. Write it as `docs/CHECKUP-QA.md`. The facts are already collected here; only the answers are missing.

1. **Why is background left out of mIoU?** The paper (Section V-A) reports "their mean (mIoU)" over per-class IoU and never says whether background is in it, but every printed mIoU in Table IV equals the mean of its six object-class values (MiniNet-v2 HYPER: (61.2 + 61.0 + 78.8 + 28.8 + 30.5 + 56.3) / 6 = 52.77). Background is 82.0% of train pixels. Including it would move us from 0.391 to 0.459 while changing nothing about the objects. `src/sw/metrics.py` takes the nanmean over classes 1 to 6, and an all-background predictor scores 0. Trap: do not say "the paper excludes it"; say "the paper's numbers are a six-class mean, so we match it".
2. **Why is 39.1 so far below 52.8?** Different conditions: our number is val, theirs is the reported number; we ran half resolution (128 by 128), they ran 256; 30 epochs vs 200; batch 4 vs 12; one run each. Per class, filament (0 vs 30.5) and cardboard (57.8 vs 78.8) are 63% of the gap; video_tape is already tied. Thin classes lose the most at half resolution. Trap: do not promise 52.8; `docs/PLAN.md` says below 52.8 is a normal from-scratch result on 514 images.
3. **Why a transformer on only 514 cubes?** The paper's SegFormer-B0 HYPER (54.3) is a 4.06M-param transformer, and the released code loads no pretrained weights (the `init_weights` loader is commented out), so a from-scratch transformer already trains on this split. Ours is about 0.5M. Each cube is 65,536 labelled pixels times 224 bands, so pixel-level supervision is plentiful; spectral attention is over 14 tokens and spatial attention over 1024. The recipe's small-data choices: the paper's augmentation, stochastic depth 0.1, weight decay 0.05. Trap: "from scratch" is inferred from the code, not stated in the paper; say so.
4. **What is actually new?** The honest claim: a spectral-then-spatial band-token transformer under 0.6M params, built from our own attention and tokenizer, on hyperspectral only, with the Phase 2 ablation (spectral-only, spatial-only, fused) and the Phase 3 INT8 mIoU-drop measurement, none of which the paper reports. Trap: the wavelength encoding is technique adoption, not the novelty (`docs/RECIPE.md`); SSFT at 0.516M is classification, not segmentation; do not compare against CMX-B0's 58.2, which uses RGB plus hyperspectral and 11.5M params.
5. **Why is filament exactly zero?** 33,101 train pixels, 0.098% of the total, 111 instances in the paper's Table I. The paper's weight scheme at exponent 0.12 only lifts it to 1.516; at 0.5 it would be 5.66. Half resolution thins it further. Next steps in order: full resolution, exponent 0.5, add Dice, Lovász fine-tune, and the pending count of filament images in val and test (`docs/PLAN.md`, known issue).
6. **Is the attention really from scratch?** Yes: `src/sw/attention.py` is `nn.Linear` for Q, K, V, O plus softmax. `tests/test_transformer.py` fails if `nn.MultiheadAttention`, `nn.TransformerEncoder`, or `timm` appears in the model file. Jeong allowed built-in pieces; a built-in `nn.TransformerEncoderLayer` version is kept only as a comparison row.
7. **Why these class weights and not the paper's?** `meta.json` pixel counts are RGB-label counts; we weight with the hyperspectral `labels_hyper_lt` counts. At exponent 0.12 the two differ by under 1% (filament 1.525 vs 1.516), so this is protocol correctness, not a score lever.
8. **Where do the hyperspectral labels come from?** Label transfer from the RGB masks (paper Section IV-C: contour points matched with COTR, one affine per component). Table II scores those masks at 79.6 mIoU against 81 hand-labelled instances in 20 images, so about 20 points of label noise is the ceiling nobody beats.

One more trap, outside the paper: the GitHub repository description still reads "Group 10 ... wavelength-as-input ... INT8 on Mobilint ARIES. See docs/research/", which is the abandoned September pivot. Fix the description before Jeong opens the repo.

## F. Source card

A table, `docs/CHECKUP-SOURCES.md`, duplicated as the last backup slide, with one row per cited number: slide, number, source, exact location. The six links and what each one backs:

| Link | Backs |
|---|---|
| Paper, arXiv 2403.18033 | Table IV (all per-class and mIoU rows, params, GFLOPs); Table I (instance counts); Table II (79.6 label-transfer mIoU); Section III-A (FX17, 224 bands, 900 to 1700 nm, raw 600 by 640); Section IV-A (resize to 256 by 256); Section IV-C (label transfer); Section V-A (metric, 200 epochs, optimizers, LR 1e-3, poly schedule, median-frequency weights, rotation ±30 and flips) |
| Official code, github.com/ferpb/spectralwaste-segmentation | `scripts/train_model.py`: batch 12, weights `median_frequency_exp(counts, 0.12)` hard-coded from RGB counts, no pretrained weights loaded, best-val checkpoint; `scripts/dim_reduction.py` for HYPER3 |
| Zenodo record 10880544 | `spectralwaste_segmentation.zip`, 23 GB, md5 `ae2c268bb385345f12e55b4389d2d3b9`, CC BY 4.0, preprocessed 256 by 256 cubes and `labels_hyper_lt` masks |
| Our repo, `docs/RESULTS.md`, `src/sw/metrics.py`, `src/sw/data.py` | The 0.391 row and its per-class IoU; the FG mIoU definition; the hyperspectral train pixel counts and weights |
| Canvas 7916136 | The checkup rubric: 5 of 20 points, the three required items |
| Lecture 1 and Lecture 9 slides | ViT components (patch and position embedding); segmentation and mIoU |

Numbers that are not in the paper and must be labelled as ours: the 514 / 167 / 171 split (counted on disk; 514 × 65,536 = 33,685,504, which equals the sum of the train pixel counts, so both numbers cross-check), whether Table IV is the test split (not stated; our 0.391 is val, say so), and background exclusion (inferred by the arithmetic in Q&A item 1). If a different six links were intended, keep the "backs" column and swap the link.

## G. Transformer run evidence (the missing artifact)

The checkup asks for evidence that the transformer trained on real data. The figure is cheap; the run is the critical path.

- **Figure:** val FG mIoU per epoch for `runs/transformer` next to the `cnn_ds2` curve, plus train loss per epoch, from each run's `metrics.csv`. Even five epochs is evidence. Label the x axis in epochs and put the run command in the caption (`--model transformer --downsample 2 --epochs 30 --batch-size 4`).
- **Text box:** the harness startup lines, verbatim: the model class name, the parameter count, and the class-weight vector. That is what proves it is our model on the real split.
- **Prerequisites, in order:** review and merge PR #12; fill `SpatialPE`, `SpatialBlock`, `SegHead` (the order and tests are in `docs/TRANSFORMER-TODO.md`); `pytest -q` green; `python -m sw.train --model transformer --smoke --device cpu` on the synthetic root; then the Colab command in `docs/TRANSFORMER-TODO.md` step 8.
- **Also do today:** copy `cnn_ds2/metrics.csv` from Drive into `docs/runs/cnn_ds2_metrics.csv` (`runs/` is gitignored; the CSV is a few KB) so the CNN curve can be plotted and the 0.391 run has per-epoch evidence on GitHub. Notebook 03 should get its real cells back at the same time.

## Shared style rules

- Class colours everywhere are the dataset palette from `src/sw/data.py`: background #000000, film #DAF706, basket #33DDFF, cardboard #3432DD, video_tape #CA98C3, filament #008000, bag #FFA500. Then the bar chart, the mask, and the spectra all agree.
- IoU is a fraction on our charts (0.391), matching `docs/RESULTS.md`. Convert the paper's percentages once (52.8 → 0.528) and never mix the two on one axis.
- Every chart caption states split, resolution, epochs, and parameter count. Val vs test is the first thing Jeong will ask.
- One message per figure. Fonts at 18 pt or larger on the slide. Export PNG at dpi 200 and SVG for the diagram. 16:9 proportions.
- Figures go in `docs/figures/` with the cell or script that made them, so a teammate can re-run them. Cubes, zips, and checkpoints never do.

## Schedule

| When | Do | Blocks on |
|---|---|---|
| Nov 3 evening | A (chart from numbers), F (source card), E (Q&A facts, then answers). Start the Colab download. Review PR #12. | Nothing |
| Nov 4 morning | B (diagram in Slides). Fill the three open transformer modules, smoke test, start the Colab run. C once the data is up. | PR #12 merged for the run |
| Nov 4 early afternoon | G from whatever the run has logged. D if the Colab session is still up. Assemble the deck. | The run |
| Nov 4 by 4 PM | Dry run with the Q&A sheet. Freeze. | |

Drop order if the day runs short: D first, then C (use the committed notebook outputs), then simplify B to the transformer row only. A, E, F, and G stay.
