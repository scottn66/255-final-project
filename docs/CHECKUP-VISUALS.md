# Nov 4 checkup: visual artifacts plan

Written Tue Nov 3, revised Wed Nov 4, 2026, for the 6 PM progress checkup (5 of 20 points, Canvas 7916136). Owner: Scott. The deck is the 10-slide copy in Google Slides; the Nov 3 deck review, `NOV4-META-DECK.md`, and `NOV4-DECK-TODO.md` are the requirements this plan serves.

Jeong treats AI-written presentation content as cheating. So this file, `docs/CHECKUP-QA.md`, and `docs/CHECKUP-SOURCES.md` hold specs, facts, and sources only. Scott plots the figures and writes every word on the slides and in the notes.

`notebooks/04_checkup_figures.ipynb` makes A, B, C, D, and G from the specs below and saves them under `docs/figures/`. Cells A, B, and G need no data; C and D read `SW_ROOT` like notebook 01 and skip themselves without it.

## What the slides have to carry

`docs/PLAN.md` says the checkup wants three things: the CNN score, evidence the transformer trained on real data, and the Phase 2/3 plan. The deck covers the first and third. The second (artifact G) is deferred to Nov 5 by decision on Nov 4; the deck says "no transformer score" and "designed" instead.

Status wording, the same on slides 8, 9, and 10: the band-token transformer is designed; tokenizer merged (PR #9); spectral block and pool in review (PR #12); spatial PE, spatial blocks, and seg head are stubs.

What exists today:

| Fact | Where it lives |
|---|---|
| CNN run `cnn_ds2`: best val FG mIoU 0.391 at epoch 29, per-class IoU for all 7 classes | `docs/RESULTS.md` |
| Per-epoch metrics for that run | Scott's Drive only (`sw_runs/cnn_ds2/metrics.csv`). `notebooks/03_train_cnn.ipynb` on `main` is an empty badge cell. |
| Band tokenizer, 66,624 params, `(B, 1024, 14, 64)` at 256 input | `src/sw/models/tokenizer.py`, merged |
| `SpectralBlock` (wraps `ResidualSelfAttentionBlock` across the 14 groups) and `SpectralPool` (mean) | Open PR #12, head 86dc065 |
| `SpatialPE`, `SpatialBlock`, `SegHead` | Raise `NotImplementedError` |
| One-cube figures: false colour, mask, legend, per-class spectrum, pixel counts | `notebooks/01_look_at_the_data.ipynb`, executed Oct 6 on `train[0]` |
| Train pixel counts per class, class palette | `src/sw/data.py` |

Constraints: the data is 23 GB and on no machine right now, so anything that reads cubes needs a Colab session and a download first. The harness strides both the cube and the mask (`prepare_sample` in `src/sw/train.py`), so the 0.391 was scored on 128 by 128 val masks. Say that with the number.

## Build order for Nov 4

| # | Artifact | Deck slot | Needs data | Time | Cut line |
|---|---|---|---|---|---|
| A | Per-class IoU bar chart (TODO 7) | slide 7 | No | 30 min | Keep |
| B | Architecture strip, current vs goal (TODO 2) | slide 8 | No | 90 min | Keep; simplify to the transformer row if short |
| E | Q&A facts sheet | `docs/CHECKUP-QA.md`, written | No | answers only | Keep |
| F | Source card | `docs/CHECKUP-SOURCES.md`, written; copy to the backup slide | No | 10 min | Keep |
| C | Data example, three panels (TODO 3) | slide 6, or slide 3 for the RGB claim | Yes | 30 min after download | Fall back to the Oct 6 notebook outputs |
| D | Spectral signature chart, 6 classes over 224 bands | slide 3 or 6, not both with C's panel 3 | Yes | 45 min after download | First to drop |
| G | Transformer run evidence | Nov 5 | Yes, plus PR #12 and the three stubs | | Deferred |

## A. Per-class IoU bar chart (slide 7)

Spec, as agreed in the review's figure B:

- Data: `docs/RESULTS.md` row `cnn_ds2`, epoch 29, in percent: film 55.2, basket 49.5, cardboard 57.8, tape 28.6, filament 0.0, bag 43.6.
- Horizontal bars, one per object class, sorted high to low so filament sits last. Percent axis 0 to 100 to match the 39.1 on the card.
- One reference tick per class for MiniNet-v2 HYPER from Table IV: film 61.2, basket 61.0, cardboard 78.8, tape 28.8, filament 30.5, bag 56.3, labelled "paper, full res, 200 epochs". MiniNet-v2 is the fair reference because MiniNetSeg is from the same family.
- A vertical line at 39.1 labelled "FG mIoU, mean of the six bars". Optional second line at 52.8, "paper mean".
- Print "0.0" on the filament bar so it reads as a result, not as missing data.
- Background (86.8) stays off; it is not in the mean.
- Condition line under the chart: MiniNetSeg, 571,879 params, half resolution (cube and mask), 30 epochs, val, best epoch 29, one run.
- Bar colours: the dataset palette (below), so the chart agrees with the mask on slide 6.

What it shows, for the notes: tape already matches the paper (28.6 vs 28.8); the largest gaps are filament (30.5 points) and cardboard (21.0). Filament and cardboard are 63% of the 13.7-point gap. If filament matched the paper's 30.5, the mean would be 44.2.

Save as `docs/figures/per_class_iou.png`, dpi 200, about 10 by 4.5 inches, with the cell or script that made it.

## B. Architecture strip, current vs goal (slide 8)

One left-to-right strip, one box per stage. Show status with box style plus a word, never colour alone: solid = merged, hatched = in review, outline only = stub. Full-resolution shapes on the boxes; footnote the half-resolution grid.

| Stage | Output shape, 256 input | Params | Status |
|---|---|---|---|
| Input cube | (B, 224, 256, 256) | 0 | done (loader) |
| Band tokenizer: 14 groups of 16 bands, one shared Conv2d 16 to 64 (kernel 8, stride 8), LayerNorm, learned 14 by 64 group embedding | (B, 1024, 14, 64) | 66,624 | merged, PR #9 |
| Spectral block x2: attention across the 14 groups inside each patch | (B, 1024, 14, 64) | 49,984 each | in review, PR #12 |
| Spectral pool: mean over groups | (B, 1024, 64) | 0 | in review, PR #12 |
| Spatial PE: fixed 2D sin-cos on the 32 by 32 grid | (B, 1024, 64) | 0 | stub |
| Spatial block x6: attention across the 1,024 patches | (B, 1024, 64) | 49,984 each | stub |
| Seg head: (B, 64, 32, 32) to 7 classes, upsample x8 | (B, 7, 256, 256) | about 0.5k (1x1) to about 21k (recipe head) | stub |

- Mark the axis each attention runs over: 14 band groups in the spectral blocks, 1,024 patches in the spatial blocks. That split is what the hypothesis tests.
- Total under the strip: 66,624 + 8 x 49,984 = 466,496 plus the head, so about 0.47M to 0.49M depending on the head. Label it "designed" until the harness prints a measured count. Use "about 0.49M" on every slide, not "about 0.5M".
- Footnote: `--downsample 2` gives a 16 by 16 grid (256 tokens) for the first Colab run.
- Right edge: cross-entropy with paper-scheme weights on the hyperspectral counts, then FG mIoU over classes 1 to 6.
- Goal boxes, dashed: Phase 2 ablation rows (spectral only, spatial only, both); Phase 3 INT8 as one more column, labelled "planned".
- Lecture tie-ins as small side labels if they fit: tokenizer = patch embedding, group embedding = position embedding (Lecture 1); attention = `nn.Linear` Q, K, V and softmax; seg head and mIoU = Lecture 9.

The CNN control can be a second, thinner row if there is room: 1x1 band mix 224 to 48, separable conv, four stride-2 stages at widths 64, 96, 128, 176 with multi-dilation modules (rates 1, 2, 4, 8), bottleneck, three decoder stages with skips, 1x1 to 7. 571,879 params, merged, scored.

Build in Slides with shapes so it stays editable during the dry run. An SVG in `docs/figures/architecture.svg` is the fallback; it needs no data and no torch.

## C. Data example, three panels (slide 6, or slide 3 for the RGB claim)

Notebook 01 already draws all three pieces for `train[0]`, which has only film, bag, and background. For the slide, pick a cube with filament and overlapping objects, so the panel shows why the label is per pixel. Scanning the mask PNGs costs nothing (no cube reads): loop over `labels_hyper_lt/train`, count classes per mask, and keep the stem. Print the stem on the slide.

1. False colour from bands 56, 112, 168 with the notebook's 2 to 98 percentile stretch. `meta.json` has no wavelengths; with even spacing over 900 to 1700 nm these sit near 1,100, 1,300, and 1,500 nm. Say "approx." or use band numbers.
2. Label mask in the dataset palette, with a legend for all seven classes in the same colours as panel 3.
3. Mean spectrum per class. The notebook averages one image; for the slide, either average each class over all train images or label it "one image". X axis band index or approximate nm, y axis scaled value 0 to 1.

- If two class curves overlap in panel 3, say so; that pair is likely where the model confuses classes.
- 900 to 1700 nm is all infrared, so the cube cannot show what an RGB camera sees. A true RGB comparison needs the dataset's RGB images, which the loader does not read. Do not present the false-colour panel as "what RGB sees".
- Caption: "One train cube, 224 by 256 by 256, uint16 scaled to [0, 1]. Masks are label-transfer masks (`labels_hyper_lt`), 79.6 mIoU against manual labels in the paper's Table II."

Build: one cell at the end of notebook 01 that composes `plt.subplots(1, 3)` with equal heights from the variables already in scope and saves `docs/figures/data_example.png` at dpi 200. Commit the PNG, never the cube. Owner: Yash (his optional task is running notebook 01) or Surya (data slide), Scott otherwise.

## D. Spectral signature chart, 224 bands and 6 materials

Only if C's third panel is "one image" and there is time. Mean spectrum per object class over N train cubes with a faint plus or minus 1 sd band, background as a thin grey line, x axis band index with a second axis in nominal nm (900 + i x 800 / 223, labelled "nominal, even spacing assumed"), the 14 token groups shaded alternately, the three false-colour bands marked. Inset: class pixel share on a log axis (background 82.0%, filament 0.098%).

Build: a Colab loop over `SpectralWasteDataset(split="train")` accumulating per-class sum, sum of squares, and count as (7, 224) arrays. N = 50 sorted stems is about 1.5 GB of reads and a couple of minutes; all 514 is about 15 GB and roughly ten minutes from local disk. Say N on the figure.

## E and F. Q&A facts sheet and source card

Written: `docs/CHECKUP-QA.md` (ten questions, the six most likely first, facts and traps only, blank answer lines) and `docs/CHECKUP-SOURCES.md` (the six links from TODO item 11, every number on the deck mapped to its source and its condition label). The source card's slide-by-slide table is the backup slide content if a number is challenged.

## G. Transformer run evidence (Nov 5)

- Figure: val FG mIoU per epoch for `runs/transformer` next to the `cnn_ds2` curve, plus train loss, from each run's `metrics.csv`. Any run length counts. Caption carries the run command.
- Text box: the harness startup lines verbatim: model class, parameter count, class-weight vector.
- Order: review and merge PR #12; fill `SpatialPE`, `SpatialBlock`, `SegHead` (`docs/TRANSFORMER-TODO.md`); `pytest -q` green; `python -m sw.train --model transformer --smoke --device cpu`; then the Colab command in that file's step 8.
- Also: copy `cnn_ds2/metrics.csv` from Drive into `docs/metrics/cnn_ds2.csv` (`docs/runs/` would be caught by the `runs/` ignore rule; the CSV is a few KB) and restore the real cells of notebook 03, so the 0.391 run has per-epoch evidence on GitHub.

## Shared style rules

- Class colours everywhere are the dataset palette from `src/sw/data.py`: background #000000, film #DAF706, basket #33DDFF, cardboard #3432DD, video_tape #CA98C3, filament #008000, bag #FFA500. The bar chart, the mask, and the spectra then agree.
- Status is shown by box style plus a word, never colour alone.
- IoU is in percent on the slides (39.1, 52.8) because the deck already uses percent; `docs/RESULTS.md` keeps fractions. Never mix the two on one axis.
- Every chart caption states split, resolution, epochs, and parameter count. Val vs test is the first thing Jeong will ask.
- One message per figure. Fonts at 18 pt or larger. Export PNG at dpi 200, SVG for the diagram. 16:9.
- Figures go in `docs/figures/` with the cell or script that made them. Cubes, zips, and checkpoints never do.

## Nov 4 schedule

| When | Do |
|---|---|
| Morning | Plot A and build B from the specs. Fill slides 1, 2, 5 and the backup; merge slides 9 and 10; delete "five modules open". |
| Midday | C if the Colab download is up; otherwise the committed notebook 01 outputs with "one image" said out loud. D only after C. |
| By 4 PM | Trim notes to the review's per-slide word budgets (about 680 words for 5:00). Dry run with the Q&A sheet. Delete every dashed box. |
| Nov 5 | G. |
