# Nov 4 checkup: visual artifacts plan

Written Tue Nov 3, revised Wed Nov 4, 2026, for the 6 PM progress checkup (5 of 20 points, Canvas 7916136). Owner: Scott. The deck is the 10-slide copy in Google Slides; the Nov 3 deck review, `NOV4-META-DECK.md`, and `NOV4-DECK-TODO.md` are the requirements this plan serves.

Jeong treats AI-written presentation content as cheating. So this file, `docs/CHECKUP-QA.md`, and `docs/CHECKUP-SOURCES.md` hold specs, facts, and sources only. Scott plots the figures and writes every word on the slides and in the notes.

`notebooks/04_checkup_figures.ipynb` makes A, B, C, D, and G from the specs below and saves them under `docs/figures/`. Cells A, B, and G need no data; C and D read `SW_ROOT` like notebook 01 and skip themselves without it.

## What the slides have to carry

`docs/PLAN.md` says the checkup wants three things: the CNN score, evidence the transformer trained on real data, and the Phase 2/3 plan. The deck covers the first and third. The second (artifact G) is deferred to Nov 5 by decision on Nov 4; the deck says "no transformer score" and "designed" instead.

Status wording, the same on slides 8, 9, and 10: the band-token transformer is designed; tokenizer merged (PR #9); spectral block and pool merged (PR #12, Nov 4); spatial PE, spatial blocks, and seg head are stubs.

What exists today:

| Fact | Where it lives |
|---|---|
| CNN run `cnn_ds2`: best val FG mIoU 0.391 at epoch 29, per-class IoU for all 7 classes | `docs/RESULTS.md` |
| Per-epoch metrics for that run | Scott's Drive only (`sw_runs/cnn_ds2/metrics.csv`). `notebooks/03_train_cnn.ipynb` on `main` is an empty badge cell. |
| Band tokenizer, 66,624 params, `(B, 1024, 14, 64)` at 256 input | `src/sw/models/tokenizer.py`, merged |
| `SpectralBlock` (wraps `ResidualSelfAttentionBlock` across the 14 groups) and `SpectralPool` (mean) | Merged into `main` (PR #12, Nov 4) |
| `SpatialPE`, `SpatialBlock`, `SegHead` | Raise `NotImplementedError` |
| One-cube figures: false colour, mask, legend, per-class spectrum, pixel counts | `notebooks/01_look_at_the_data.ipynb`, executed Oct 6 on `train[0]` |
| Train pixel counts per class, class palette | `src/sw/data.py` |

Constraints: the data is 23 GB and on no machine right now, so anything that reads cubes needs a Colab session and a download first. The harness strides both the cube and the mask (`prepare_sample` in `src/sw/train.py`), so the 0.391 was scored on 128 by 128 val masks. Say that with the number.

## Build order for Nov 4

| # | Artifact | Deck slot | Needs data | Time | Cut line |
|---|---|---|---|---|---|
| A | Per-class IoU, grouped bars with the gap (TODO 7) | slide 7 | No | done, 8397ff3 | Keep |
| B | Architecture strip, current vs goal (TODO 2) | slide 8 | No | done, 8397ff3 | Keep |
| E | Q&A facts sheet | `docs/CHECKUP-QA.md`, written | No | answers only | Keep |
| F | Source card | `docs/CHECKUP-SOURCES.md`, written; copy to the backup slide | No | 10 min | Keep |
| C | Data example, three panels (TODO 3) | slide 6, or slide 3 for the RGB claim | Yes | done, 8397ff3 | Keep |
| D | Mean spectrum per class, 8 richest cubes each | slide 3 or 6, not both with C's panel 3 | Yes | done, 8397ff3 | Keep |
| G | Transformer run evidence | Nov 5 | Yes, plus the three stubs | | Deferred |

## A. Per-class IoU (slide 7)

Drawn by notebook 04, cell A, committed as `docs/figures/per_class_iou.png` (8397ff3):

- Grouped horizontal bars per object class. The upper bar is ours in the class colour, the lower bar is the paper's MiniNet-v2 HYPER row in light grey, both with the value printed. MiniNet-v2 is the fair reference because MiniNetSeg is from the same family.
- Classes are sorted by the gap, paper minus ours, and that gap is printed on the right edge in points: filament 30.5, cardboard 21.0, bag 12.7, basket 11.5, film 6.0, tape 0.2.
- The zero filament bar gets a small green marker at the axis and a bold "0.0", so it reads as a result.
- The title's second line states the claim: FG mIoU 39.1 vs paper 52.8, filament and cardboard are 63% of the gap.
- Footer: the condition line (MiniNetSeg, 571,879 params; half resolution for cube and mask; 30 epochs; val; best epoch 29; one run; background 86.8 not in the mean) and the source (Table IV; SegFormer-B0 HYPER 54.3 named as context, not drawn).
- Percent axis 0 to 100, matching the 39.1 on the card.

## B. Architecture strip, current vs goal (slide 8)

One left-to-right strip, one box per stage. Show status with box style plus a word, never colour alone: solid = merged, hatched = in review, outline only = stub. Full-resolution shapes on the boxes; footnote the half-resolution grid.

| Stage | Output shape, 256 input | Params | Status |
|---|---|---|---|
| Input cube | (B, 224, 256, 256) | 0 | done (loader) |
| Band tokenizer: 14 groups of 16 bands, one shared Conv2d 16 to 64 (kernel 8, stride 8), LayerNorm, learned 14 by 64 group embedding | (B, 1024, 14, 64) | 66,624 | merged, PR #9 |
| Spectral block x2: attention across the 14 groups inside each patch | (B, 1024, 14, 64) | 49,984 each | merged, PR #12 |
| Spectral pool: mean over groups | (B, 1024, 64) | 0 | merged, PR #12 |
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

Drawn by notebook 04, cell B, committed as `docs/figures/architecture.png` and `.svg` (8397ff3), with the spectral block and pool solid (merged, PR #12), the three stubs dashed, and the subtitle "Merged through the spectral pool. The three dashed stages are still stubs, so there is no transformer score."

## C. Data example, three panels (slide 6, or slide 3 for the RGB claim)

Notebook 01 already draws all three pieces for `train[0]`, which has only film, bag, and background. For the slide, pick a cube with filament and overlapping objects, so the panel shows why the label is per pixel. Scanning the mask PNGs costs nothing (no cube reads): loop over `labels_hyper_lt/train`, count classes per mask, and keep the stem. Print the stem on the slide.

1. False colour from bands 56, 112, 168 with the notebook's 2 to 98 percentile stretch. `meta.json` has no wavelengths; with even spacing over 900 to 1700 nm these sit near 1,100, 1,300, and 1,500 nm. Say "approx." or use band numbers.
2. Label mask in the dataset palette, with a legend for all seven classes in the same colours as panel 3.
3. Mean spectrum per class. The notebook averages one image; for the slide, either average each class over all train images or label it "one image". X axis band index or approximate nm, y axis scaled value 0 to 1.

- If two class curves overlap in panel 3, say so; that pair is likely where the model confuses classes.
- 900 to 1700 nm is all infrared, so the cube cannot show what an RGB camera sees. A true RGB comparison needs the dataset's RGB images, which the loader does not read. Do not present the false-colour panel as "what RGB sees".
- Caption: "One train cube, 224 by 256 by 256, uint16 scaled to [0, 1]. Masks are label-transfer masks (`labels_hyper_lt`), 79.6 mIoU against manual labels in the paper's Table II."

Drawn by notebook 04, cell C, committed as `docs/figures/data_example.png` (8397ff3). The chosen cube is `20221028_02_092121` with film, tape, filament (657 px, the thin green curve), and bag. The spectrum panel carries a legend with each class's pixel count in that cube and uses the loader's scaled values, not the display stretch. Film and background run close together in this cube; say so if asked, since that is where a model confuses classes.

## D. Mean spectrum per class, 8 richest cubes each (slide 3 or 6)

Drawn by notebook 04, cell D, committed as `docs/figures/band_signatures.png` (8397ff3). For each object class the cell takes the 8 train cubes with the most pixels of that class (every mask scanned, only those 47 cubes read) and plots the mean of the 8 per-cube mean spectra. Only cardboard carries a plus or minus 1 sd band, because shading every class hid the lines. Background is the background pixels of those same cubes. The 14 token groups are striped, the three false-colour bands dotted, and the top axis is nominal nm with the even-spacing caveat written out. The right panel is the class share of all train pixels, a different population, and its title says so.

Two cautions for the slide. The title's "overlap" is a reading: film, bag, and filament run about 0.05 to 0.08 above the background curve with the same shape, so "run close to the background" is the safer wording if challenged. The legend percentages (cardboard 60%, filament 42%) are the share of each class's train pixels that sit inside its 8 cubes, not accuracy numbers.

## E and F. Q&A facts sheet and source card

Written: `docs/CHECKUP-QA.md` (ten questions, the six most likely first, facts and traps only, blank answer lines) and `docs/CHECKUP-SOURCES.md` (the six links from TODO item 11, every number on the deck mapped to its source and its condition label). The source card's slide-by-slide table is the backup slide content if a number is challenged.

## G. Transformer run evidence (Nov 5)

- Figure: val FG mIoU per epoch for `runs/transformer` next to the `cnn_ds2` curve, plus train loss, from each run's `metrics.csv`. Any run length counts. Caption carries the run command.
- Text box: the harness startup lines verbatim: model class, parameter count, class-weight vector.
- Order: fill `SpatialPE`, `SpatialBlock`, `SegHead` (`docs/TRANSFORMER-TODO.md`); `pytest -q` green; `python -m sw.train --model transformer --smoke --device cpu`; then the Colab command in that file's step 8.
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
