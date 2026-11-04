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
| A | Per-class IoU, grouped bars, gap column, 39.1 and 52.8 lines (TODO 7) | slide 7, the box beside the card | No | done, in the deck | Keep |
| B | Architecture strip with lecture tie-ins and ablation brackets (TODO 2) | its own slide after slide 8 | No | done, in the deck | Keep |
| E | Q&A facts sheet | `docs/CHECKUP-QA.md`, written | No | answers only | Keep |
| F | Source card | `docs/CHECKUP-SOURCES.md`, written; copy to the backup slide | No | 10 min | Keep |
| C | Data example: RGB frame, false colour with outlines, mask, spectrum (TODO 3) | its own slide after slide 6 | Yes | done, in the deck | Keep |
| D | Spectral shape per material, six small multiples, plus a raw version | its own slide after slide 3 | Yes | done, in the deck | Keep |
| E | The metric on two real label sets: RGB-frame label vs transferred label, agreement map, IoU per class | its own slide after the data example | Yes | done, in the deck | Keep |
| Bib | Bibliography: every link Jeong asked for, clickable | `References` slide at the end, and `docs/CHECKUP-BIBLIOGRAPHY.md` | No | done, in the deck | Keep |
| G | Transformer run evidence | Nov 5 | Yes, plus the three stubs | | Deferred |

## A. Per-class IoU (slide 7)

Drawn by notebook 04, cell A, committed as `docs/figures/per_class_iou.png`:

- Grouped horizontal bars per object class. The upper bar is ours in the class colour, the lower bar is the paper's MiniNet-v2 HYPER row in light grey, both with the value printed. MiniNet-v2 is the fair reference because MiniNetSeg is from the same family.
- Classes are sorted by the gap, paper minus ours, and the gap is printed after each pair: filament 30.5, cardboard 21.0, bag 12.7, basket 11.5, film 6.0, tape 0.2. No second axis.
- Two vertical lines draw the headline numbers, 39.1 and 52.8, instead of only stating them.
- The filament row carries the rare-class fact, 0.098% of train pixels and a loss weight of only 1.5x, so the row hands off to the next-steps slide as the meta-deck asks.
- The zero filament bar gets a small green marker at the axis and a bold "0.0".
- Title: FG mIoU 39.1 vs paper 52.8, filament and cardboard are 63% of the gap. Footer: the condition line in two short lines.
- Percent axis 0 to 100. Drawn on a 13.33 by 7.5 inch canvas with 13 to 16 pt type, so it reads from the back of the room.

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

Drawn by notebook 04, cell B, committed as `docs/figures/architecture.png` and `.svg`: the spectral block and pool solid (merged, PR #12), the three stubs dashed, a lecture tie-in under each box title (L1 patch and position embedding, L1 QKV from `nn.Linear` plus softmax, L9 segmentation head, L9 mIoU), brackets over the stages each Phase 2 ablation row removes, and a subtitle that says from scratch, no pretrained weights, about 0.49M designed, and states the hypothesis. The footer keeps the half-resolution grid, the next steps, and Phase 3 INT8 as planned. The canvas is 13.33 by 7.5 inches with the axes filling it, so box text is sized against the real slide width.

## C. Data example, four panels (its own slide after slide 6)

Drawn by notebook 04, cell C, committed as `docs/figures/data_example.png` at 9.28 by 3.90 inches: cube `20221028_02_092121` with film, tape, filament (657 px) and bag. Four panels: the release's RGB frame (the Hugging Face mirror ships `<stem>.rgb.png`; the cell uses it when an `rgb/train` folder sits next to `hyper/`, and skips the panel otherwise), the false colour with the label outlines so the transferred masks can be judged against the scene, the mask with a one-row class legend under the images, and the mean spectrum per class with the closest pair named in the title (background and film, gap 0.009), which is where a model confuses classes. The footer carries the band numbers and the Table II label-transfer figure. Type is 8 to 9.5 pt at 1:1.

## D. Spectral shape per material (its own slide after slide 3)

Drawn by notebook 04, cell D, committed at 9.28 by 3.90 inches as two files:

- `band_signatures.png`: six small multiples, one object class per panel, with each pixel's spectrum divided by its own mean before averaging. Brightness from illumination and thickness drops out, so only the shape stays. The highlighted class carries a plus or minus 1 sd band across its 8 cubes; the other classes are grey; background is dotted. Identity comes from the panel, not from colour alone. The x axis is nominal nm with the even-spacing caveat.
- `band_signatures_raw.png`: the same grid on raw scaled values, for the backup. There film, bag and filament run about 0.05 to 0.08 above the background with the same shape, so "run close to the background" is the honest wording for the raw view.

What the normalized panels show: film has a sharp dip near 1180 nm and a second near 1380; filament and bag share deep dips near 1200 and 1390 with different slopes; cardboard rises to 1300 and falls hard past 1600; tape is nearly flat. That is the evidence slide 3 needs that look-alike materials separate by spectral shape, which raw brightness hides. The caption percentages (cardboard 60%, filament 42%) are the share of each class's train pixels that sit inside its 8 cubes, not accuracy numbers. The class pixel share panel is gone; slide 6 states those numbers.

## E. The metric on two real label sets (its own slide after the data example)

Drawn by notebook 04, cell E, committed as `docs/figures/iou_illustration.png` at 9.28 by 3.90 inches. Jeong asked for the metric to be defined; this does it on real masks rather than a toy diagram. For the data-example cube it shows the label drawn on the RGB frame (A), the same label transferred to the HSI frame (B), their agreement map (class colour where both agree, red only in A, blue only in B), and the IoU per class between the two with the formula IoU = TP / (TP + FP + FN) and the FG mean as a dashed line. Numbers: film 0.757, tape 0.576, filament 0.000, bag 0.831, FG mean 0.541 over the four classes present, pixel agreement 0.899. The cell asserts the hand count against `sw.metrics.fg_miou_from_cm` when the package imports.

Say with it: two real label sets, not a model output. The filament zero is the lesson: a 2-pixel-wide curve offset by a few pixels between frames has no overlap, so IoU punishes thin classes for small shifts. That is also why half resolution hurt filament in the CNN run. The paper's Table II (79.6) is a different comparison, transferred vs manual labels on the HSI frame. Needs an RGB-frame label folder (`labels_rgb/train`); the Hugging Face mirror ships it, and the cell skips itself without it.

## Bibliography (the `References` slide)

`docs/CHECKUP-BIBLIOGRAPHY.md` lists thirteen entries with what each backs: the paper, the Zenodo release, the Hugging Face mirror, the official code, MiniNet-v2, SegFormer, CMX, ViT, SpectralFormer, SSFT, DeiT and Steiner, our repository, and the two lectures. The same list is the last slide of the deck copy, every entry a clickable link except the lectures, whose Canvas links are Scott's to add. It is the only slide text I wrote, and it is citations.

## F. Q&A facts sheet and source card

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
| Midday | C and D are done: rendered here from the Hugging Face mirror of the dataset (`ferpb/spectralwaste-segmentation`, the same 256 by 256 cubes and `label_hsi_lt` masks, plus the RGB frames). All four figures are in the deck copy. |
| By 4 PM | Trim notes to the review's per-slide word budgets (about 680 words for 5:00). Dry run with the Q&A sheet. Delete every dashed box. |
| Nov 5 | G. |
