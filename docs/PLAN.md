# DATA 255 Group 1 — SpectralWaste team plan

Rev 3. Updated Thu Oct 22, 2026. Group: Scott Nelson, Surya Reddy, Tim Wada, Prakhar, Yash.

Scott Nelson owns the critical path and reviews the pull requests. Teammates have not answered the Oct 6 email. The optional tasks below do not block anything. This revision is scaffolding for the Nov 4 checkup, not a finished graded hand-in.

## Where we are

- We segment jam objects on a recycling line with a small spectral–spatial transformer we build from scratch (no pretrained weights). Topic approved by Jeong; confirmed verbally with Mamta.
- Data: Zenodo record 10880544, `spectralwaste_segmentation.zip` ([Casao et al., IROS 2024](https://arxiv.org/abs/2403.18033)), 23 GB, md5 `ae2c268bb385345f12e55b4389d2d3b9`. Checked on the real files: 514 train / 167 val / 171 test. Cubes are 224 bands from 900 to 1700 nm, stored 224 × 256 × 256. Masks use classes 0–6 (0 = background, 6 object classes).
- Metric: mIoU over the 6 object classes. Our bar is the published hyperspectral-only results, MiniNet-v2 at 52.8 and SegFormer-B0 at 54.3. CMX-B0's 58.2 uses RGB and hyperspectral together, so we cite it as context, not as a target.
- In the repo now: the loader, foreground mIoU, from-scratch attention (`src/sw/attention.py`), wavelength encoding (`src/sw/pe.py`), the paper class-weight default, a training harness (`python -m sw.train`), and a MiniNet-family CNN control (`MiniNetSeg`, 571,879 parameters). No real validation score yet. No band-token transformer yet.

## Critical path (Scott)

Scott directs and reviews. He will do these whether or not the optional tasks come back.

| When | What | Done when |
|---|---|---|
| This week (from Oct 22) | Training harness + CNN baseline | `python -m sw.train --smoke` finishes on CPU. `MiniNetSeg` maps 224 bands to 7 full-resolution classes and has about 0.4–0.8M parameters. |
| ~Oct 26 | First real CNN validation score | One Colab run of `MiniNetSeg` on the official train split, FG mIoU on the official val split, logged in `metrics.csv`. Any number is a result. 52.8 is context, not a gate. |
| ~Nov 1 | Minimal band-token transformer | A small model that uses our band tokenizer and `src/sw/attention.py`, trained briefly on Colab on the real cubes. |
| **Wed Nov 4, 6 PM** | Progress checkup, 5 of 20 points (Canvas 7916136) | Three things on the slides: the CNN score, evidence the transformer trained on real data, and the Phase 2/3 plan. A score below 52.8, or below the CNN, is a normal from-scratch result on 514 images. |

## Optional tasks (1–2 hours, blocks nothing)

Each person has one task. If it does not happen, the critical path still moves.

| Who | Task |
|---|---|
| Yash | Run `notebooks/01_look_at_the_data.ipynb` on Colab and post a screenshot in the group thread. |
| Surya | Draft the data slide: what a cube is, the 514/167/171 split, and the six object classes. |
| Tim | Review one pull request, or run one training config and paste the val FG mIoU. |
| Prakhar | Review one pull request, or run one training config and paste the val FG mIoU. |

## Phases

1. **Phase 1.** A minimal transformer that segments the hyperspectral images, on the official split, with an honest test mIoU and per-class scores. The Nov 4 checkup wants a CNN control score plus the transformer training on real data. Any transformer score is a bonus.
2. **Phase 2.** Spectral–spatial fusion with ablations (spectral-only / spatial-only / fused).
3. **Phase 3.** INT8 quantization for portable or NPU deployment. We measure how much mIoU drops.

## Work packages

The old suggested-owner column is retired. Scott holds every package on the critical path. The four optional tasks are the only teammate asks.

| # | Work package | Done when | Status on Oct 22 |
|---|---|---|---|
| WP1 | Data loader + repo cleanup | `SpectralWasteDataset` returns a (224, 256, 256) float in [0, 1] and a (256, 256) mask; asserts 514/167/171 | Done (Scott, Oct 6) |
| WP2 | Colab data setup | One notebook cell downloads the zip, checks md5, and unzips | Scaffold is in the notebooks. Yash's optional task is to run notebook 01 and post a screenshot. |
| WP3 | Training + evaluation harness | Loss with class weights, val early stopping, foreground mIoU plus per-class IoU, checkpoint and a results CSV | Scaffold landed Oct 22 (`src/sw/train.py`). A real Colab score is the Oct 26 row. |
| WP4 | CNN control model | Small CNN (~0.4–0.8M params) trains through WP3 | `MiniNetSeg` is in `src/sw/models/cnn.py`. First val score is still open. |
| WP5 | Band tokenizer + positional encoding | Groups the 224 bands into tokens; shape checks pass | Positional encoding is in `src/sw/pe.py`. The tokenizer is part of the Nov 1 model. |
| WP6 | Transformer blocks + seg head | Our own QKV attention and a decoder that outputs (7, 256, 256) logits | Attention blocks are in `src/sw/attention.py`. The seg head is part of the Nov 1 model. |
| WP7 | Experiments + results table | Runs the transformer through the harness and fills the table vs 52.8 / 54.3 (58.2 as context) | After the checkup. |
| WP8 | Checkup deck + writeup | Nov 4 slides: problem, data, model diagram, mean IoU table, failure notes, Phase 2/3 plan | Not started. Surya's optional task is the data slide only. |

## Built-in layers, the model we report, and the Nov 4 slides

Jeong said built-in PyTorch transformer pieces are OK, and customization is preferred. A first transformer may start from `nn.TransformerEncoderLayer` so something trains quickly. `torch.nn.functional.scaled_dot_product_attention` is fine inside our own block.

The model we report as ours must use our own band tokenizer (WP5) and the attention block in `src/sw/attention.py`, because the README claims a transformer built from scratch. The built-in `nn.TransformerEncoderLayer` version stays a starting point and a comparison row (built-in vs custom block), which is also an ablation. If that built-in version ends up as the reported model, tell Vision research and change the claim to "trained from scratch with a custom spectral tokenizer" first.

Lecture tie-ins are a helpful framing for the slides. Band tokens and the wavelength encoding relate to the ViT-components slide in Lecture 1 (patch embedding, position embedding, class token). Segmentation and mIoU relate to Lecture 9 (Oct 21). What was actually said in class, and Jeong's direct guidance, take precedence over the syllabus text.

## Contributions log (Oct 6–22)

Every pull request on `main` in this window was merged by Scott. Surya, Tim, Prakhar, and Yash have no commits here.

| Date | What landed | Who |
|---|---|---|
| Oct 6 | Group 1 restart: README claim, loader, foreground mIoU, attention, wavelength encoding, starter notebooks, download script (PRs #2 and #3) | Scott Nelson |
| Oct 6 | Plan note: the model we report uses `src/sw/attention.py`; a built-in encoder layer is only a comparison row (PR #4) | Scott Nelson |
| Oct 6 | Email to the group asking people to take a work package. No reply through Oct 22. | Scott Nelson |
| Oct 7 | Executed notebooks committed so the Oct 6 real-data run is visible on GitHub (PR #6) | Scott Nelson |
| Oct 19 | Plan note: lecture framing, and the syllabus text is not the slide gate (PR #5) | Scott Nelson |
| Oct 22 | Paper class weights, training harness, MiniNet-family CNN baseline (this revision) | Scott Nelson |

## What rev 3 adds in the code

- `class_weights()` defaults to the SpectralWaste scheme `w_c = (median(f) / f_c) ** 0.12`. `class_weights(scheme="inverse")` is the old mean-1 inverse frequency.
- `python -m sw.train` (same flags as `python scripts/train.py`) trains with AdamW, those weights, horizontal and vertical flips, and an optional crop or integer downsample. Each epoch appends val FG mIoU and per-class IoU to `metrics.jsonl` and `metrics.csv`, and keeps `best.pt`. `--patience N` turns on early stopping. `--smoke` runs two steps.
- `MiniNetSeg` is a small encoder-decoder in the MiniNet family: a 1×1 band-reduction stem, depthwise separable convolutions, and a multi-dilation module. The default width is 571,879 parameters. `python -m sw.models.cnn` prints that count.

## Timeline after the checkup

| When | Goal |
|---|---|
| Nov 5–Dec 1 | Transformer tuning, Phase 2 fusion ablations, Phase 3 INT8 if there is time |
| **Wed Dec 2** | Final presentation (10 points). The report is the other 5 points. |

## Ground rules

- Never commit data. `data/`, `spectralwaste_segmentation/`, and the tiff cubes are gitignored. The loader reads the folder from the `SW_ROOT` environment variable. Checkpoints and `runs/` stay untracked too.
- One branch per work package, small pull requests, at least one review. Scott reviews until someone else picks up a review.
- Train on Colab GPUs (SJSU account). The data is about 24 GB unzipped, so download it inside Colab rather than through Drive.
- Use the official split only. Do not add other datasets to help the transformer.
- On every slide, tie band tokens, attention, and the segmentation head back to lecture topics. Class guidance wins over the syllabus text.

## Data gotchas

- Masks are named `<stem>.png` next to `<stem>.tiff`. There are no `*semantic.png` files, so a loader that expects that suffix will crash. Pair cubes and masks by sorted stem.
- Classes, from `meta.json`: 0 background, 1 film, 2 basket, 3 cardboard, 4 video_tape, 5 filament, 6 bag.
- Cubes are uint16, band-first `(224, 256, 256)`, 900 to 1700 nm. Read them with `tifffile`. Scale to float32 in [0, 1] by dividing by 65535.
- Class weights must come from the `labels_hyper_lt` train counts `[27612233, 1819248, 1060150, 220623, 397471, 33101, 2542678]`, not the official repo's RGB counts. `meta.json` `categories_pixel_counts` are the RGB-label counts. The training default applies the paper formula `w_c = (median(f) / f_c) ** 0.12` to those hyperspectral counts (about `[0.676, 0.937, 1.000, 1.207, 1.125, 1.516, 0.900]`). Inverse frequency, mean 1, is `class_weights(scheme="inverse")` and spans about 830×. The results slide should say, in one sentence, that the paper weighted hyperspectral runs with RGB-label counts. We weight with the hyperspectral counts.
- License: CC BY 4.0. Download: <https://zenodo.org/records/10880544/files/spectralwaste_segmentation.zip?download=1>
