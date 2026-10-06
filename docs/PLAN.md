# DATA 255 Group 1 — SpectralWaste team plan

Updated Tue Oct 6, 2026. Group: Scott Nelson, Surya Reddy, Tim Wada, Prakhar, Yash.

## Where we are

- We segment jam objects on a recycling line with a small spectral–spatial transformer we build from scratch (no pretrained weights). Topic approved by Jeong; confirmed verbally with Mamta.
- Data: Zenodo record 10880544, `spectralwaste_segmentation.zip` ([Casao et al., IROS 2024](https://arxiv.org/abs/2403.18033)), 23 GB, md5 `ae2c268bb385345f12e55b4389d2d3b9`. Checked on the real files: 514 train / 167 val / 171 test. Cubes are 224 bands from 900 to 1700 nm, stored 224 × 256 × 256. Masks use classes 0–6 (0 = background, 6 object classes).
- Metric: mIoU over the 6 object classes. Our bar is the published hyperspectral-only results, MiniNet-v2 at 52.8 and SegFormer-B0 at 54.3. CMX-B0's 58.2 uses RGB and hyperspectral together, so we cite it as context, not as a target.

## Phases

1. **Phase 1.** A minimal transformer that segments the hyperspectral images, on the official split, with an honest test mIoU and per-class scores. **Nov 4 checkup target:** a CNN control score, plus the transformer training on real data. Any transformer score is a bonus. A v1 below 52.8 (or below the CNN) is a normal from-scratch result on 514 images.
2. **Phase 2.** Spectral–spatial fusion with ablations (spectral-only / spatial-only / fused).
3. **Phase 3.** INT8 quantization for portable or NPU deployment. We measure how much mIoU drops.

## Work packages and suggested owners

Swap owners freely.

| # | Work package | Done when | Suggested owner |
|---|---|---|---|
| WP1 | Data loader + repo cleanup | `SpectralWasteDataset` returns a (224, 256, 256) float in [0, 1] and a (256, 256) mask; asserts 514/167/171; README says Group 1 SpectralWaste | Scott |
| WP2 | Colab data setup | One notebook cell downloads the zip from Zenodo, checks md5, and unzips; every member loads one batch on Colab | Yash |
| WP3 | Training + evaluation harness | `train.py` or a notebook: loss with class weights, val early stopping, foreground-only mean IoU plus per-class IoU, saves checkpoints and a results CSV | Yash |
| WP4 | CNN control model | Small CNN (~0.4–0.8M params) trains end to end through WP3 and gives a first mean IoU; proves the pipeline works | Surya |
| WP5 | Band tokenizer + positional encoding | Groups the 224 bands into tokens; unit test on a random cube passes shape checks | Tim |
| WP6 | Transformer blocks + seg head | Our own QKV attention, feed-forward, norm, and a decoder that outputs (7, 256, 256) logits; unit tests pass | Prakhar |
| WP7 | Experiments + results table | Runs WP5 and WP6 through WP3, tunes band-group size and width, fills the table vs 52.8 / 54.3 (58.2 as context) | Tim + Prakhar |
| WP8 | Checkup deck + writeup | Nov 4 slides: problem, data, model diagram tied to lecture topics, mean IoU table, failure notes, Phase 2/3 plan. Vision research drafts the related-work and baseline slide | Surya + Scott |

## Timeline

| Week | Goal |
|---|---|
| Oct 6–12 | WP1 merged, WP2 working for all five; WP5/WP6 skeletons on random tensors. Assignment 5 is due Wed Oct 7, 6 PM, so keep Oct 6–7 light. Loader target is end of week. |
| Oct 13–19 | WP3 + WP4, first CNN mean IoU; transformer forward pass on real batches. Lab 2 is due Wed Oct 14, 6 PM, so keep Oct 13–14 light. |
| Oct 20–26 | Transformer v1 trains; first validation mean IoU |
| Oct 27–Nov 3 | Tune, per-class analysis, fill the results table, build the deck |
| **Wed Nov 4, 6 PM** | Progress checkup (5 of 20 project points, Canvas 7916136): CNN mean IoU, transformer training on real data, and the Phase 2/3 plan |
| Nov 5–Dec 1 | Transformer tuning, Phase 2 fusion ablations, Phase 3 INT8 if there is time |
| **Wed Dec 2** | Final presentation (10 points). The report is the other 5 points. |

## Ground rules

- Never commit data. `data/`, `spectralwaste_segmentation/`, and the tiff cubes are gitignored. The loader reads the folder from the `SW_ROOT` environment variable.
- One branch per work package, small pull requests, at least one teammate review.
- Train on Colab GPUs (SJSU account). The data is about 24 GB unzipped, so download it inside Colab rather than through Drive.
- Use the official split only. Do not add other datasets to help the transformer.
- The syllabus deducts for content that was not covered in class. On every slide, tie band tokens, attention, and the segmentation head back to lecture topics.

## Data gotchas

- Masks are named `<stem>.png` next to `<stem>.tiff`. There are no `*semantic.png` files, so a loader that expects that suffix will crash. Pair cubes and masks by sorted stem.
- Classes, from `meta.json`: 0 background, 1 film, 2 basket, 3 cardboard, 4 video_tape, 5 filament, 6 bag.
- Cubes are uint16, band-first `(224, 256, 256)`, 900 to 1700 nm. Read them with `tifffile`. Scale to float32 in [0, 1] by dividing by 65535.
- Class weights must come from the `labels_hyper_lt` train counts `[27612233, 1819248, 1060150, 220623, 397471, 33101, 2542678]`, not the official repo's RGB counts. `meta.json` `categories_pixel_counts` are the RGB-label counts. The results slide should say so in one sentence: the paper weighted hyperspectral runs with RGB-label counts.
- License: CC BY 4.0. Download: <https://zenodo.org/records/10880544/files/spectralwaste_segmentation.zip?download=1>
