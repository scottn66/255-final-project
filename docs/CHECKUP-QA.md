# Nov 4 checkup: Q&A facts sheet

Written Nov 4, 2026 for the 1:30 of questions. This sheet holds facts, sources, and traps. The spoken answers are Scott's, in his own words (Jeong's rule on AI-written presentation content, deck TODO item 10). Write each one in the blank line, then say it out loud once before 6 PM.

Numbers are the ones already on the deck. Slide numbers are the 10-slide order in the deck copy. Paper = Casao et al., IROS 2024, arXiv 2403.18033.

## Limitation, hypothesis, novelty (a draft to rewrite in your own words)

Jeong's arc for the talk: the limitation of prior work, then a hypothesis that answers it, then the technical novelty stated the way an ECCV reviewer would weigh it. This is the best version I can build from the evidence on hand. Slide 5 and the end of slide 4 are yours to write; nothing here goes on a slide as written.

### The limitation, with the number that proves it

Both published hyperspectral baselines consume the 224 bands through one first convolution: MiniNet-v2 through its stem, SegFormer-B0 through its overlapped patch embedding. The spectral axis is collapsed into feature channels before any spatial reasoning, and nothing downstream can attend across wavelengths. Three facts from the paper show what that costs.

- Table IV: SegFormer-B0 loses only 0.8 points when the 224 bands are replaced by 3 PCA components (HYPER3 53.5 vs HYPER 54.3). The published transformer baseline is using little of the spectrum beyond three components. MiniNet-v2 loses 3.8 (49.0 vs 52.8).
- Table IV: the thin, look-alike classes stay weak in both models: filament 30.5 / 33.1, tape 28.8 / 21.9, against cardboard at 78.8 / 85.2.
- The paper has no row that separates the spectral contribution from the spatial one, and no INT8 row.

Our own data figure supports the reading: after each pixel is divided by its mean, film, bag, filament and cardboard separate by spectral shape (dips near 1180 and 1380 nm for film; near 1200 and 1390 for filament and bag; a hard fall past 1600 for cardboard), while tape stays flat. A first convolution that sums bands sees brightness; the shape is what distinguishes the materials.

One-sentence form: "The published baselines collapse the 224 bands in their first layer, and the paper's own PCA-3 row shows the transformer barely uses them."

### The hypothesis, falsifiable, with the margin

Claim limit from the context prompt: "competitive", never "beat"; INT8 is planned.

A from-scratch model that attends over 14 band groups before it attends over space, at no more than 0.6M parameters, will (a) reach a validation FG mIoU within 3 points of MiniNet-v2's 52.8 under the paper's protocol (full resolution, 200 epochs, official split), and (b) beat its own spatial-only ablation by at least 2 points. If (a) fails, the model is not competitive at this size. If (b) fails, attention over bands did not earn its cost, and the honest conclusion is that a spatial model with a learned band reduction is enough. Either outcome is reported.

Why 3 and 2: the paper's numbers are single runs with no seed spread reported. SSFT-style work on hyperspectral data shows about 1 point of seed spread at this model size, so 3 points is about three spreads, and 2 points on the ablation is the smallest difference that would survive a second seed. Say the margins are a judgment, and that the final table runs three seeds (`docs/RECIPE.md` section 5).

What tests it: the Phase 2 rows, spectral-only, spatial-only, both, at matched parameters, on the same split and metric. The brackets on the architecture figure mark what each row removes.

### Technical novelty, stated for an ECCV reader

Contributions, in the order a reviewer would check them:

1. A factorized spectral-then-spatial token attention for dense segmentation at under 0.6M parameters on hyperspectral waste data. The precedents (SSFT at 0.516M, SpectralFormer) are pixel or patch classifiers, and the only segmentation transformers reported on SpectralWaste are SegFormer-B0 at 4.06M and the 11.5M fusion model.
2. An ablation that isolates the spectral attention's contribution at matched parameters and matched training, which the paper does not report. This is the experiment that answers the limitation above.
3. A measured INT8 drop for the resulting model, planned for Phase 3, which no SpectralWaste result reports.

An optional fourth row, untested and motivated by the data figure: brightness-invariant band tokens, where each pixel's spectrum is divided by its own mean before tokenization. It is a one-line loader change and, if it moves the thin classes, it becomes the most defensible single technical idea in the project. Do not claim it until it is run.

Not new, and said so: wavelength encoding (technique adoption), grouping bands into tokens (SpectralFormer), the from-scratch training recipe (DeiT, Steiner), the dataset and the metric.

### How to say it in Jeong's frame

Problem in one line, then the limitation with its number (the PCA-3 row), then the hypothesis with its two conditions and margins, then the three contributions, then what is not new. The novelty is moderate and should be called moderate: a new combination on a new dataset, plus the ablation and the deployment measurement that make it a result rather than a demo. A reviewer forgives moderate novelty with a clean experiment; they do not forgive an unfalsifiable claim.

## The six most likely

### 1. Is 39.1 comparable to 52.8? Why is it so far below?

Facts

- Ours: val split, half resolution for the cube and the mask (`prepare_sample` in `src/sw/train.py` strides both for every split), 30 epochs, batch 4, best epoch 29, one run. `docs/RESULTS.md`, row `cnn_ds2`.
- Paper: 256 by 256 (Sec IV-A), 200 epochs (Sec V-A), batch 12 (official code, not the paper). The split Table IV is scored on is not stated.
- Per-class gap, paper MiniNet-v2 HYPER minus ours, in points: film 6.0, basket 11.5, cardboard 21.0, tape 0.2, filament 30.5, bag 12.7. Filament and cardboard are 63% of the 13.7-point gap. Tape already matches.
- Same family: MiniNetSeg is a MiniNet-style encoder-decoder, 571,879 params vs the paper's 0.585M.

Trap: do not promise 52.8. `docs/PLAN.md` says below 52.8 is a normal from-scratch result on 514 images. Say "control", and that the full-resolution rerun is the next run.

Your answer:

### 2. Why is background left out of mIoU?

Facts

- Paper Sec V-A: per-class IoU, "we report their mean (mIoU)". It never says whether background is in that mean.
- Every Table IV mIoU equals the mean of its six object classes. MiniNet-v2 HYPER: (61.2 + 61.0 + 78.8 + 28.8 + 30.5 + 56.3) / 6 = 52.77. SegFormer-B0 HYPER: 326.0 / 6 = 54.33. CMX-B0: 349.1 / 6 = 58.18.
- Background is 82.0% of train pixels (27,612,233 of 33,685,504, `src/sw/data.py`). Our background IoU is 0.868. Including it would move 0.391 to 0.459 with no object getting better.
- `src/sw/metrics.py`: nanmean over classes 1 to 6. An all-background predictor scores 0.

Trap: "the paper excludes it" is not in the paper. "Same as Table IV" is fair, and the arithmetic is the proof.

Your answer:

### 3. Why is filament exactly zero, and what is the fix?

Facts

- 33,101 train pixels, 0.098% of train pixels (`HYPER_LT_TRAIN_PIXEL_COUNTS`). 111 instances in the paper's Table I.
- Paper-scheme weights at exponent 0.12 span 0.676 to 1.516, about 2.2x. Exponent 0.5 would give filament 5.66; 1.0 gives 32.
- Half resolution: stride-2 sampling can delete a 1 to 2 pixel filament from both the cube and the mask. Not yet measured: filament pixels in val masks at 256 vs 128, and how many val and test images contain filament (`docs/PLAN.md`, known issue).
- The CNN run used weighted cross-entropy only. `docs/RECIPE.md` section 6 adds soft Dice over classes 1 to 6; Lovász fine-tune and focal are the alternatives. The paper's MiniNet-v2 reaches 30.5 at full resolution and 200 epochs.

Trap: do not blame the data alone; the paper's same-family model gets 30.5.

Your answer:

### 4. If 3 PCA channels get 53.5, why attend over 224 bands?

Facts

- Table IV: SegFormer-B0 HYPER3 53.5 vs HYPER 54.3; MiniNet-v2 HYPER3 49.0 vs HYPER 52.8. PCA-3 costs 0.8 points on the transformer and 3.8 on the CNN.
- The Phase 2 ablation (spectral only, spatial only, both) is the planned answer. A PCA-3 input to our own backbone is the direct control (`docs/RECIPE.md` section 1). `hyper_pca3/` is not in the Zenodo zip, so it has to be built.
- Spectral attention runs over 14 group tokens per patch, so it is cheap. Recipe estimate about 8 GFLOPs for the whole model vs the paper's 6.347 for SegFormer-B0 HYPER and 3.429 for MiniNet-v2 HYPER.

Trap: 53.5 is SegFormer-B0 on HYPER3. Name the model.

Your answer:

### 5. What is new here?

Facts

- Spectral-then-spatial transformers exist for HSI pixel classification (SSFT at 0.516M, ChannelViT, LESSViT; references in `docs/RECIPE.md`). None is a segmenter on SpectralWaste.
- The paper's only transformer is SegFormer-B0 at 4.062M, and the released code loads no pretrained weights (inferred from the code, not stated in the paper).
- Ours: that combination, from scratch, about 0.49M, for segmentation on SpectralWaste, plus the Phase 2 ablation and a planned INT8 measurement, which the paper does not report.
- Claim limits: "competitive", never "beat"; INT8 is planned; novelty is moderate, a new combination on a new dataset plus a deployment measurement.

Trap: the wavelength encoding is technique adoption, not the novelty (`docs/RECIPE.md`, claim guard). Do not compare to CMX-B0's 58.2 (RGB plus HSI, 11.539M).

Your answer:

### 6. What result would make the hypothesis false?

Facts

- The margin on slide 5 is the answer. Without a margin, "competitive" cannot fail. The comparison is FG mIoU on the official split against 52.8 and 54.3 at about 0.49M params.
- The Phase 2 ablation tests it: spectral only vs spatial only vs both. If "both" does not beat "spatial only", attention over bands did not earn its cost.
- Fallbacks if the transformer underfits 514 images: `docs/RECIPE.md` section 4 (the paper's augmentation, stochastic depth 0.1, weight decay 0.05) and the top fix in section 7, a small conv stem.

Trap: "a score under the CNN is still a result" is true but is not a hypothesis test. Say what number would count as not competitive.

Your answer:

## If there is time

### 7. Is the attention really from scratch? Jeong allowed built-in layers.

- `src/sw/attention.py` is `nn.Linear` for Q, K, V and the output projection, then softmax. `tests/test_transformer.py` fails if `nn.MultiheadAttention`, `nn.TransformerEncoder`, or `timm` appears in the model file.
- A built-in `nn.TransformerEncoderLayer` version is allowed as a comparison row only (`docs/PLAN.md`).

Your answer:

### 8. Why these class weights and not the paper's?

- `meta.json` pixel counts are RGB-label counts, and the official code hard-codes them. We weight with the hyperspectral `labels_hyper_lt` counts.
- At exponent 0.12 the two differ by under 1% (filament 1.525 vs 1.516). Protocol correctness, not a score lever (`docs/RECIPE.md` section 6).

Your answer:

### 9. Where do the hyperspectral labels come from, and what is 79.6?

- Paper Sec IV-C: contour points on each RGB mask component are matched into the HSI image with COTR, one affine transform per component, and the pieces become the HSI mask (label transfer, LT).
- Table II: LT masks score 79.6 mIoU against 81 manually labelled instances in 20 HSI images. Manual alignment scores 56.3.
- It is an agreement score between transferred and manual labels. "Ceiling" is our reading.

Your answer:

### 10. Why hyperspectral at all? Can an RGB camera not separate these?

- Table IV, same model, RGB vs HYPER: MiniNet-v2 44.5 vs 52.8, SegFormer-B0 48.4 vs 54.3. On MiniNet-v2, cardboard moves 55.4 to 78.8 and filament 10.0 to 30.5; film and tape move slightly the other way.
- 900 to 1700 nm is all infrared. The false-colour panel is not what an RGB camera sees; the dataset's RGB images are a separate folder the loader does not read.

Your answer:

## Numbers to have on one line

- Train pixels: 33,685,504 = 514 x 65,536. Background 82.0%. Filament 0.098%.
- Ours, val, half resolution, epoch 29: 39.1. Film 55.2, basket 49.5, cardboard 57.8, tape 28.6, filament 0.0, bag 43.6. Background 86.8, not in the mean.
- Paper MiniNet-v2 HYPER, Table IV: 52.8. 61.2, 61.0, 78.8, 28.8, 30.5, 56.3. 0.585M, 3.429 GFLOPs.
- Paper SegFormer-B0 HYPER: 54.3. 65.4, 63.2, 85.2, 21.9, 33.1, 57.2. 4.062M, 6.347 GFLOPs.
- CMX-B0 RGB-HYPER: 58.2, 11.539M. Context only.
- Model: tokenizer 66,624; 49,984 per block, 8 blocks; about 0.47M with a 1x1 head, about 0.49M with the recipe head. Designed, not measured.
