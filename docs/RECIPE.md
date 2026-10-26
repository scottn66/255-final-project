# SpectralWaste: training recipe for a small from-scratch spectral–spatial transformer
**DATA-255 Group 1** (Scott, Surya, Tim, Prakhar, Yash) · Written Tue Oct 6, 2026 · Due Sun Oct 18 · v1 training starts Mon Oct 20
**Feeds:** WP5 (band tokens + PE), WP6 (blocks + seg head), WP7 (v1 training + tuning). It uses the WP3 harness and the WP4 CNN control.
**What this is:** cited guidance with defaults and sweep ranges. It is not model code. The pseudo-config is a spec, not an implementation.

> **Claim guard (from CLAIM.md):** Phase 1 is a *minimal* from-scratch band-token transformer on HSI only, scored on the official 514/167/171 split by FG mIoU over classes 1–6. The bars are **MiniNet-v2 HYPER 52.8** and **SegFormer-B0 HYPER 54.3**. **CMX-B0 58.2 is RGB+HSI and is context only.** Dual-path fusion with cross-attention (SSFT-style) is Phase 2, and INT8 is Phase 3. λ/wavelength PE is technique adoption, not the novelty. A v1 below 52.8 is a normal result for a from-scratch model trained on 514 images.

**Legend:** **[E]** marks an evidence-based default (cited paper, or the SpectralWaste repo/paper itself). **[J]** marks our judgment call. A [J] is reasoned but untested on this dataset, so WP7 should treat it as a hypothesis.

---

## 0. Top-line v1 config (one table)

| Knob | **v1 default** | WP7 sweep (small) | Basis |
|---|---|---|---|
| Input | `hyper` 224×256×256, target `labels_hyper_lt`, native 256² (no downsampling) | — | [E] paper/repo protocol |
| Input norm | per-band z-score from **train** stats | none (raw [0,1], as in repo) | [J] |
| Band-group size *g* | **16 bands/token → G = 14 spectral tokens** (≈57 nm each) | {8, 16, 32} | [E] grouping precedent (SpectralFormer, SatMAE); size is [J] |
| Spatial patch *P* | **8×8 → 32×32 = 1024 patch tokens** | {8, 16}; 4 only with SR-attention | [J], from the compute math in §1 |
| Tokenizer | shared linear patch-embed per band group (`Conv2d(g→D, k=P, s=P)` applied to each group) | 1×1-conv 224→32 reduction, then same backbone; PCA-3 control | [E] ViT patch embed; ChannelViT shared per-channel embed |
| Encoder | **2 spectral blocks** (attention over the G group tokens inside each patch) → mean-pool over G → **6 spatial blocks** (attention over the 1024 patch tokens) | spectral {1,2}, spatial {4,6} | [J] factorized design (TimeSformer/axial precedent) |
| Width / heads / MLP | **D = 64, 4 heads, MLP ratio 4, pre-LN** | D {64, 96}; heads {2, 4}; ratio {2, 4} | [E] SSFT uses D=64, 4 heads; ratio 4 per ViT/SegFormer |
| Params | **≈0.49 M** (table in §2) | stay ≤ ~0.6 M; report the count | [E] SSFT 0.516 M, MiniNet HYPER 0.585 M |
| Positional | spatial: fixed 2D sin-cos (0 params); spectral: learned group embedding (G×D) | + Mix-FFN 3×3 depthwise conv; LESSViT λ-RoPE (optional, cite-only) | [E] SegFormer, SSFT, ChannelViT |
| Seg head | 1×1 D→32 → 2× (bilinear ×2 + 3×3 conv) to 128² → bilinear ×2 → concat full-res 1×1 spectral skip (224→16) → 1×1 48→32 → 1×1 → 7 logits (~26 k params, dropout 0.1) | with/without skip (**required ablation**) | [J]; dropout 0.1 [E] repo SegFormer head |
| Augment | paper's aug: **rot ±30° + H/V flips**, plus lossless rot90. Mask fill = **ignore 255** | + gain ×U(0.9,1.1); + band-group dropout; + random resized crop | [E] paper; fill fix [J] |
| Regularization | stochastic depth **0.1** (linear over depth), dropout 0, label smoothing 0 | drop-path {0.1, 0.2}; LS {0, 0.05, 0.1} | [E] DeiT, repo SegFormer-B0 config |
| Optimizer | **AdamW, peak LR 1e-3**, β=(0.9, 0.999), **wd 0.05** (no wd on norms/biases/embeddings) | LR {5e-4, 1e-3, 2e-3}; wd {0.01, 0.05, 0.1} | [E] repo SegFormer LR; DeiT wd |
| Schedule | **5-epoch linear warmup → cosine to 1e-5**, stepped per iteration | poly 0.9 (repo MiniNet) | [E] DeiT |
| Batch / epochs | **batch 8**, **max 200 epochs** | batch {8, 12} | [E] paper 200 ep / bs 12; SSFT bs 8 |
| Selection / stop | best **val FG mIoU** checkpoint; early-stop patience **40** after a 60-epoch minimum; test **once** | — | [E] repo best-val checkpoint; patience [J] |
| Precision | fp16 autocast + GradScaler on T4 (bf16 on L4/A100); attention softmax in fp32; grad clip 1.0 | — | [E] Micikevicius; ViT/Steiner clip at 1 |
| Loss | **CE, median-freq weights from `labels_hyper_lt` counts (exp 0.12), ignore 255, + soft Dice over classes 1–6 (weight 1.0)** | CE only; CE+Lovász fine-tune; focal γ=2; weight exp {0.12, 0.5} | [E] nnU-Net / Ma compound loss; [J] for this dataset |

```yaml
# v1 pseudo-config (spec only, not code)
data:   {input: hyper, target: labels_hyper_lt, split: official, norm: per_band_zscore_train, ignore_index: 255}
token:  {band_group: 16, patch: 8, embed: shared_linear, spectral_pe: learned_group, spatial_pe: sincos_2d}
model:  {dim: 64, heads: 4, mlp_ratio: 4, spectral_blocks: 2, spatial_blocks: 6, pool: mean_over_groups,
         norm: pre_ln, drop_path: 0.1, head: {proj: 32, up: 2x(bilinear2+conv3x3)+bilinear2, skip: conv1x1_224to16, dropout: 0.1}}
optim:  {name: adamw, lr: 1.0e-3, betas: [0.9, 0.999], wd: 0.05, no_wd: [norm, bias, pos/group_embed]}
sched:  {warmup_epochs: 5, type: cosine, min_lr: 1.0e-5, max_epochs: 200}
train:  {batch: 8, amp: fp16, clip_grad: 1.0, early_stop: {metric: val_fg_miou, patience: 40, min_epochs: 60}}
loss:   {ce_weights: median_freq(labels_hyper_lt_train, exp=0.12), dice: {classes: [1,2,3,4,5,6], weight: 1.0}}
```

---

## 1. Spectral tokenization, patch size, resolution

**What the published bars did (reference).** From the paper (§V-A) and the repo (`scripts/train_model.py`, `models/__init__.py`, `datasets/spectralwaste.py`, commit 77289f5):

| | MiniNet-v2 HYPER (52.8) | SegFormer-B0 HYPER (54.3) |
|---|---|---|
| Optimizer | Adam, LR 1e-3, wd 1e-4 | AdamW, LR 1e-3, wd = torch default (0.01) |
| Schedule | `PolynomialLR(max_epochs, power 0.9)`, stepped per epoch, no warmup | power **0.1** (LR is still ≈0.79e-3 at epoch 180), no warmup |
| Epochs / batch | 200 / 12 | 200 / 12 |
| Aug | RandomRotation(30) + V/H flips (torchvision v2, fill 0) | same |
| Loss | CE, `median_frequency_exp(counts, 0.12)`, counts **hard-coded from RGB labels** | same |
| Selection | best val mIoU checkpoint → test | same, drop-path 0.1, head dropout 0.1 |
| Params / GFLOPs | 0.585 M / 3.43 | 4.06 M / 6.35 |

The repo loads **no pretrained weights** for SegFormer or MiniNet (the `init_weights` loader is commented out). So **54.3 looks like a from-scratch transformer with about 8× our parameter budget**. That is inferred from the code; the paper does not say so.

**Band grouping [E precedent, J size].** Grouping neighbouring bands into one token is established: SpectralFormer's group-wise spectral embedding and SatMAE's spectral band groups. Per-band tokens (SSFT, ChannelViT) are the other extreme. Consecutive FX17 bands are ~3.6 nm apart (800 nm / 224), so *g* = 16 gives ~57 nm per token. With 14 tokens per patch, spectral attention is cheap. The trade-off: smaller *g* means more tokens and more spectral FLOPs. Larger *g* means fewer tokens but a bigger shared embed (*g*·P²·D).

| g | G tokens | embed params (P=8, D=64) | approx. fwd GFLOPs, whole model* |
|---|---|---|---|
| 8 | 28 | 33 k | 11.2 |
| **16** | **14** | **66 k** | **8.1** |
| 32 | 7 | 131 k | 6.6 |

\*Our own formula estimate for D=64, 2+6 blocks, counting FLOPs as 2×MACs (the paper measured with `torch.profiler`). Reference points: SegFormer-B0 HYPER 6.35, MiniNet HYPER 3.43. **This matters for Phase 3 latency:** P=16 drops the default to ~3.9 GFLOPs.

**Reduce first? [E].** PCA-3 barely hurt SegFormer-B0 in the paper (HYPER3 53.5 vs HYPER 54.3; MiniNet 49.0 vs 52.8). Keep PCA-3 or a learned 1×1 conv (224→32) as a **control**, not the default, because band tokens are the Phase-1 deliverable. Gotchas: `hyper_pca3/` is **not in the Zenodo zip**. `dim_reduction.py` fits PCA on the first 100 files from an *unsorted* `iterdir()`, and by default its scores are not normalized. If you build it, sort the files, fit on train only, and standardize.

**Patch size and resolution [J].** Keep the native 256², because the thin classes (video_tape, filament) vanish when downsampled. At P=8 the spatial sequence is 1024 tokens. Attention there costs about 4 MB per image per layer in fp16 at 2–4 heads, so batch 8 fits on a 16 GB T4. P=4 (4096 tokens, 16× the attention memory) needs SegFormer-style spatial-reduction attention (R=4) or windows. P=16 is the cheap or Phase-3-friendly option. Expect it to cost thin-class IoU unless the decoder skip is kept.

## 2. Width, depth, heads, MLP ratio under ~0.5 M

Per pre-LN block ≈ (4 + 2r)·D² params. Totals include the tokenizer (g=16, P=8), group embedding, and the ~26 k decoder (formula count, not a built model):

| D | heads | MLP r | spectral + spatial blocks | params/block | **total** |
|---|---|---|---|---|---|
| 64 | 4 | 4 | 2 + 4 | 50 k | 0.39 M |
| **64** | **4** | **4** | **2 + 6** | **50 k** | **0.49 M ← v1** |
| 80 | 4 | 4 | 2 + 4 | 77 k | 0.58 M |
| 96 | 3 | 2 | 2 + 4 | 75 k | 0.58 M |
| 96 | 3 | 4 | 1 + 3 | 112 k | 0.58 M |
| 96 | 3 | 4 | 2 + 4 | 112 k | 0.80 M (over budget) |
| 128 | 4 | 2 | 1 + 3 | 132 k | 0.69 M |

Why D=64, 4 heads [E]: this is SSFT's exact setting for HSI at 0.516 M. MLP ratio 4 is the ViT/DeiT/SegFormer-B0 standard. Depth over width at this size [J]. Steiner et al. find that at a fixed compute budget a larger patch size beats a thinner model ("prefer increasing patch-size to shrinking model-size", classification). That is a reason to try P=16 before cutting D below 64, though for segmentation larger patches cost boundary detail. Use pre-LN (Xiong et al.), which trains more stably and needs less warmup.

## 3. Positional / wavelength encoding

- **Spatial [E]:** fixed 2D sin-cos has zero params and nothing to overfit. SegFormer shows that segmentation can drop PE in favour of **Mix-FFN** (a 3×3 depthwise conv inside the FFN), with less sensitivity to resolution. CPVT (Chu et al.) makes the same argument. Mix-FFN is WP7's first locality knob.
- **Spectral [E]:** a learned per-group embedding (G×D), as in SSFT's learned channel embedding and ChannelViT.
- **λ-aware PE (optional, cite-only):** LESSViT's SSRoPE uses each band's centre wavelength. With one FX17, the wavelength table is fixed, so a learned group embedding carries the same information. Use λ-PE only if the team wants the cite-not-invent demo. **It is not the novelty.**

## 4. Small-data regularization (514 images)

- **Prefer augmentation to model regularization [E].** Steiner et al.: when data is small, almost every AugReg setting helps, and augmentation helps more often than dropout or stochastic depth. **Caveat [E]:** on HSI *classification*, SSFT found every tested augmentation (flip, rotate, crop, band-block drop, gain/offset, shading) slightly *hurt* overall. HSI augmentations must respect the sensor physics (Nalepa et al.).
- **v1 = paper's geometric augmentation (rot ±30°, H/V flips) plus rot90** [E]. Fix [J]: torchvision's `RandomRotation` fills with 0. That turns rotated-in corners into **background labels**, and 0 is out of distribution for the cube (P0 sample min ≈ 0.104). Fill the mask with **255 = ignore** via the v2 per-type `fill` dict.
- **WP7, add only if train ≫ val:** global gain ×U(0.9, 1.1) for illumination; **band-group dropout** (drop 1–2 group tokens, p=0.2). ChannelViT's hierarchical channel sampling is the precedent and was a strong regularizer there. Random resized crop with scale [0.5, 1]→256 is standard for SegFormer but was the worst augmentation in SSFT. All three are [J] on this data.
- **Stochastic depth 0.1 [E]** (Huang et al.; DeiT; the repo's own SegFormer-B0 used drop_path 0.1). **Dropout 0** in blocks (DeiT), 0.1 before the classifier (repo head).
- **Weight decay 0.05 [E]** (DeiT, decoupled per Loshchilov & Hutter). Steiner's ranges are 0.03–0.1.
- **Label smoothing 0 in v1, sweep 0.05/0.1 [J].** The LT labels are noisy (79.6% agreement ceiling), and Lukasik et al. show smoothing is competitive with loss correction under label noise. It interacts with class weights, so add it in WP7 only.
- **Hybrid conv stem [E for optimization].** Xiao et al.: a small stride-2 3×3 conv stem makes ViTs converge faster and much less sensitive to LR/wd. Lee et al. (SPT + LSA) add locality for small datasets. SSFT's ablation says spatial modeling contributed most. It is **not in v1**, to keep the Phase-1 "band-token transformer" clean, and it is the top fix if v1 underfits (§7).

## 5. Optimizer, schedule, batch, epochs, precision, stopping

- **AdamW, LR 1e-3 [E for this dataset].** The repo trained 4 M-param SegFormer-B0 from scratch with AdamW at 1e-3 and batch 12, at a nearly constant LR. Do **not** use DeiT's linear rule (5e-4 × bs/512 ≈ 8e-6 at bs 8). It targets 1024-image batches.
- **Warmup 5 epochs then cosine to 1e-5 [E]** (DeiT; Goyal et al. on warmup; SGDR cosine). Step per iteration (~65 iter/epoch at bs 8).
- **Batch 8, 200 epochs [E]** (paper: 200 epochs; SSFT: bs 8). For WP7 screening, run 60-epoch screens and only run finalists to 200 [J].
- **Mixed precision [E]** (Micikevicius et al.). Colab T4s have no bf16 tensor cores, so use fp16 with GradScaler there. Upcast attention logits to fp32 before softmax [J]. **Grad clip at norm 1.0 [E]** (ViT, Steiner).
- **Selection [E]:** evaluate val FG mIoU every epoch and keep the best checkpoint (repo practice). Patience 40 with a 60-epoch minimum [J], because cosine runs often improve late. Run test **once**, on the best-val checkpoint. **Final table numbers: 3 seeds, mean ± sd** (SSFT practice). The paper reports one number per model, and its seed variance is unknown.
- **I/O note [J]:** a uint16 train epoch is about 15 GB of reads. Convert the TIFFs once to per-split uint16 `.npy` memmaps on Colab local disk, and cast to float on the GPU.

## 6. Loss for the rare class 5 (filament)

Train pixels in `labels_hyper_lt` (TEAM-PLAN/P0): bg 81.97%, film 5.40%, basket 3.15%, cardboard 0.65%, video_tape 1.18%, **filament 0.098% (33,101 px)**, bag 7.55%. Weights are `(median/count)^exp` with median = 1,060,150:

| exp | bg | film | basket | cardb. | v_tape | **filament** | bag |
|---|---|---|---|---|---|---|---|
| **0.12 (repo)** | 0.676 | 0.937 | 1.000 | 1.207 | 1.125 | **1.516** | 0.900 |
| 0.5 | 0.196 | 0.763 | 1.000 | 2.192 | 1.633 | 5.659 | 0.646 |
| 1.0 (Eigen & Fergus) | 0.038 | 0.583 | 1.000 | 4.805 | 2.667 | 32.03 | 0.417 |

**Honest note:** at exp 0.12, weights from the RGB counts differ from the hyper_lt weights by **< 1%** (e.g. filament 1.525 vs 1.516). Swapping in hyper_lt counts is a *protocol-correctness* fix, worth one sentence on the slide. It does not move the score much. **The exponent is the real lever.**

**Default: CE (hyper_lt weights, exp 0.12, ignore 255) + soft Dice over classes 1–6, weight 1.0 [E general, J here].** Compute Dice over the batch and skip classes absent from both prediction and target. CE+Dice is nnU-Net's default across many segmentation datasets. Ma's loss survey concludes compound losses are the more robust choice for imbalance. Dice (Milletari et al.) directly counters the 82% background. Use the same loss for the WP4 CNN control so the comparison stays fair. Run CE-only once if you want exact paper-protocol parity.

**Alternatives (WP7):**
- **Lovász-softmax** (Berman et al.) is a convex surrogate of per-class IoU. Berman et al. note that batch IoU ≠ dataset mIoU and compute it only over classes present in the batch. Best used as a **fine-tune for the last 30–50 epochs** [J].
- **Focal γ=2** (Lin et al.) down-weights easy background pixels.
- Raise the CE weight exponent to 0.5.

## 7. Decision tree if v1 is ≤ 40 val mIoU or well under the CNN control (> 5 pts below)

```
0. Bug check first (§8 items 1–6). Run the CNN control through the SAME harness/loss. Still bad → go on.
1. Log train-subset FG mIoU (50 fixed train images, no aug) next to val mIoU.
   ├─ UNDERFIT (train < ~60 and close to val):
   │    a. LR sweep {5e-4, 2e-3}, warmup 10            [Smith LR range test]
   │    b. add locality: Mix-FFN dwconv → conv stem     [SegFormer, Xiao, Lee; SSFT spatial>spectral]
   │    c. finer tokens: P 8→4 with SR-attention R=4    [SegFormer]
   │    d. widen D 64→96 (≤0.8M, report params)
   ├─ OVERFIT (train − val > ~20):
   │    a. spectral aug on: gain, band-group dropout    [ChannelViT HCS]
   │    b. drop-path 0.2, wd 0.1, label smoothing 0.1   [DeiT, Steiner, Lukasik]
   │    c. random resized crop; d. shrink depth 2+4
   ├─ THIN CLASSES DEAD (video_tape or filament IoU < 10 while big classes OK):
   │    a. keep/enable the full-res skip; P=4
   │    b. CE weight exp 0.12→0.5; then Lovász fine-tune or focal γ=2
   └─ UNSTABLE (spikes/NaN, val swings > ±5):
        lower LR, warmup 10, fp32 softmax, QK-LayerNorm [Dehghani et al.], confirm pre-LN
2. Tokenizer check: same backbone on a 1×1-conv 224→32 reduction (or PCA-3).
   If reduction beats band tokens by > 2 pts → band grouping is the bottleneck: try g = 8 / 32. Report it honestly.
3. Stop rule (~Nov 1): freeze the best run and report the honest gap (TEAM-PLAN: < 52.8 is normal).
   The locality fixes carry into Phase 2 as SSFT's spatial-conv branch.
```

Per-class expectations from the paper (HYPER, LT labels): filament 30.5 (MiniNet) / 33.1 (SegFormer); video_tape 28.8 / 21.9; cardboard 78.8 / 85.2. Single-digit thin-class IoU means step 1's thin-class branch applies.

## 8. Sanity checklist (run before reporting any number)

1. **Overfit one batch:** 2–4 train images, no aug, no regularization, constant LR. Train FG mIoU should exceed ~95% within a few hundred steps (Karpathy's "overfit one batch"). If it doesn't, the model/loss/labels are wired wrong.
2. **Shapes:** logits are (B, 7, 256, 256), masks are (B, 256, 256) int64 in {0…6, 255}. Eval uses `model.eval()` and no aug.
3. **Mask–cube alignment after aug:** overlay one augmented sample in false colour (repo uses bands 134/170/200) and confirm the ignore-255 corners.
4. **Metric:** reported mIoU = `IoU[1:].mean()` (bg excluded). P0's `miou.py` matches torchmetrics to <1e-6. An all-background predictor must score FG mIoU = 0.
5. **Class weights** printed at startup equal the §6 hyper_lt values. Normalization stats and any PCA are fit on **train only**.
6. **Per-class IoU (all 6 FG plus bg) logged every epoch**, with a val confusion matrix saved for the best checkpoint.
7. Param count printed and in the results CSV, alongside the CNN control's count.
8. Required ablations in the table: with/without decoder skip; spectral-only (0 spatial blocks). The second becomes the Phase-2 spectral-only row.
9. Test evaluated once per final config. 3 seeds for the reported row.
10. The results slide says HYPER-only and labels_hyper_lt weights, and lists CMX-B0 58.2 as RGB+HSI context.

**Attention implementation (resolved 2026-10-06):** Built-in PyTorch pieces are OK per Jeong (as relayed by Scott), with customization preferred. `F.scaled_dot_product_attention` is fine inside our own QKV block. The model we report uses our own tokenizer and `src/sw/attention.py`, and `nn.TransformerEncoderLayer` stays a starting point and comparison row (see `docs/PLAN.md`).

---

## References (all checked on arXiv/Crossref Oct 6, 2026)

- Casao, Peña et al., *SpectralWaste Dataset: Multimodal Data for Waste Sorting Automation*, IROS 2024. [arXiv:2403.18033](https://arxiv.org/abs/2403.18033) · code [ferpb/spectralwaste-segmentation](https://github.com/ferpb/spectralwaste-segmentation) · data [doi:10.5281/zenodo.10880544](https://doi.org/10.5281/zenodo.10880544)
- Musiat, Ebert, Wasenmüller, *SSFT: A Lightweight Spectral–Spatial Fusion Transformer for Generic Hyperspectral Classification*, 2026. [arXiv:2604.15828](https://arxiv.org/abs/2604.15828) (0.516 M; D=64, 4 heads, s=8; AdamW 4e-4, wd 1e-2, bs 8, ≤50 ep, patience 10; classification, not segmentation)
- Si et al., *LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift*, 2026. [arXiv:2605.18541](https://arxiv.org/abs/2605.18541)
- Hong et al., *SpectralFormer*, IEEE TGRS 2022. [arXiv:2107.02988](https://arxiv.org/abs/2107.02988) · [doi:10.1109/TGRS.2021.3130716](https://doi.org/10.1109/TGRS.2021.3130716)
- Cong et al., *SatMAE*, NeurIPS 2022. [arXiv:2207.08051](https://arxiv.org/abs/2207.08051)
- Bao et al., *Channel Vision Transformers (ChannelViT, HCS)*, ICLR 2024. [arXiv:2309.16108](https://arxiv.org/abs/2309.16108)
- Dosovitskiy et al., *ViT*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron et al., *DeiT*, ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Steiner et al., *How to train your ViT?*, TMLR 2022. [arXiv:2106.10270](https://arxiv.org/abs/2106.10270)
- Xiao et al., *Early Convolutions Help Transformers See Better*, NeurIPS 2021. [arXiv:2106.14881](https://arxiv.org/abs/2106.14881)
- Lee et al., *Vision Transformer for Small-Size Datasets* (SPT/LSA). [arXiv:2112.13492](https://arxiv.org/abs/2112.13492)
- Xie et al., *SegFormer*, NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
- Chu et al., *Conditional Positional Encodings for Vision Transformers*. [arXiv:2102.10882](https://arxiv.org/abs/2102.10882)
- Bertasius et al., *Is Space-Time Attention All You Need for Video Understanding?* (divided attention). [arXiv:2102.05095](https://arxiv.org/abs/2102.05095) · Ho et al., *Axial Attention*. [arXiv:1912.12180](https://arxiv.org/abs/1912.12180)
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Dehghani et al., *Scaling Vision Transformers to 22B Parameters* (QK-LayerNorm). [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
- Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) · *SGDR*. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- Goyal et al., *Accurate, Large Minibatch SGD* (warmup). [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Smith, *A Disciplined Approach to Neural Network Hyper-Parameters*. [arXiv:1803.09820](https://arxiv.org/abs/1803.09820)
- Micikevicius et al., *Mixed Precision Training*, ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Müller et al., *When Does Label Smoothing Help?* [arXiv:1906.02629](https://arxiv.org/abs/1906.02629) · Lukasik et al., *Does Label Smoothing Mitigate Label Noise?*, ICML 2020. [arXiv:2003.02819](https://arxiv.org/abs/2003.02819)
- Nalepa et al., *Hyperspectral Data Augmentation*. [arXiv:1903.05580](https://arxiv.org/abs/1903.05580)
- Eigen & Fergus, *Predicting Depth, Surface Normals and Semantic Labels…* (median-frequency balancing). [arXiv:1411.4734](https://arxiv.org/abs/1411.4734)
- Milletari et al., *V-Net* (Dice loss). [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
- Isensee et al., *nnU-Net*, Nature Methods 2021. [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z) · [arXiv:1809.10486](https://arxiv.org/abs/1809.10486)
- Ma, *Segmentation Loss Odyssey*, 2020. [arXiv:2005.13449](https://arxiv.org/abs/2005.13449)
- Berman et al., *The Lovász-Softmax Loss*, CVPR 2018. [arXiv:1705.08790](https://arxiv.org/abs/1705.08790)
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Karpathy, *A Recipe for Training Neural Networks* (blog, 2019): https://karpathy.github.io/2019/04/25/recipe/

**Not verified / estimates:** our GFLOPs and param numbers come from formulas, not a built model, so WP5/WP6 should re-measure. Colab wall-clock per epoch is not estimated. Seed variance of the published bars is unknown. "SegFormer-B0 was trained from scratch" is inferred from the released code, not stated in the paper.
