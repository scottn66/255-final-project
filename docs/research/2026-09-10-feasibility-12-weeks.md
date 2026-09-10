# Feasibility check: can "One Model, Any Camera" be built in 12 weeks?

**Date:** 2026-09-10 · **Companion to:** [2026-09-10-project-direction-review.md](2026-09-10-project-direction-review.md) · **New input:** the professor has sponsored Mobilint NPU lab equipment and wants the project to use it, so MLA100 hardware and compiler access are guaranteed rather than a risk. · **Method:** second deep-research pass (5 angles, 16 primary sources, ~80 claims; adversarial verification in progress, see Appendix A) plus direct checks of the Hugging Face dataset listing and the Mobilint GitHub guides.

> **Verdict.** The recommendation stands, with one substantive revision to the headline claim. **Confirmed:** compact wavelength-conditioned spectral–spatial segmentation for robotic recycling, deployed INT8 on the MLA100, is buildable and evaluable in 12 weeks by a team of 2–4, using the labeled SpectralWaste subset (25 GB) as the primary dataset. **Revised:** "train on one sensor, test zero-shot on a sensor with a different wavelength range" is not credible and must not be the headline. Every 2025–2026 source that tested it (GeoCrossBench, SMARTIES, LESSViT, CARL, HyperFree) either reports collapse or restricts its claims to overlapping ranges. The headline becomes **spectral-configuration robustness within the trained range, few-shot adaptation to a new camera, and preservation of both under INT8 on the NPU**, with disjoint-range transfer kept as an explained negative control. The single most important schedule action is an NPU compile smoke test in week 1, not week 7.

---

## 1. What changes and why

| Original plan element | Evidence | Decision |
|---|---|---|
| Zero-shot VNIR → SWIR → MWIR transfer via wavelength PE | GeoCrossBench (ICLR 2026 ML4RS workshop): with no band overlap, every band-flexible model (DOFA, ChannelViT-style ChiViT, DINOv3, TerraFM) loses 2–4× of its performance; even adding extra test-time bands costs 5–25%. SMARTIES (ICCV 2025) states wavelength interpolation works only inside the pretraining min/max and "not for unseen regions out of the pretraining spectra." LESSViT's disjoint C82 setting is its hardest and is described as unsolved. CARL's every cross-camera test is inside 418–2445 nm. HyperFree's dictionary covers only 400–2500 nm. MWIR-4-Plastic's own physics: carbon black absorbs NIR/SWIR, and the FX50 measures reflected plus thermally emitted radiance, so MWIR features differ in kind, not just position. | Drop as headline. Keep FENIX → FX50 as a negative control with the physics explanation. |
| HyperPlastic as an "overlapping-range" cross-camera set | The released fused cubes have no overlap: VIS-NIR keeps 134 bands at 489–850 nm and NIR-SWIR keeps 98 bands at 980–1670 nm. It is also patch classification (1,220 labeled 64×64 patches), and its scene (plastics floating in water) differs from recycling lines. | Optional only, as a cheap patch-level robustness testbed. Not needed for the paper. |
| SpectralWaste as a segmentation testbed | Labeled subset is 25 GB on Hugging Face (train 15.2 GB / val 4.9 GB / test 5.1 GB), already aligned and resized to 256×256; the unlabeled subset is 201 GB and unnecessary. Paper checkpoints exist for MiniNet, SegFormer-B0 and CMX-B0, so baselines can be evaluated test-only without retraining. Caveats: no band-center wavelengths ship with the HF release or with either of the authors' code repos (verified by cloning and grepping), so the model must use the Specim FX17 nominal grid (224 bands over 900–1700 nm) as a documented approximation or request the ENVI header from the authors; HSI labels are transferred from RGB with a measured 79.6% mIoU label ceiling, license is CC BY-NC 4.0, and PCA-3 input loses only ~0.8 mIoU (SegFormer-B0 53.5 vs 54.3), so the model must beat a 3-channel projection. | Primary dataset. Compare against MiniNet-v2 HYPER (52.8 mIoU, 0.585M params) as the like-for-like sub-1M baseline and CMX-B0 (58.2 mIoU, 11.5M) as the fusion ceiling. |
| MWIR-4-Plastic as the second dataset | 13 scenes, 4 polymers, fixed split with only 5 test cubes; FENIX (450 bands, 380–2500 nm) and FX50 (308 bands, 2.7–5.3 µm) share one 640×1114 grid, and the RGB cube is stored at exactly 6× that resolution so a stride-6 subsample aligns it; the released baselines use only 143 "fingerprint" bands (1600–2200 nm plus five MWIR windows) and carve validation from the test cubes (leakage to fix). ViT and SpectralFormer baselines are 0.23–0.26M parameters, trained on 9×9 patches, hours on a Colab GPU. Zenodo size and license could not be verified from this environment (estimate 25–30 GB uncompressed). | Second dataset, used for (a) within-FENIX configuration shift, (b) few-shot new-camera adaptation of the SpectralWaste-trained spectral path to the FENIX SWIR window (970–1700 nm overlaps FX17's 900–1700 nm), and (c) the FX50 negative control. Carve our own validation from training cubes. |
| NPU deployment in weeks 7–8 | Compile requires the qbcompiler wheel from Mobilint's Download Center, Docker, and for ARIES an actual card in the compile host (`--device /dev/aries0`). Calibration is mandatory, `.npy` tensors must match the model input shape, 100–128 samples recommended, and multi-channel calibration is documented (Qwen3-VL uses 6-channel HWC; BERT uses embedded tensors). The BERT tutorial compiles LayerNorm/softmax/GELU attention with `cpu_offload=True`, but embedding lookups and the input LayerNorm ran on CPU, and the vendor's own transformer examples needed graph surgery. There is no published operator list, opset, latency figure, or segmentation-transformer example, and the public repos have zero GitHub issues. | Move the compile smoke test to week 1 on the lab machine. Use the professor's vendor channel for support. |

Sponsored equipment changes the risk profile in two ways. The compiler-access risk from the first memo disappears. In exchange, the lab machine is on the critical path: it must have Docker, the Mobilint driver, the SDK wheel, and scheduled time slots, because ARIES compilation cannot be done on a laptop.

## 2. Feasibility by workstream

### 2.1 Data logistics: feasible, one weekend

| Dataset | Download | Preprocessing | Notes |
|---|---|---|---|
| SpectralWaste labeled (HF WebDataset) | 25 GB, 27 shards; Hub viewer is broken so pull shards directly with the `webdataset` library | None beyond TIFF decode; cubes are (224, 256, 256) uint16 | Wavelengths: FX17 nominal grid as an approximation, or ask the authors for the header; split by `sequence_id`, never by frame |
| MWIR-4-Plastic (Zenodo 22142947) | Size unverified, likely 25–30 GB as ENVI cubes | Loader exists; reconcile `semantic_mask.png` vs `semantic_mask2.png` filename mismatch; subsample RGB by 6 to land on the HSI grid | Start the download in week 1 as a background task |
| HyperPlastic (optional) | ~2–5 GB | None | Patch classification only |

### 2.2 Model and training compute: feasible on one Colab-class or lab GPU

- **Baselines are cheap or free.** SpectralWaste recipes are 200 epochs of B0-class models on 514 images at 256×256, hours on one GPU, and paper checkpoints allow test-only reproduction. MWIR-4-Plastic baselines are 0.23–0.26M-parameter patch classifiers at batch 512, hours on Colab.
- **Per-pixel spectral attention over 224 bands must be factorized.** LESSViT reports that joint spatial-spectral attention (ChannelViT) runs out of memory near 200 channels even on a 144 GB GPU. Our own estimate: one spectral MHSA + FFN block at D=64 costs roughly 17 MFLOPs per pixel at C=224, about 1.1 TFLOP per 256×256 cube, which is trainable but wasteful. Three mitigations are in the design already or are one-line changes: spatial max-pool by s=4 (16× fewer spectral tokens, as in SSFT), random band sub-sampling per training step (CARL: 16 of 100 bands cut training FLOPs ~75% with 68.8 vs 69.1 mIoU, and it doubles as band-drop augmentation), and 64×64 patch-wise training with full-frame inference.
- **Accuracy target that justifies the design.** Beat MiniNet-v2 HYPER (52.8 mIoU at 0.585M params) and the PCA-3 variants at similar cost, then show the configuration-shift advantage. Do not plan to beat CMX-B0 or BCAF.

### 2.3 Cross-sensor claim: feasible only in the revised form

Three tiers, in order of credibility:

1. **Within-sensor configuration shift (headline).** LESSViT's protocol shrunk to SpectralWaste (LESSViT is an unrefereed May 2026 preprint, so cite it as a design precedent, not as a standard): train on a fixed 120-band subset that is short-wavelength-heavy, test on the same subset, on the long-wavelength-heavy complement of equal size, on the fully disjoint 104-band complement, and on all 224 bands. Report in-distribution mIoU and relative drop per setting, three seeds. Expect the fixed-channel baseline to win in-distribution and to lose under shift, exactly as LESSViT reports.
2. **Simulated cross-camera transfer by spectral-response-function resampling (recommended addition).** LESSViT's authors added exactly this to their repo after the preprint: they resample EnMAP cubes through other instruments' spectral response functions to build `prisma_like`, `sentinel2_like`, `desis_like` and `eo1h_like` test configurations, then evaluate the same fixed-configuration checkpoint on them, and they validate against real DESIS (235 bands, 402–999 nm) and EO-1 Hyperion (198 bands, 427–2396 nm) test sets. Training always stays on the native configuration; only validation and test vary. Applied here: resample SpectralWaste's 224 FX17 bands through coarser or shifted response functions to simulate cheaper SWIR cameras (for example a 25-band snapshot mosaic, or a 12 nm-resolution instrument), and test the same checkpoint on each. This is a genuine "any camera" result, costs one afternoon of signal processing, needs no second dataset, and removes the scene-change confound that the few-shot experiment below carries. **Make this the second headline experiment.**
3. **Few-shot new-camera adaptation.** Adapt the SpectralWaste-trained spectral path to MWIR-4-Plastic's FENIX SWIR window with K = 1, 2, 4 training cubes versus training from scratch. This confounds sensor change with scene change, so present it as the recycling-relevant "new plant, new camera, K labeled cubes" question, not as a pure sensor result.
4. **Disjoint-range negative control.** FENIX → FX50 zero-shot, expected to collapse, reported with the physics explanation. Reviewers reward an honest, explained failure case.

Two cheap controls reviewers recognize: shuffling the wavelength vector (HyperFree drops from 93.4 to 75.3 OA) and removing the wavelength PE entirely (CARL drops from 61.5 to 18.3 mIoU on mixed cameras).

### 2.4 NPU deployment: feasible, but front-load it

What the vendor material establishes and what it does not:

| Established | Not established (must be tested in week 1) |
|---|---|
| Standard ViT/DeiT/Swin blocks compile and quantize on ARIES with under 0.5-point loss; worst cases on small or hybrid models lose 1.0–1.5 points, so budget 1–2 points | Whether Reshape/Transpose patterns that turn a (1, 224, H, W) cube into per-location band tokens compile, and whether they stay on the NPU |
| Multi-channel `.npy` calibration inputs are accepted when their shape matches the model input | Whether a 224-channel image input is handled by the image path or must be supplied as a generic tensor; whether the NHWC boundary needs a layout change |
| Quantization knobs: per-channel weights, multi-layer activation statistics, percentile clipping (0.999 / top-k 0.01), histogram/KL | Compile time for a custom graph; per-layer INT8 error on spectral attention |
| `cpu_offload=True` rescues unsupported layers instead of failing | How much of a custom graph lands on CPU (inspect with Netron) |
| `mblt-tracker` gives NPU power, utilization, memory, temperature with avg/max/p99; latency is wall-clocked around `Model.infer` | Nothing about latency is published; every number will be ours |

Design rules that follow: export a static batch-1 graph; pool spatially before the spectral block so the token tensor is small (at s=4, 4,096 locations × 224 bands × 64 dims); bake the wavelength PE in as a constant; implement band drop as a constant mask multiply plus additive attention mask so shapes never change; keep the tied embedding as a 1×1 convolution rather than a gather. Calibration cubes are ~59 MB each in float32, so 100 samples is ~6 GB; PTQ4ViT shows 32 samples calibrate a small ViT to within 0.4 points in minutes, so start with 32 spatial crops.

If INT8 accuracy drops more than 2 points: percentile or histogram calibration first, then SmoothQuant folded offline into the ONNX weights (exact transform, no runtime cost, compiler-agnostic), then per-layer offload of the softmax path. Stay at W8A8; W6 collapses unpredictably on small ViTs.

### 2.5 Precedents for the minimum acceptable result

HyperspectralViTs (JSTARS 2025) reports one edge table: per-device latency (0.64 s per EMIT granule on Jetson AGX Xavier via TensorRT), parameter count, no INT8, no measured energy. HYPSO's onboard classifier papers report one model, one dataset, one latency/power/energy table with a named GPU baseline (5.4 W, 21.5 mJ per inference). TinyCapsViT (Aug 2026) reports a 2,781-parameter model with Raspberry Pi and Jetson timing and no INT8 drop. None of them reports robustness after quantization. The team's differentiator is measured INT8 accuracy plus band-drop robustness on a real segmentation task.

## 3. Twelve-week plan with the critical path

Assumes Sep 10 start, class deliverable Dec 3–10, 2–4 people. Critical path in bold.

| Week | Build | Experiment / evidence | Owner pattern (3 people) |
|---|---|---|---|
| 1 | **NPU smoke test:** export the existing `SpectralStub` (C=32, 16×16) to ONNX, compile in the lab Docker with 32 random calibration cubes, run on the MLA100, inspect NPU vs CPU placement. Download SpectralWaste labeled and start MWIR-4-Plastic. | Reproduce SegFormer-B0 / CMX-B0 numbers test-only from checkpoints. Freeze the protocol document: sequence-level splits, the four band configurations, metrics (mIoU excluding background, macro-F1, relative drop). | A: NPU · B: data + baselines · C: protocol + tickets |
| 2–3 | **Finish SSFT-Wave** (open tickets SSFT-2/3): spectral path with s=4 pooling and random band sub-sampling, spatial path, fusion, per-pixel decoder. | HSI-only training on SpectralWaste, fp16. Target: above 52.8 mIoU at under 1M params; report against PCA-3. Wrong-wavelength and no-PE controls. | B+C: model · A: calibration pipeline for 224-band cubes |
| 4 | Index-PE and fixed-channel variants; retrain MiniNet or SegFormer-B0 on the 120-band subset with zero-fill at test. **SRF resampler** (Gaussian response functions at target band centers and widths). | **Configuration-shift table** on SpectralWaste, three seeds, plus the **simulated-camera table** (native FX17 vs 3–4 resampled instruments, same checkpoint). | B: training · C: analysis |
| 5 | MWIR-4-Plastic loader with a clean validation split; reproduce its ViT / SpectralFormer baselines. | Few-shot new-camera curve (K = 1, 2, 4) on the FENIX SWIR window; FX50 negative control. | C |
| 6–7 | **Compile the full model:** static export, 32–100 calibration crops, MXQ; ONNX Runtime CPU parity harness. | **NPU table:** INT8 vs FP32 mIoU, batch-1 latency (p50/p99 over 100 runs), power via mblt-tracker, energy per frame; band drop after quantization. Apply calibration fixes if the gap exceeds 2 points. | A (+B for accuracy debugging) |
| 8 | Ablations: fused vs spectral-only vs spatial-only; index vs wavelength vs dictionary embedding. Optional cut-line item: RGB+HSI fusion variant. | Ablation table. | B |
| 9–10 | Writeup, figures, class presentation. | Class deliverable. | All |
| 11–12 | Buffer. arXiv preprint; IGARSS abstract once the CFP posts (2026 pattern: mid-January); IROS 2027 extension plan (deadline Mar 1, 2027). | | All |

Critical path: week-1 compile smoke test → model complete by end of week 3 → shift table in week 4 → NPU table by end of week 7. Weeks 5 and 8 are parallel work that can be cut without losing the paper.

## 4. Top five schedule risks

| Risk | Likelihood | Mitigation | Cut-line if behind |
|---|---|---|---|
| Compiler rejects the token-reshaping graph or offloads most of it to CPU | Medium | Week-1 smoke test; restructure as 1×1 conv + Conv1d over the band axis; avoid 5-D tensors; `cpu_offload` for the embedding only; use the professor's vendor channel | Report FP32 on ONNX Runtime CPU plus PyTorch fake-quant INT8 (FQ-ViT) and drop the energy column; the paper becomes a robustness study with a deployment appendix |
| Cross-sensor headline collapses | Certain for disjoint ranges | Already reframed to within-range shift plus few-shot; disjoint transfer is a negative control | Keep only the within-sensor shift table and band drop |
| Training too slow or out of memory on 224 bands | Medium | s=4 pooling, random band sub-sampling, 64×64 patch training, fp16 | Reduce to 128 bands or use PCA-k for the spatial path; keep the spectral path on sub-sampled bands |
| Data logistics (MWIR-4-Plastic size and license unverified; labels noisy) | Medium | Start downloads in week 1; sequence-level splits; state the 79.6% label ceiling | SpectralWaste-only paper; MWIR-4-Plastic moves to the IROS extension |
| INT8 accuracy drop above 2 points | Low–medium | Percentile/histogram calibration, SmoothQuant fold, per-layer inspection | Report the drop with a per-layer error analysis; that is itself a finding for the Embedded Vision workshop |

## 5. Minimum viable paper (what the course deliverable must contain)

1. One dataset (SpectralWaste labeled), one compact model with a sub-1M parameter count, one like-for-like baseline (MiniNet-v2 HYPER) and one PCA-3 baseline.
2. One configuration-shift table (in-distribution, swapped subset, disjoint complement, full 224) with relative drop, three seeds, plus the wrong-wavelength control. One simulated-camera table: the same checkpoint on 3–4 SRF-resampled instruments.
3. One NPU table: FP32 GPU/CPU vs INT8 MLA100 mIoU, batch-1 latency, energy per frame, and band-drop robustness before and after quantization.
4. One limitations paragraph: label-transfer ceiling, non-commercial license, no disjoint-range transfer, single sensor family.

That is IGARSS-abstract or CVPR-workshop grade. Adding the few-shot new-camera curve on MWIR-4-Plastic, the FX50 negative control, and the ablations makes it an IROS 2027 submission.

## 6. Repo actions this week

- Retire tickets SSFT-4 (Indian Pines) and SSFT-6 (vanilla ViT). Rewrite SSFT-2/3 to target the pooled spectral path and per-pixel decoder. Add tickets: `DATA-1` SpectralWaste loader with wavelengths and sequence-level splits; `NPU-0` compile smoke test on the stub; `PROTO-1` protocol document.
- Add a `deploy/` directory with the ONNX export, calibration-crop generator, and a parity harness that runs the same cubes through ONNX Runtime and the MXQ runtime.
- Update `PLAN.md` locked defaults: input is a 256×256×224 cube with a wavelength vector; spatial pool s=4 before the spectral block; band sub-sampling on by default.

## 7. Sources

Cross-sensor credibility
- GeoCrossBench: Cross-Band Generalization for Remote Sensing (ICLR 2026 ML4RS workshop). https://arxiv.org/abs/2511.02831
- SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder (ICCV 2025). https://arxiv.org/abs/2506.19585
- LESSViT (May 2026). https://arxiv.org/abs/2605.18541
- CARL (ICLR 2026). https://arxiv.org/abs/2504.19223 · code https://github.com/IMSY-DKFZ/CARL
- HyperFree (CVPR 2025). https://github.com/Jingtao-Li-CVer/HyperFree
- MWIR-4-Plastic (Aug 2026). https://arxiv.org/abs/2608.28874 · code https://github.com/Elias-Arbash/MWIR-4-Plastic · data https://zenodo.org/records/22142947

Datasets
- SpectralWaste segmentation release. https://huggingface.co/datasets/ferpb/spectralwaste-segmentation · code https://github.com/ferpb/spectralwaste-segmentation · dataset tools https://github.com/ferpb/spectralwaste-dataset · paper https://arxiv.org/abs/2403.18033
- HyperPlastic (Zenodo 14773387). https://zenodo.org/records/14773387

Mobilint toolchain
- SDK tutorial (compile guides, BERT and Qwen3-VL examples, runtime). https://github.com/mobilint/mblt-sdk-tutorial
- Vision toolkit and NPU-vs-GPU table. https://github.com/mobilint/mblt-vision-python
- mblt-tracker (NPU power and utilization). https://github.com/mobilint/mblt-tracker
- MLA100 announcement. https://embeddedvisionsummit.com/posts/2025-04-mobilint-introduces-mla100-mxm-an-80-tops-npu-module-for-high-efficiency-embedded-ai-pcs/

Quantization and compute
- FQ-ViT (IJCAI 2022). https://github.com/megvii-research/FQ-ViT
- PTQ4ViT (ECCV 2022). https://github.com/hahnyuan/PTQ4ViT
- SmoothQuant for ONNX (neural-compressor docs). https://github.com/onnx/neural-compressor/blob/main/docs/smooth_quant.md
- Transformers Meet Hyperspectral Imaging (2025 survey). https://arxiv.org/abs/2506.08596

Precedents and venues
- HyperspectralViTs (JSTARS 2025). https://arxiv.org/abs/2410.17248
- HYPSO onboard hyperspectral classification accelerator (IEEE 2025). https://ieeexplore.ieee.org/document/11049208/
- TinyCapsViT (Remote Sensing, Aug 2026). https://doi.org/10.3390/rs18162661
- IROS 2027 deadline. https://mldeadlines.com/conference/iros-2027/ · IGARSS 2027. https://2027.ieeeigarss.org/

## Appendix A. Verification status

Eighteen extracted claims went to three adversarial verifiers each; seven more claims (the GeoCrossBench, SMARTIES and CARL cross-sensor findings) were queued but never voted on because the run hit a usage limit. Verifiers worked from primary artifacts (cloned repos, downloaded dataset shards, vendor guides) since arxiv.org is blocked from this environment.

**Confirmed 3–0 and used as stated.** SpectralWaste split sizes and the ~25 GB / ~201 GB labeled-vs-unlabeled footprint; the FX17 sensor spec (224 bands, 900–1700 nm); the published baseline table (MiniNet-v2 HYPER 52.8 mIoU at 0.585M params, SegFormer-B0 HYPER 54.3, CMX-B0 RGB-HYPER 58.2 at 11.5M, RTX 4090 batch-1 throughput); MWIR-4-Plastic's 13 scenes, four polymers, hard-coded split and 143-band "fingerprint" baselines; HyperPlastic's disjoint 489–850 nm and 980–1670 nm ranges and its patch-classification format.

**Corrected after refutation.**
- *Wavelengths.* The claim that the FX17 band centers can be fetched from the authors' code repo was refuted 3–0: verifiers cloned both `ferpb` repos and found no wavelength vector, header, or band table anywhere, and the shard TIFFs carry no wavelength tags. The memo now says to use the nominal FX17 grid as a documented approximation or to ask the authors.
- *RGB alignment in MWIR-4-Plastic.* "Not pixel-aligned" was refuted 2–3: the RGB cube is exactly 6× the HSI grid in both axes (3840×6684 vs 640×1114), so a stride-6 subsample lands on it. Corrected.
- *LESSViT protocol.* Refuted 2–3, on two grounds: calling it "reviewer-accepted" overstates an unrefereed preprint, and the description was outdated. The authors' repo now adds SRF-resampled sensor configurations and real EnMAP→DESIS and EnMAP→Hyperion evaluations. This is the source of the simulated-camera experiment added in section 2.3, and is the single most valuable thing this verification pass produced.
- *SpectralWaste distribution.* "Unlabeled data available only from OneDrive" was refuted 3–0 as an inference from omission; the 23 GB / 178 GB figures themselves hold. The memo relies on the Hugging Face release, so nothing changed.

**Never verified.** The cross-sensor evidence from GeoCrossBench (2–4× collapse with no band overlap), SMARTIES (interpolation only, not extrapolation) and CARL (all cross-camera tests inside 418–2445 nm) was extracted from those papers but the verifier panels never ran. Each was read from a primary source during the search phase and the three agree with each other and with the LESSViT and HyperFree evidence that did verify, so the direction of the conclusion is well supported; treat the specific percentages as unconfirmed.
