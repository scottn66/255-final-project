# Project direction review: is there a better DATA 255 project than SSFT-Wave?

**Date:** 2026-09-10 · **Status:** research memo for the team · **Method:** deep-research workflow (5 search angles, 21 primary sources fetched, 80 claims extracted, top 25 sent to 3-vote adversarial verification; see Appendix A for what survived and what was corrected) plus a Hugging Face Hub sweep of 2025–2026 papers, datasets, and Mobilint NPU artifacts.

> **Bottom line.** The professor's critique holds up against the 2026 literature. Wavelength-aware positional encoding and channel-agnostic embeddings are now a standard component of at least seven published models, the dual-path fusion is copied from SSFT, and Indian Pines / Pavia results are discounted by reviewers even with block splits. But the *engineering* the team has built (from-scratch attention, wavelength PE, band-mask forward) is exactly what a stronger project needs. **Recommendation: keep the code, change the problem.** Move to sensor-agnostic hyperspectral material segmentation for robotic recycling on public multi-sensor datasets, and add the one axis nobody has measured: INT8 deployment of spectral attention on the course's Mobilint MLA100 NPU, including robustness to dropped bands *after* quantization.

---

## 1. Why the critique is correct (evidence)

| Critique | What the record shows | Source |
|---|---|---|
| **Unoriginal** | Wavelength-conditioned, variable-channel HSI transformers are a well-populated line of work: DOFA (2024), HyperFree (CVPR 2025), CARL (DKFZ/Siemens, Apr 2025), HyperSL (TGRS 2025), SpecAware (ISPRS 2026), LESSViT (May 2026), HyperVision (May 2026). CARL, HyperFree, HyperSL and HyperVision report public code or weights; LESSViT has code but no checkpoints as of Jul 2026; a Copenhagen preprint (Laprade et al., Mar 2025) proposes the same wavelength-encoding idea for proximal sensing with no code found. LESSViT's *stated problem* is exactly "spectral configuration shift" and its solution is channel-agnostic tied embedding plus wavelength-aware rotary PE (verified 2–1). SSFT-Wave re-implements that idea at small scale. | LESSViT 2605.18541; HyperVision 2605.17286; CARL 2504.19223; HyperFree 2503.21841 |
| **Toy** | SSFT itself (CVPR 2026 PBVS workshop) was validated on HSI-Benchmark, the Tübingen DeepHS benchmark (~240 GB) whose three domains are fruit ripeness, debris material recognition, and a remote-sensing domain that is exactly Indian Pines, Salinas and Pavia University under class-stratified random-pixel splits, plus a SpectralEarth transfer test; it ranked first with <2% of the prior leader's parameters. The team's plan keeps one of those three domains and drops the transfer. The spatial block split is a genuine methodological improvement over the source's protocol, but on its own it makes the evidence narrower, not new. | SSFT 2604.15828, CVPRW 2026; cogsys-tuebingen/hsi_benchmark |
| **Toy (benchmark)** | "Ten Architectures, One Error" (Faghih, Ashrafi, Moore, Saki; unreviewed arXiv v1, Sep 1, 2026; abstract claims verified 3–0): under the customary random split ~47.7% of test pixels sit within one pixel of a training pixel; a leakage-free protocol drops Macro-F1 by 0.147 on average across ten architecture families and reshuffles rankings by up to five places. Crucially, all ten models misclassify largely the *same* pixels: the residual error is spectral ambiguity in the data, not something a new architecture fixes. A block split is only leakage-free if the separation radius is at least the receptive field, which must be reported. | arXiv 2609.01786 |
| **No captivating motivation** | "Degrades less under band drop" names no application. SSFT's own motivation (scarce labeled HSI outside earth observation) is inherited, not added. The 2026 GRSS author checklist requires the gap, contribution, and novelty justification to be stated explicitly and tied to real sensor variability (multiple scenes / sensors, disjoint splits). | GRSS checklist; SSFT abstract |
| **Wrong baseline** | LESSViT's code compares against DOFA, DINOv3, SpecViT, SpatSigma, ChannelViT and HyperFree, and reports average relative drops of 15% (spectral shift), 26% (unseen wavelengths) and 5% (channel expansion) across five SpectralEarth downstream products. A 2026 reviewer will not accept "vanilla ViT with zero-filled bands" as the only comparison for a robustness-under-band-shift claim. | LESSViT repo and project page |

One more fact worth internalizing: SSFT's ablation attributes most of its accuracy to the **spatial** pathway, and its robustness result is about training without augmentation, not about missing bands. SSFT-Wave modifies the weaker branch and measures an axis the source never studied. That is a gap, but it also means the measurable effect may be small.

## 2. What the 2025–2026 landscape leaves open

The searches surfaced a consistent set of open problems that are (a) unsolved, (b) tractable on Colab-class compute in 12 weeks, and (c) aligned with the team's interests in HSI, transformers, robotics perception, and NPUs.

| Open gap | Evidence it is open | Fit |
|---|---|---|
| **Compact HSI models on real industrial / robotic sorting data with edge deployment** | On SpectralWaste (IROS 2024) the original best fusion model, CMX-B0, scored 58.2% mIoU at 54.7 img/s with 11.5M parameters; the self-declared state of the art, BCAF (Mar 2026 preprint, verified 3–0), reaches 76.4% mIoU at 31.4 img/s but with 101.3M parameters and 282 GFLOPs on an RTX 4090. EWSegNet (Jun 2026) reportedly uses a 23.3M-parameter encoder. None reports edge deployment. WoodVIT and HSI-AgriFoodAnomaly (both Scientific Data 2026) ship only 4-layer CNN patch baselines with no transformer, no cross-sensor test, no latency numbers. | HSI + transformers + robotics + NPU |
| **Cross-sensor transfer on co-registered proximal sensors** | MWIR-4-Plastic (Aug 2026): 13 co-registered RGB / VNIR / SWIR / MWIR scenes of shredded black automotive plastics, 9-method benchmark with tiny ViT / SpectralFormer baselines at the same ~0.2M-parameter scale as SSFT-Wave. Cross-sensor transfer, band-drop robustness, and edge deployment are all unaddressed. | Direct reuse of `forward(cube, wavelengths, band_mask)` |
| **INT8 spectral attention on an NPU** | Mobilint compiles vanilla ViTs near-losslessly (ViT-B/16: 81.00 NPU vs 81.04 GPU top-1), so "a ViT runs on the MLA100" is not a contribution. But the public calibration path accepts only 3-channel HWC tensors, no public example compiles a 100–200-band cube, dynamic band masking is not expressible, and no latency, energy, or operator-support figures are published. TinyCapsViT (Aug 2026) deployed a tiny HSI transformer on Raspberry Pi / Jetson in FP32 only. HyperVision's authors reportedly name backbone efficiency as their main unsolved problem (verification pending). | NPU + transformers; the course's unique hardware |
| **Spectral information that actually beats RGB in driving / terrain perception** | HS3-Bench: a pseudo-RGB DeepLabV3+ (ImageNet MobileNetV2) averages 64.75 mIoU vs 57.68 for RU-Net on the full cube; on HSI-Drive the best result used PCA-1 alone. Cross-domain HyperSL transfer (ICPR 2026) gains up to +12 mIoU on HyKo2. Exploiting spectra in a compact model is stated as open by the benchmark authors. | Robotics perception |
| **Cross-sensor methane plume detection** | UNEP MARS dataset (EMIT / EnMAP / PRISMA, on Hugging Face) underpins an operational system (1,351 leaks verified in seven months). Models are trained per sensor; HyperspectralViTs runs on Jetson. A wavelength-conditioned model generalizing across the three sensors is untested. | HSI + NPU (on-board), weaker robotics fit |
| **Verifier-guided agentic experiment execution** | Independent evaluations of AI-scientist systems: 42% of experiments failed from coding errors, novelty checks labeled established techniques novel, output likened to "an unmotivated undergraduate." An MBZUAI/TUM/TU Berlin position paper (Apr 2026) argues earth-observation agents need structured state and verifier-guided execution; these are stated as principles, not implemented. | Agentic workflows, but high reviewer skepticism |

## 3. Candidate directions

Each candidate is scored on the five things the professor asked for. "Reuse" says how much of the current `src/` survives.

### Candidate A (recommended): Sensor-agnostic HSI material segmentation for robotic recycling, quantized to an edge NPU

- **Problem.** Recycling plants sort polymers on conveyors at speed; conventional NIR sorters fail on black plastics; every plant's hyperspectral camera has a different band count and wavelength grid (VNIR, SWIR, MWIR), so perception models are trained per camera and cannot be shared or upgraded when a sensor changes. Real-time inference must run on an embedded accelerator, not a GPU.
- **Gap / novelty.** (1) First cross-sensor evaluation (train VNIR, test SWIR / MWIR and vice versa) of a compact wavelength-conditioned segmenter on co-registered industrial plastics. (2) First INT8 NPU deployment of spectral (channel-axis) attention with latency, energy, and post-quantization band-drop robustness measured. (3) A documented hyperspectral calibration path for an RGB-only NPU toolchain. (4) Honest comparison against the channel-agnostic baselines a 2026 reviewer expects (index-PE, HyperFree-style wavelength dictionary embedding, SpectralFormer, 1D-CNN), not just a vanilla ViT.
- **Datasets (all public, all Colab-scale).** MWIR-4-Plastic (Zenodo 22142947; 4 co-registered sensors, held-out cube per class). SpectralWaste (Hugging Face `ferpb/spectralwaste-segmentation`, CC BY-NC 4.0 per its LICENSE file; Specim FX17, 224 SWIR bands 900–1700 nm; 852 labeled + 6,801 unlabeled; published baselines run from 58.2% mIoU for an 11.5M-parameter model to 76.4% for a 101M-parameter one). BCAF's companion K3I-Cycling set (354 co-registered pairs, 205-band HSI) has released only its RGB subset so far. Optional third: HyperPlastic (VIS-NIR + NIR-SWIR, 232 bands) or HSI-AgriFoodAnomaly (300 VNIR bands, foreign-object masks).
- **Approach.** SSFT-Wave modules become a per-pixel or small-patch segmenter: tied `Linear(1,D)` embed, sinusoidal λ-PE, spectral MHSA over C, spatial conv path, cross-attention fusion. Train on one sensor's wavelength grid; test zero-shot and few-shot on the others by passing the new sensor's wavelength vector. Then ONNX export, MXQ compile for the ARIES target (`aries-rb`), custom multi-channel calibration set, measure INT8 vs FP32 accuracy, latency, and power with `mblt-tracker`. Finally, re-run band drop on the *quantized* model.
- **Evaluation.** Scene-level held-out splits (never pixel splits); mIoU, macro-F1, per-class IoU; cross-sensor relative drop; INT8 minus FP32 gap; latency at batch 1; energy per frame. Report separation radius vs receptive field wherever a spatial split is used.
- **Feasibility / risk.** Data are small (MWIR-4-Plastic is 13 scenes; overfitting risk, mitigated by the authors' fixed split and by SpectralWaste as a second dataset). SpectralWaste HSI labels are transferred from RGB masks, so label noise is a caveat to state. The MXQ compiler (`qbcompiler`) is gated behind Mobilint's package channel, so board access depends on the course; prototype with the ONNX Runtime CPU fallback and plan two dedicated NPU weeks.
- **Reuse.** Everything in `src/`: self- and cross-attention blocks, wavelength PE, the spectral stub with its band-mask API, and the tests. The spatial path and full SSFT assembly are still the open SSFT-2 ticket, so nothing built so far is wasted. Indian Pines / Pavia drop out or remain as a one-line sanity check.
- **Venues.** IROS 2027 (deadline Mar 1, 2027), CVPR 2027 workshops PBVS or Embedded Vision (~Mar 2027), ECV@CVPR 2027, ICIP 2027 (~Feb 2027), arXiv preprint in December. Out of reach: ICRA 2027 main track (Sep 15, 2026) and CVPR 2027 main track (Nov 16, 2026).

### Candidate B: Spectral-preserving perception for driving / terrain robots on HS3-Bench, distilled from HyperVision to an NPU-sized student

- **Problem.** Hyperspectral cameras are now on robots and test vehicles, yet published baselines are beaten by RGB backbones on 3-channel projections; the only backbones that exploit spectra are ViT-B/L/H foundation models too heavy for a robot.
- **Gap / novelty.** Distill the frozen HyperVision-B backbone (released Jul 2026, loaders for 31 ground-level datasets) into a ~1M-parameter wavelength-conditioned student; test across HyKo2 (15 bands) and HSI-Drive v2 (25 bands); deploy INT8.
- **Datasets.** HyKo2 (2.4 GB), HSI-Drive v2 (8.7 GB, password-protected download); avoid Hyperspectral City V2 (419 GB). Hyper-Drive (Northeastern; snapshot VNIR 24 bands + SWIR 9 bands, 500+ labeled cubes, ROS) is a third option.
- **Risk.** Distillation at 512×512 from a ViT-B teacher is heavy for Colab; HyperVision sets a very high accuracy bar; the driving datasets have long-tailed classes and awkward access. Reuse of `src/` is partial (spectral branch and PE).
- **Venues.** IROS 2027, ICRA 2027 workshops (Feb–May 2027 CFPs).

### Candidate C: Cross-sensor methane plume detection (EMIT / EnMAP / PRISMA) with a wavelength-conditioned compact model for on-board processing

- **Problem.** UNEP's Methane Alert and Response System already runs deep models operationally, but each sensor gets its own model and matched-filter products; new sensors (SBG, CHIME, Carbon-I) will multiply data tenfold and require on-board triage.
- **Gap / novelty.** Train on one sensor's wavelength grid and test on the other two using the public MARS-Hyperspectral dataset; compare to HyperspectralViTs (SegFormer / EfficientViT on Jetson); deploy on the NPU as an on-board proxy.
- **Risk.** Remote-sensing domain knowledge (radiative transfer, matched filters); data volume is larger (multi-GB); weaker robotics story. Strongest publication fit for IGARSS 2027 (abstracts ~mid-Jan 2027) and WHISPERS 2027.

### Candidate D (not recommended as the main project): Agentic / AI-scientist workflow for HSI experiments

The evidence is unfavorable: independent evaluations found 42% experiment failure rates, unreliable novelty checks, silent dataset subsampling, and post-hoc selection bias; a 2026 survey of 35 systems finds claims "harder to verify than the code is to run." A generic agent wrapped around HSI tools would likely draw the same "toy" verdict. **Use agents as tooling, not as the claim:** an LLM agent that audits every experiment for leakage (checks separation radius vs receptive field, flags pixel splits, verifies metric definitions) is cheap to build, defensible, and directly answers the professor's methodological worry. If the team wants an agentic *paper*, the tractable, publishable form is a verifier-guided execution loop evaluated on trajectory-level correctness, which the MBZUAI/TUM position paper explicitly leaves open.

### Salvage assessment of the current proposal

SSFT-Wave as written cannot be rescued by better wording alone. It becomes credible only if the team (1) replaces Indian Pines / Pavia with a real multi-sensor task, (2) replaces the zero-filled vanilla ViT with channel-agnostic baselines, (3) adds a measured axis no one has published, and (4) states the leakage protocol precisely. Candidate A does all four while keeping every line of code written so far.

## 4. Recommended proposal, in the required outline

**Title.** *One Model, Any Camera: Sensor-Agnostic Hyperspectral Material Segmentation for Robotic Recycling on an Edge NPU*

**Problem statement.** Automated recycling depends on identifying materials on fast conveyors. Hyperspectral cameras separate polymers that look identical in RGB, but each plant's camera has a different number of bands and a different wavelength grid, so today's models are trained per camera, break when a band fails, and are benchmarked on GPUs rather than the embedded accelerators that sorters actually use. We ask: can a single compact spectral–spatial transformer, conditioned on physical wavelengths rather than band indices, segment materials across VNIR, SWIR, and MWIR cameras, survive dropped bands, and run in INT8 on an 80-TOPS edge NPU without losing that robustness?

**Research gap and novelty.** Channel-agnostic HSI backbones exist (HyperFree, CARL, LESSViT, HyperVision) but are ViT-B or larger, trained for airborne or generic scenes, and never measured on an INT8 NPU. Industrial HSI datasets released in 2024–2026 (SpectralWaste, MWIR-4-Plastic, WoodVIT, HSI-AgriFoodAnomaly) ship CNN or heavy Swin baselines with no cross-sensor or edge-deployment evaluation. Our contributions: (1) the first cross-sensor (VNIR ↔ SWIR ↔ MWIR) evaluation of a ~0.2–1M-parameter wavelength-conditioned segmenter on co-registered black-plastic scenes; (2) the first measured INT8 NPU deployment of spectral attention, including latency, energy, and post-quantization band-drop robustness; (3) a documented hyperspectral calibration path for an RGB-only NPU toolchain; (4) a leakage-audited protocol that reports separation radius against receptive field.

**Dataset.** MWIR-4-Plastic (Zenodo 22142947): 13 co-registered RGB / VNIR / SWIR / MWIR scenes, five classes, fixed held-out cube per class. SpectralWaste (Hugging Face `ferpb/spectralwaste-segmentation`): 224-band SWIR, 852 labeled + 6,801 unlabeled frames from an operating plant, 514 / 167 / 171 split. Both fit on a laptop.

**Proposed approach.** Finish SSFT-Wave as planned and point it at segmentation: tied scalar embedding, sinusoidal wavelength PE (microns), one spectral MHSA over bands (built), one spatial conv block and cross-attention fusion (the open SSFT-2 ticket), per-pixel head. Baselines: index-PE variant, HyperFree-style wavelength-dictionary embedding, SpectralFormer and 1D-CNN from the MWIR-4-Plastic repo, and CMX-B0 / BCAF numbers on SpectralWaste. Deployment: ONNX → MXQ (ARIES target), custom multi-band calibration set, band mask baked as a constant-shape multiply, wavelength PE as constants.

**Evaluation plan.** Scene-level splits only. Metrics: mIoU, macro-F1, per-class IoU. Headline tables: (i) in-sensor vs cross-sensor relative drop per model; (ii) FP32 vs INT8 accuracy, latency at batch 1, energy per frame on the MLA100; (iii) band-drop robustness before and after quantization, 3 seeds. Two ablations: fused vs spectral-only vs spatial-only; index-PE vs wavelength-PE vs dictionary embedding.

**Timeline (12 weeks from Sep 10).**

| Weeks | Sprint | Deliverable |
|---|---|---|
| 1–2 | Pivot and data | MWIR-4-Plastic + SpectralWaste loaders with wavelength vectors; reproduce SpectralFormer / 1D-CNN baselines; freeze scene-level splits and cross-sensor protocol |
| 3–4 | Wave segmenter | SSFT-Wave as per-pixel segmenter; in-sensor results on both datasets; ONNX export verified on CPU runtime |
| 5–6 | Cross-sensor + baselines | VNIR→SWIR→MWIR zero-shot and few-shot; index-PE and dictionary-embedding baselines; leakage audit script |
| 7–8 | NPU | MXQ compile with multi-band calibration; INT8 accuracy, latency, energy; post-quantization band drop |
| 9–10 | Ablations + writeup | Two ablations, limitations, class presentation |
| 11–12 | Polish + submit | arXiv preprint; IGARSS abstract (Jan), IROS 2027 (Mar 1) or CVPR-W PBVS / Embedded Vision (Mar) |

**Success criterion.** A ≤1M-parameter wavelength-conditioned segmenter that (a) loses less accuracy than index-PE and fixed-C baselines when moved to an unseen sensor, and (b) keeps that advantage after INT8 quantization on the MLA100 at real-time latency. Will not claim: state of the art on SpectralWaste against 30M-parameter Swin fusion; foundation-model pretraining; Indian Pines results.

## 5. Implications for this repo

- **Keep:** `src/attention.py`, `src/pe.py`, `src/spectral_stub.py`, tests, WIP-1 board process, the "no random-pixel split" rule.
- **Retire:** Indian Pines / Pavia loader plan (SSFT-4), the vanilla-ViT-only baseline (SSFT-6), the 516k-parameter chase.
- **Rewrite:** PLAN.md locked defaults (patch → per-pixel / small-tile segmentation; datasets; baselines; NPU sprint), tickets SSFT-4 through SSFT-8, and the class-intro deck's problem, gap, and dataset slides.
- **Add:** a leakage-audit script (separation radius vs receptive field) that runs in CI, and a `deploy/` sprint for ONNX → MXQ.

## 6. Sources

Primary papers and datasets

- Ten Architectures, One Error: Shared Failure Modes in Hyperspectral Classification under Spatially Disjoint Evaluation (Sep 2026). https://arxiv.org/abs/2609.01786
- LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift (May 2026). https://arxiv.org/abs/2605.18541 · code https://github.com/uiuctml/LESSViT
- HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone (May 2026). https://arxiv.org/abs/2605.17286
- SSFT: A Lightweight Spectral-Spatial Fusion Transformer for Generic Hyperspectral Classification (CVPR 2026 Workshops, PBVS). https://arxiv.org/abs/2604.15828
- Cross-Domain Transfer of Hyperspectral Foundation Models (ICPR 2026). https://arxiv.org/abs/2604.26478
- Benchmarking Foundation Models for Hyperspectral Image Classification: Cereal Crop Type Mapping (arXiv Oct 2025; listed as under review for WHISPERS). https://arxiv.org/abs/2510.11576
- A General Purpose Spectral Foundational Model for Both Proximal and Remote Sensing Spectral Imaging (Mar 2025). https://arxiv.org/abs/2503.01628
- CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis (Apr 2025). https://arxiv.org/abs/2504.19223
- HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery (CVPR 2025). https://arxiv.org/abs/2503.21841
- SpectralWaste Dataset: Multimodal Data for Waste Sorting Automation (IROS 2024, pp. 5852–5858). https://arxiv.org/abs/2403.18033 · data https://huggingface.co/datasets/ferpb/spectralwaste-segmentation · HSI-Benchmark (Tübingen) https://github.com/cogsys-tuebingen/hsi_benchmark
- Bidirectional Cross-Attention Fusion of High-Res RGB and Low-Res HSI for Multimodal Automated Waste Sorting (Mar 2026). https://arxiv.org/abs/2603.13941
- MWIR-4-Plastic: hyperspectral dataset of shredded black end-of-life-vehicle plastics (Aug 2026). https://arxiv.org/abs/2608.28874 · code https://github.com/Elias-Arbash/MWIR-4-Plastic · data https://doi.org/10.5281/zenodo.22142947
- WoodVIT: multimodal and hyperspectral bulky-waste dataset (Scientific Data 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13035887/
- HS3-Bench: Hyperspectral Semantic Segmentation in Driving Scenarios (IROS 2024). https://arxiv.org/abs/2409.11205 · code https://github.com/nickstheisen/hyperseg
- Hyper-Drive: VNIR-SWIR hyperspectral datasets for robots in unstructured environments (2023). https://arxiv.org/abs/2308.08058
- HyperspectralViTs: General Hyperspectral Models for On-board Remote Sensing (2025). https://arxiv.org/abs/2410.17248
- Operational machine learning for remote spectroscopic detection of CH4 point sources (UNEP MARS, Nov 2025). https://arxiv.org/abs/2511.07719 · data https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral
- TinyCapsViT: Ultra-Lightweight HSI Classification for UAV Edge Deployment (Remote Sensing, Aug 2026). https://doi.org/10.3390/rs18162661
- AutoNeural: Co-Designing Vision-Language Models for NPU Inference (Dec 2025). https://arxiv.org/abs/2512.02924

NPU toolchain

- Mobilint model zoo on Hugging Face (ViT / DeiT / Swin / Qwen3-VL compiled to MXQ). https://huggingface.co/mobilint/ViT_Tiny_Patch16_224
- Mobilint vision toolkit, calibration and MXQ compile guide. https://github.com/mobilint/mblt-vision-python
- Mobilint MLA100 MXM, 80 TOPS ARIES module (Apr 2025). https://embeddedvisionsummit.com/posts/2025-04-mobilint-introduces-mla100-mxm-an-80-tops-npu-module-for-high-efficiency-embedded-ai-pcs/

Agentic workflows (cautionary)

- The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems (2025). https://arxiv.org/abs/2509.08713
- Evaluating Sakana's AI Scientist (SIGIR Forum 2025). https://dl.acm.org/doi/10.1145/3769733.3769747
- Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap (Jun 2026). https://arxiv.org/abs/2608.05179
- Agentic AI for Remote Sensing: Technical Challenges and Research Directions (Apr 2026). https://arxiv.org/abs/2604.24919
- HM-Bench: Multimodal LLMs in Hyperspectral Remote Sensing (Apr 2026). https://arxiv.org/abs/2604.08884

Venues and reviewer expectations

- IEEE GRSS Checklist for Authors. https://www.grss-ieee.org/publications/checklist-for-authors/
- IROS 2027 deadline Mar 1, 2027. https://mldeadlines.com/conference/iros-2027/
- IGARSS 2027, Reykjavik. https://ieee.is/event/igarss-2027/
- CVPR PBVS workshop. https://pbvs-workshop.github.io/ · ECV workshop https://ecv-workshop.github.io/
- ICIP 2027 call for papers. https://2027.ieeeicip.org/call-for-papers/

## Appendix A. Verification status of the claims this memo rests on

Twenty-five extracted claims went to three independent adversarial verifiers each (a claim dies on two refutations). Every verifier worked from primary artifacts (author repos, Hugging Face mirrors, dataset cards) because arxiv.org is blocked from this environment.

**Confirmed (used as stated).** Random-pixel leakage on single-scene HSI benchmarks, the 0.147 Macro-F1 drop and rank shifts, and the receptive-field rule (all 3–0, from the Sep 2026 preprint's abstract). LESSViT's problem statement and solution (2–1; the dissent objects only to calling the reimplementation "no new idea"). Cross-domain foundation-model transfer gains on HS3-Bench (3–0). HS3-Bench code, license, loaders, and the pseudo-RGB-beats-full-cube baseline table (3–0). BCAF's SpectralWaste numbers and model size (3–0). MWIR-4-Plastic's existence, sensors, classes and code release (2 of 3 votes in; both confirm).

**Refuted or rewritten (and what changed in this memo).**
- "SpectralWaste is CC BY 4.0" and "CMX-B0 reached 64.5% mIoU at 58.2 img/s": the LICENSE file is CC BY-NC 4.0, and the numbers were a column shift in the paper's Table IV. Corrected to 58.2% mIoU at 54.7 img/s.
- "SSFT does not use Indian Pines or Pavia as headline evidence": false. HSI-Benchmark's remote-sensing domain is exactly those scenes plus Salinas, under random-pixel splits. Section 1 now says so.
- "The team has already reimplemented the dual-path fusion": only the spectral stub and attention primitives exist; the spatial path is the open SSFT-2 ticket. Reuse statements were corrected.
- "LESSViT's four-setting protocol is the 2026 evidence standard": the settings are real (C120 VNIR+, C120 SWIR+, C82 unseen, C202 expansion) but calling them a standard was an overreach. The memo cites the settings and reported drops without that framing.
- Crop-mapping foundation-model numbers (HyperSIGMA 34.5%, DOFA 62.6%, SpectralEarth 93.5%, from-scratch 91%): the numbers are real but come from a frozen-backbone probe on one binary region pair, under review, and do not support "architecture beats pretraining". Dropped from the body.
- "Spectral foundation models fall into two limited classes": hedged in the source and stale by 2026. Dropped.

**Still pending at time of writing.** Three HyperVision claims (channel range, ground-level datasets, efficiency as open problem), the third MWIR-4-Plastic vote, and two claims about MWIR-4-Plastic's stated gap and its transformer baselines; these were interrupted by a usage limit and are being re-run. They are used in the body with the qualifier "reported" where they matter.
