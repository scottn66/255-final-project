# Review of last week's memos against the professor's architecture focus, the syllabus penalty, and the ZeroWaste pivot

**Date:** 2026-09-17 (course week 5 of 16) · **Team:** Group 10 (Scott Nelson, Surya Reddy, Tim Wada) · **Supersedes the "current claim" in `docs/research/README.md` on `main`** · **Method:** judge-panel workflow (5 research angles, 4 independent proposals, 3 judges with different lenses, 3-vote adversarial verification of 7 load-bearing claims, completeness critic), plus a direct self-audit of the two 2026-09-10 memos. All 34 agents completed; all 7 verified claims held 3–0. Appendix A has the scores, Appendix B the verdicts and corrections.

> **Verdict.** Last week's memos were right about the problem and wrong about the center of gravity. They put the NPU where the architecture should be, and the team's condensed "current claim" on `main` inherited that. The professor's three signals (architecture focus, did not see the novelty, syllabus penalty) are a predictable response to that framing, not a rejection of the direction. **The ZeroWaste RGB pivot should not be abandoned; it should become Milestone 1 of a project whose contribution is a custom fusion architecture on SpectralWaste**, the same recycling domain with co-registered RGB and hyperspectral frames. The precise, verified gap: on SpectralWaste, every fusion model that beats its own RGB-only and HSI-only parts is 11M to 101M parameters, and every attempt under 5M scores *below* its hyperspectral part alone. A sub-5M fusion that beats both parts, one of which is an ImageNet-pretrained RGB segmenter, is an architecture claim a CNN-trained grader can read in one table. The NPU becomes one measured column, which is exactly what Lab 1 taught and what the sponsored hardware is for.

---

## 1. Review of the 2026-09-10 memos

**They optimized for a different objective than the grader.** Both were built to answer "publishable and buildable in 12 weeks." Neither was built against a rubric that grades architecture customization and penalizes out-of-syllabus material. Word counts make the drift visible:

| Term family | Direction memo | Feasibility memo | Team's condensed claim on `main` |
|---|---|---|---|
| architecture | 5 | 0 | 1 |
| ablation, fusion, attention | 23 | 11 | 4 |
| INT8, quantization | 20 | 17 | 11 |
| NPU, ARIES, compile | 42 | 54 | 19 |

The three "novelty claims" in the direction memo were all measurement claims: first cross-sensor evaluation, first INT8 measurement of spectral attention, a documented calibration path. Not one was a design claim. A grader who wants architecture read three engineering contributions and zero architectural ones, then said "focus on the architecture." That is the memos' fault, not his.

**What still stands.** Indian Pines and Pavia are a dead end, and SSFT's own benchmark contains them under random-pixel splits. Wavelength positional encoding plus tied embedding is not novel on its own, which is also *why* the professor did not see novelty: presented as a component, there is not much to see. Recycling is a strong, physically grounded motivation. NPU work is syllabus-safe because Lab 1 covered ONNX, MXQ and the Mobilint board, and it is now also institutionally encouraged: Mobilint signed an MOU with SJSU's Applied Data Science department in 2025 covering accelerator-based research, curriculum, and capstone support. Zero-shot transfer across disjoint wavelength ranges collapses and must stay out of any headline.

**What the memos got wrong for this grader.**
- They made the NPU the spine. The architecture has to be the noun and the hardware an adjective.
- They under-specified the network. "Tied embedding, wavelength PE, one spectral MHSA, one spatial conv block, cross-attention fusion" is a parts list, not a design argument. There was no design question, and no ablation table built to make each component earn its place.
- They never asked what the syllabus covers. Given the penalty, that was a real omission.

**Three factual corrections to those memos.** SpectralWaste was relicensed to CC BY 4.0 on 2026-09-11 (LICENSE file and card metadata both changed); both memos and `main`'s README say CC BY-NC. The claim that Mobilint's BERT tutorial needed `cpu_offload` is wrong; the current tutorial compiles the LayerNorm/softmax/GELU encoder onto the NPU and hand-splits at the encoder boundary. The "wavelength vector as a runtime input tensor" idea has a vendor precedent (Qwen3-VL's dynamic mode feeds host-computed position embeddings and RoPE as input tensors), but it is a model-specific patcher, not a documented generic feature.

**On the ZeroWaste pivot.** It is a sane reaction to a deployment-heavy plan that read as out of reach, and it is fully in syllabus. But research on ZeroWaste itself is not encouraging for an architecture claim: every competitive model there (FANet, COSNet, EWSegNet, LWCHNet) is a 20–30M ImageNet-pretrained encoder with a bolt-on boundary or frequency module feeding UperNet, the leaderboard has moved 4.5 mIoU in four years, and a DINOv2 backbone reportedly beats all of them. A course team cannot pretrain an encoder, so "our custom architecture on ZeroWaste" reduces to a fifth bolt-on module, which is precisely the critique already received. What ZeroWaste *does* buy is 5× more labeled frames, clean polygon labels, a 7 GB download, and a domain that is the RGB-only ancestor of SpectralWaste. Used as Milestone 1, it is an asset. Used as the whole project, it forfeits the architecture story.

## 2. What the professor is asking for, mapped to the syllabus

The public record: the catalog describes DATA 255 as CNNs for image recognition, RNNs for sequence generation, and GANs for image generation; the MSADI program has retitled it "Deep Learning and Computer Vision"; Lab 1 covered ImageNet-20, ONNX, MXQ and the Mobilint NPU. No public greensheet or rubric was found, so the exact wording of the scope deduction is unknown and should be quoted from Canvas before the proposal is finalized.

| Component | Syllabus status | Risk | How to present it |
|---|---|---|---|
| CNN encoder-decoder for segmentation | In (CNNs for image recognition; segmentation is pixel-level recognition) | Low | The spine of the project. MiniNet-v2 and SegFormer-B0 as like-for-like baselines. |
| Transfer learning from ImageNet and from ZeroWaste | In (Lab 1 used pretrained classifiers; standard SJSU DL content) | Low | Pretrained compact RGB encoder is the default RGB branch. ZeroWaste weights as conveyor-domain initialization. |
| ONNX, INT8, ARIES/MLA100 measurement | In (Lab 1) and encouraged (MOU, sponsored hardware) | Low | One row per table in the course's own vocabulary: params, FLOPs, INT8 vs FP32 mIoU, batch-1 latency, energy. |
| One spectral self-attention block, one cross-attention fusion block, from scratch | Allowed by exception | Medium, controllable | Get the exception in writing and confirm it covers cross-attention. Ship in-syllabus fallbacks in the same ablation grid (see below). |
| Grouped tied band embedding + wavelength code as input | Gray (pure CNN vocabulary: a 1×1 conv with weight sharing across bands; a CoordConv-style coordinate input) | Medium, a communication risk | Describe in those terms. Report the expected in-distribution tie with index encoding honestly. |
| Bidirectional GRU over the spectrum | In (RNNs for sequences; a spectrum is a sequence over wavelength) | Low | The in-syllabus alternative to the spectral attention block, as an ablation row. |
| 224-channel hyperspectral input | Gray (a dataset choice, "various application domains") | Low–medium | "An image with 224 channels." Keep the RGB track on the same frames; report PCA-3. |
| GANs | In, but no research role | Negative if forced | Omit and say so. |
| Foundation backbones, 4-bit, cross-sensor zero-shot, UDA | Out | Avoid | Cite as ceilings or prior art only. |

Three in-syllabus fallbacks cover all three attention components if the exception turns out narrow: gating-only fusion (channel and spatial rectification without attention, CMX-style) replaces the cross-attention block; a bidirectional GRU replaces the spectral self-attention; masked mean over bands (already in `src/spectral_stub.py`) replaces the learned-query pooling.

## 3. Why he did not see the novelty, and how to show it

He did not see it because it was presented as components, and the components are not novel. The novelty is in the *asymmetry of the design* and it has to be shown as an absence in a figure and as rows in a table. The four-sentence version, in course vocabulary:

1. **Input contract.** Every network in the course starts with `Conv2d(C→D)`, which freezes the input to exactly C channels in one order and mixes them with one kernel. Replacing it with one shared `Linear(1→D)` per band, a 1×1 convolution with in-channels 1 applied to each band, is the same weight-sharing idea a conv kernel uses across positions, applied across bands. The input becomes a set of (value, wavelength) pairs of any length.
2. **Continuous coordinate.** A learned per-band table says "you are band 37," which only means something on the camera it was trained on. A sinusoidal code of the physical wavelength says "you are the 1,412 nm measurement." That is CoordConv's idea with the coordinate in nanometers.
3. **Inductive bias.** A 3D CNN mixes a fixed neighborhood of pixels and adjacent bands with content-independent weights. Attention along the spectral axis lets every band weight every other band with weights computed from the spectrum itself, while an ordinary 2D CNN handles spatial texture. That is the split R(2+1)D made when it factorized 3D video kernels into 2D space plus 1D time. It encodes a different assumption: material identity is a global function of the whole curve; object shape is a local function of neighbors.
4. **The fusion decides.** With the RGB feature as query and the spectral tokens as keys and values, every pixel computes a data-dependent softmax over which spectral evidence to read. Concatenation plus a 1×1 conv applies the same fixed weights at every pixel regardless of what the spectrum says.

The presentation pattern, taken from EdgeNeXt, MobileViT, ConvNeXt, CMX and TUNI: one "X replaces Y inside Z" sentence; three contribution bullets (mechanism, architecture, measured result on hardware); Figure 1 drawing three fusion topologies on the same RGB+HSI pair (CMX's two symmetric backbones at 11.5M, BCAF's two Swins at 101M, ours: one CNN and a spectrum reader with no spatial layers and one arrow); a ConvNeXt-style roadmap chart with one design change per bar at matched parameters and three seeds; one ablation table with one removal per row and params and latency columns; results in SpectralWaste's Table IV column order so rows line up with published baselines; and an explicit "closest prior work and our delta" paragraph.

## 4. The reconciled proposal

Four proposals were generated from different angles and scored by three judges (professor lens, workshop-reviewer lens, project-manager lens). The graduated ZeroWaste-to-SpectralWaste fusion proposal won on total; the professor-lens judge alone preferred the INT8-designed architecture by one point. What follows is the winner with the best ideas of the other three grafted in and the critic's corrections applied.

**Title.** *One CNN, One Spectrum Reader, One Lookup: A Sub-5M-Parameter RGB+Hyperspectral Fusion Segmenter for Conveyor Waste Sorting, Measured on the Course NPU.*

**One-sentence claim.** A compact ImageNet-pretrained RGB CNN that, at one mid stage, looks up the spectrum beneath each feature location through a single cross-attention over wavelength-tagged band tokens is the first RGB+HSI segmenter under 5M parameters that beats both of its own unimodal parts on SpectralWaste, and the gain survives INT8 compilation on the Mobilint ARIES.

**Problem statement.** Recycling plants sort material on fast conveyors with an RGB camera, which sees shape and texture but confuses materials that look alike, and a hyperspectral line-scan camera, which separates materials by their reflectance curve but at lower resolution. Sorters run on edge accelerators. On the only public dataset with co-registered frames from a real sorting plant, every fusion model that beats its own parts is 11M to 101M parameters, and every attempt under 5M (channel concatenation) scores below its hyperspectral part alone, so no fusion design that ships on an accelerator has been shown to help. The design question is how a full-resolution 3-channel CNN and a pooled 224-channel per-pixel spectrum reader should be fused, at which stage, in which direction, with which spectral tokenization, so that the fused model beats both parts under a hard 5M budget.

**Architecture, in three blocks.**
1. *RGB path.* An ordinary CNN encoder-decoder. Default: an ImageNet-pretrained compact encoder (MiT-B0 or MobileNetV3-Large class, 3–4M) with a light decoder. This is the network fine-tuned on ZeroWaste in Milestone 1; it carries all spatial reasoning. A from-scratch ≤1M variant is an ablation row.
2. *Spectrum reader.* No spatial layers at all. The 224-band cube is max-pooled 4× to a 64×64 grid; at each location the spectrum is cut into 28 groups of 8 adjacent bands; one shared `Linear(8→64)` embeds every group; a sinusoidal code of each group's physical center wavelength is added as a host-computed input tensor; one from-scratch self-attention block mixes the 28 tokens; four learned material queries pool them into one 64-d descriptor per location. About 0.12M parameters. This is `src/spectral_stub.py` plus pooling, grouping, and query pooling; `src/attention.py` and `src/pe.py` are used unchanged.
3. *The lookup.* One cross-attention block at the CNN's stride-4 stage: the RGB feature at each location is the query, the spectral tokens under it are keys and values, followed by a sigmoid-gated residual add into the decoder. About 0.07M parameters.

Total under 5M with a pretrained encoder; under 1.5M with the from-scratch encoder. The customization the professor grades is blocks 2 and 3, written by the team, and their placement.

**Novelty versus prior art, stated so "done before" is answered up front.** CMX uses two symmetric image backbones fused at all four stages in both directions (11.5M on SpectralWaste). BCAF uses two Swin backbones with bidirectional local cross-attention at four stages (101M, code unreleased). SSFT fuses spatial queries with spectral keys inside one hyperspectral cube for patch classification. CARL pools wavelength-encoded band tokens with learned queries in a heavy single-modality ViT. FusionSort and HLRFF-Net collapse the spectrum to PCA-3 before fusing, and FusionSort's fused model scores below its RGB-only model. LESSViT, ChannelViT and DOFA supply the tokenizer and are cited as its source. HREM-Net (Jul 2026) and a sub-1M multimodal fusion transformer on Electrolyzers-HSI are the nearest compact RGB+HSI designs and must be cited; both are patch-wise or non-waste. The precise gap, worded defensibly: *to our knowledge, no dense RGB+HSI segmentation network under 5M parameters on industrial waste-sorting data beats both of its unimodal parts, and none reports INT8 or NPU measurements.* HLRFF-Net's paywalled benchmark table is the main place a counterexample could hide and is named as unchecked.

**Datasets.** ZeroWaste-f for Milestone 1 (4,503 labeled 1080p frames, 3,002/572/929, four material classes, CC BY-NC 4.0, ~7 GB; the final Zenodo archive may be password-protected, request access in week 5; audit the final split for video overlap from `labels.json`, since the committed 2021 split lists are a different cut). SpectralWaste segmentation release for the headline (852 labeled frames, 514/167/171, RGB and HSI at 256×256 co-registered, CC BY 4.0 since 2026-09-11; pin the Hugging Face revision hash, the repo changed three times in nine days; split by `sequence_id`; wavelengths are the nominal FX17 grid, stated as an approximation; the HSI masks are transferred from RGB with a measured 79.6 mIoU ceiling, so train and evaluate fusion models on `label_rgb` as CMX-B0 did).

**Baselines, and a provenance rule.** The published SpectralWaste Table IV rows (MiniNet-v2 RGB 44.5 / HYPER 52.8 / early 49.1 at 0.59M; SegFormer-B0 RGB 48.4 / HYPER 54.3 / early 53.6 at 4.07M; CMX-B0 RGB+HSI 58.2 at 11.5M) are from-scratch, single-seed, weighted-CE numbers, and the repo's issue #1 reports a reproduction that landed several points lower. So: reprint the published rows for alignment only, and make every comparison against baselines retrained under one recipe with three seeds. Required additional row: an ImageNet-pretrained compact RGB-only segmenter, because BCAF's pretrained MiT-B0 RGB-only reaches 62.1 mIoU at 256×256 on this metric, above CMX-B0. This is the honest ceiling and the reason the RGB branch is pretrained by default.

**Pre-registered success criterion.** Fused mean mIoU exceeds max(RGB-only, HSI-only) by more than 2 points with non-overlapping three-seed ranges, against parts trained under the same recipe, plus a parameter-matched unimodal control (the RGB model widened to the fused model's size) so the win is not extra capacity. Per-class IoU on film, bag, filament and tape is always reported, because that is where HSI is expected to help and where the RGB ceiling is weakest.

**Week-9 gate: the oracle analysis, defined.** A pixel counts as recoverable if either unimodal model predicts it correctly. Oracle mIoU minus max(unimodal mIoU) is the complementarity ceiling. If it is under 3 points, no fusion can meet the criterion; stop tuning fusion, report the grid as a controlled negative result with the oracle figure, and promote the INT8-architecture rows (below) to the second contribution.

**Ablations that make the architecture visible** (matched parameters, same recipe, three seeds on headline cells):
- A. *Parts versus whole* (headline): pretrained RGB-only, from-scratch RGB-only, spectrum reader alone, PCA-3 through the RGB CNN, published early-concat rows, fused.
- B. *Fusion mechanism*: early concat + 1×1, add, gating-only (no attention, in-syllabus), cross-attention RGB→spectrum (ours), reversed direction, bidirectional.
- C. *Placement* with one block: input, stride-2, stride-4 (ours), decoder.
- D. *Spectral tokenization*: 224 per-band tokens, 28 groups of 8 (ours), 8 groups of 28, PCA-3.
- E. *Spectral operator* at equal parameters: self-attention (ours), Conv1d over bands, bidirectional GRU (in-syllabus), mean-pool.
- F. *Break matrix*, one figure: rows are the winning components, columns are test-time perturbations (spatial lesion, spectral lesion, uniform fusion weights, wavelength shuffle, 25% band drop, HSI shift of 1–4 pixels per BCAF's Table 7 protocol, INT8). If the uniform-attention lesion barely moves mIoU, the "fusion decides" claim is dropped.
- G. *Architecture versus loss* control grid: {CE, weighted CE, Dice+CE, Focal} × {concat fusion, ours}, because class imbalance (filament is 32k train pixels against 27.5M background) can drive the headline instead of the architecture.
- H. *Training*: joint versus two-phase (unimodal checkpoints then fusion fine-tune; BCAF reports 76.4 vs 55.9 at 101M, unmeasured at compact scale); RGB branch frozen during fusion fine-tune as the countermeasure to the gate collapsing to zero.
- I. *Precision × fusion type on ARIES*: FP32 vs W8A8 mIoU, per-class delta, batch-1 latency p50/p99, energy per frame, for RGB-only, concat, gating-only, ours.

**The NPU chapter, with the INT8-architecture rows grafted from the runner-up.** Protocol frozen and stated once: static batch-1 ONNX; qbcompiler W8A8 with the vendor's BERT-tutorial calibration configuration (MaxPercentile 0.999, top-k 0.01); 32–128 calibration crops as generic `.npy` tensors matching the MBLT input shape, channel-last for image-like inputs; no rotations, no 16-bit overrides except where a variant otherwise fails, with the count of overrides a reported column. Deployment knobs (percentile versus KL calibration, 16-bit boundary layers) live in a separate two-row table on the best model only, so the architecture-versus-deployment separation is visible. Then two architecture rows the professor-lens judge valued most: the vanilla spectral block (Pre-LN, softmax over tokens, GELU) versus an integer-friendly twin (conv-folded BatchNorm or DyT, clipped or gated softmax, ReLU, nearest-neighbor upsampling), each compiled under the same protocol, with a per-layer activation-range figure explaining the gap. Mobilint's own zoo loses 0.9–1.4 points on small and hybrid classifiers and the only integer-only ViT segmenter lost about 5 points, so "under 2 mIoU loss" is a target the architecture must earn, and a variant that fails is a finding. Energy is measurable: `mblt-tracker` documents NPU power, utilization, memory and temperature with average, peak and p99, with the caveat that extra rails share one firmware register and refresh at 1 s. Every INT8 row is measured on the card; PyTorch fake-quant is the fallback only, with the fake-quant-versus-card discrepancy reported if used, because INT8 predictions are not portable across accelerators.

**Timeline, weeks 5–16, with owners.** (Owners are a suggestion; the team should confirm. Scott is listed for governance because he owns the proposal.)

| Week | Build | Evidence | Owner |
|---|---|---|---|
| 5 | Proposal in the seven-section outline. Email: transformer allowance in writing, covering cross-attention; confirm "architecture ablation on one dataset measured on the NPU" is what he means. Downloads: ZeroWaste-f, SpectralWaste labeled (pin revision). **NPU smoke test** on the lab machine: a stock 3-channel segmenter via the Lab 1 path, then `SpectralStub` at C=32 as a multi-input graph; acceptance check is opening the MBLT in Mobilint Netron and listing CPU-offloaded layers. Name who holds the Download Center account and book the slots. Protocol document frozen. | Smoke-test report | Scott (proposal, NPU), Surya (data), Tim (protocol) |
| 6–7 | **Milestone 1.** Fine-tune a pretrained compact encoder plus light decoder on ZeroWaste-f at 512 crops; video-disjoint audit; per-class IoU; FP32 vs INT8 on ARIES (first edge number on ZeroWaste). In parallel: SpectralWaste loader, sequence splits, pre-decoded fp16 cache on NVMe, test-only reproduction of the 12 authors' checkpoints (budget one day; needs torch 2.2.2 and the Zenodo layout; if the SharePoint link is dead, baselines are retrained only). | ZeroWaste table + INT8 row | Tim (M1), Surya (SpectralWaste). **Hard stop end of week 7, called by Scott.** |
| 8 | Unimodal table on SpectralWaste under one recipe, 3 seeds: pretrained and from-scratch RGB, ZeroWaste-init RGB, spectrum reader alone, PCA-3, retrained MiniNet-v2 rows. | Table IV-format unimodal rows | Surya, Tim |
| 9–10 | Fusion model; joint vs two-phase; **oracle analysis and gate**; core result. Architecture frozen end of week 10 (Scott). | Headline table | All |
| 11 | Ablation grid B–E and G; 3 seeds on headline cells, 1 elsewhere, with "no measurable difference" pre-declared as an acceptable cell outcome. | Ablation table, roadmap chart | Surya, Tim |
| 12 | NPU chapter: compile fused model and fusion variants; INT8 twin rows; latency, energy, override counts. | Table I, activation-range figure | Scott |
| 13 | Break matrix F; attention-over-wavelength histograms per class; optional native-resolution RGB branch from the raw release (cut-line). | Figure | All |
| 14 | Write-up in the required outline; Figure 1; closest-prior-art paragraph; limitations; will-not list. | Draft | Scott (lead) |
| 15–16 | Presentation; code and MXQ artifacts released; buffer. | Deliverable | All |

**Compute plan.** About 26 configurations, roughly 50 runs of 200 epochs over 514 cubes at 256×256. Decode the 29 MB uint16 TIFFs once into a memory-mapped fp16 array (about 15 GB) on local NVMe; random band-group subsampling per step cuts the spectral path's cost further. A MiniNet-class run is on the order of one to two GPU-hours on a T4-class card; the spectrum reader adds attention over 28 tokens at 4,096 locations, which is cheap. Measure per-epoch time in week 6 before committing the grid.

**Top risks, with cut-lines.**

| Risk | Mitigation | Cut-line |
|---|---|---|
| Fusion does not beat both parts (the published compact regime) | Oracle gate in week 9; two-phase training; freeze the RGB branch during fusion fine-tune; HSI-primary variant first | Report the grid as a controlled negative result with the oracle figure; INT8 twin rows become the second contribution |
| Pretrained RGB ceiling (~62 mIoU) makes the headline moot | Pretrained encoder is the default RGB part; per-class argument on film, bag, filament, tape | Drop "beats CMX-B0" language; keep "beats its own pretrained part" as the only claim |
| Seed noise swallows the grid on 171 test frames | Three seeds on headline cells; pre-declare "no difference" cells | Report only cells with non-overlapping ranges |
| Compiler rejects or offloads the multi-input graph | Week-5 smoke test; 4-D tensors and 1×1 convs; PE as input tensor; vendor channel via the professor | Fake-quant with discrepancy reported; card numbers kept for the RGB-only model |
| Milestone 1 absorbs the team past week 7 | Hard stop with a named owner and a fixed deliverable | Ship M1 as one table plus one INT8 row |
| Transformer exception narrower than assumed | Ask in writing in week 5; gating-only, GRU and masked-mean fallbacks already in the grid | Promote the in-syllabus variants to the headline |
| Recipe dominance breaks alignment with published rows | Same-recipe retraining; published rows reprinted for alignment only, with issue #1 cited | Drop published rows from the headline table |
| Data access latency (ZeroWaste archive password, SharePoint checkpoints) | Request in week 5; retrain-only fallback | M1 on the VisDA 7 GB copy; baselines retrained |

**Will not.** Chase state of the art on ZeroWaste or SpectralWaste RGB against 20–30M encoders. Do cross-sensor or disjoint-wavelength transfer. Use foundation backbones as trained components. Go below 8 bits. Add a GAN. Do unsupervised domain adaptation on ZeroWaste-v2.

**Minimum viable result.** Milestone 1: a compact pretrained-encoder segmenter on ZeroWaste-f with a video-disjoint audit, per-class IoU, and an FP32-vs-INT8 ARIES row. Headline: on SpectralWaste, one recipe, three seeds, a sub-5M fused model whose mIoU exceeds both its own parts by the pre-registered margin, in Table IV columns, with one matched-parameter fusion-mechanism ablation (concat, gating-only, cross-attention) and one INT8 ARIES row. Beating CMX-B0 is not the minimum.

## 5. What to tell the professor

Put this in the seven-section outline and in the reply to the Group 10 email thread:

> Last week we framed the project around one question: does wavelength-as-input survive INT8 on the ARIES card. You told us to focus on the architecture, that you did not see the novelty, and that out-of-syllabus material costs points. We checked our framing against those and changed it. The wavelength encoding and the per-band shared embedding are standard components in at least seven published models, so they are not our contribution; we keep them as the tokenizer and cite them. Mobilint's own model zoo compiles ViT, DeiT and Swin within about a point of GPU accuracy, so "does attention survive INT8" is a measurement whose answer is probably yes; it belongs in the deployment chapter, not the title. What is actually open, and is an architecture question, is this: on SpectralWaste, the only public dataset with co-registered RGB and 224-band hyperspectral frames from a real sorting plant, every fusion model that beats its own RGB-only and HSI-only parts is 11 to 101 million parameters, and every attempt under 5 million scores below its hyperspectral part alone. The center of the project is now a custom fusion architecture under a 5M budget: an ordinary pretrained CNN for the RGB image, a per-pixel spectrum reader with no spatial layers for the hyperspectral cube, and exactly one cross-attention lookup from the CNN into the spectrum. The evidence is an ablation table that removes or swaps one block per row at matched parameters, plus one INT8 row on the course NPU using the Lab 1 path. The ZeroWaste RGB work becomes Milestone 1: that CNN is the RGB branch, and its compile is the week-5 smoke test. Nothing from the spike is thrown away. We need three things in writing: that the transformer allowance covers one spectral self-attention block and one cross-attention block written from scratch (if not, gated fusion and a GRU over the spectrum become the headline, which are CNN and RNN material); that an architecture ablation on one industrial dataset measured on the NPU is what you mean by focusing on the architecture; and week-5 access to the ARIES compile machine. Two honesty notes: the published SpectralWaste baselines are from-scratch single-seed numbers with an open reproduction issue, so every comparison will be against baselines we retrain under one recipe; and an ImageNet-pretrained RGB-only model at the same compute reaches about 62 mIoU, so we include it as the ceiling and state per class where the hyperspectral camera helps rather than claim to beat it.

## 6. Repo actions this week

- `docs/research/README.md` on `main` still carries the "one binary serves many cameras" claim and says CC BY-NC. Replace its claim section with section 4 above; fix the license; pin the dataset revision.
- `PLAN.md` "Hard out of scope: Lab 1 / NPU", `AGENTS.md`, `BOARD.md` and `bulletins/1.40-risks.md` still forbid the NPU work the proposal relies on. Retire those lines.
- Add a one-line erratum to both 2026-09-10 memos (license; `cpu_offload`).
- Tickets: `M1-1` ZeroWaste fine-tune and audit; `M1-2` ZeroWaste INT8 row; `NPU-0` smoke test with Netron acceptance check; `DATA-1` SpectralWaste loader, cache, revision pin; `BASE-1` checkpoint reproduction with the one-day budget; `ARCH-1` spectrum reader (pooling, grouping, query pooling on the existing stub); `ARCH-2` lookup block; `PROTO-1` protocol document with the pre-registered criterion and oracle definition.
- Rebuild `docs/class_intro.js` from the new claim before the next class briefing.

## Appendix A. Judge scores

Three judges, six criteria each scored 1–10, totals out of 180 after merging the split title rows.

| Proposal | Professor fit | Syllabus safety | Novelty | Feasibility | Reuse | Legibility | Total |
|---|---|---|---|---|---|---|---|
| Graduated ZeroWaste → SpectralWaste fusion (winner) | 24 | 20 | 23 | 17 | 26 | 24 | 134 |
| ZeroWaste-only class-routed decoder (safe-plus) | 15 | 29 | 11 | 27 | 18 | 27 | 127 |
| INT8-designed spectral-spatial twins | 22 | 22 | 21 | 15 | 22 | 22 | 124 |
| Controlled design-map survey | 23 | 19 | 18 | 15 | 19 | 21 | 115 |

The professor-lens judge alone ranked the INT8-designed architecture first by one point, because every row changes a float-graph block before training and the compile protocol is frozen; its rows are grafted into section 4's NPU chapter. Every judge named the same fatal risk on the winner: the headline bets on a positive result in the regime where every published compact attempt is negative. That is why the oracle gate, the pre-registered criterion, the pretrained RGB ceiling and the negative-result cut-line are in the plan. The safe-plus proposal scored highest on safety and feasibility and lowest on novelty; its judges noted its mechanism (20 sigmoid scalars on a pretrained encoder) is expressively a subset of UperNet and confounded with loss weighting.

## Appendix B. Verification and corrections

Seven load-bearing claims, three adversarial votes each, all confirmed 3–0 against primary sources (the SpectralWaste paper text and Table IV via the Hugging Face mirror, the authors' code repo and issue tracker, BCAF's full text, the Mobilint SDK tutorial repo at its 2026-09-15 head, the ZeroWaste repo and VisDA organizer README). Corrections the verifiers and critic added, all applied above:
- The published Table IV rows are from-scratch, single-seed, weighted-CE, with an open reproduction issue; the released code never loads ImageNet weights. Hence the same-recipe rule and the pretrained RGB ceiling row.
- The authors' checkpoints are 12 files on a SharePoint folder with specific naming and a torch 2.2.2 dependency; downloadability was not tested from here.
- Mobilint's compiler accepts multi-input graphs via a JSON calibration manifest (Whisper decoder, SAM2 decoder, Qwen3-VL encoder examples); calibration tensors must match the declared input shapes; image-like inputs are channel-last. `cpu_offload` was removed from the BERT tutorial in v1.3.0.
- BCAF's 76.4 vs 55.9 two-phase comparison is probably not compute-matched, so two-phase training is a cheap hypothesis to test at compact scale, not a guaranteed gain.
- ZeroWaste's committed split lists are the 2021 cut; the final 4,503-frame split must be audited separately, and the final archive may require an access request.
- Not verified and therefore stated with qualifiers: the DINOv2 result on ZeroWaste (61.94), FusionSort's parameter count, HLRFF-Net's benchmark table, HREM-Net's parameter count.

## Appendix C. Sources

Datasets and baselines: SpectralWaste paper and Table IV https://huggingface.co/papers/2403.18033 · code and issue #1 https://github.com/ferpb/spectralwaste-segmentation · release https://huggingface.co/datasets/ferpb/spectralwaste-segmentation · BCAF https://huggingface.co/papers/2603.13941 · CMX https://huggingface.co/papers/2203.04838 · FusionSort https://arxiv.org/abs/2508.19798 · HREM-Net https://arxiv.org/abs/2607.16056 · ZeroWaste https://github.com/dbash/zerowaste and https://github.com/dbash/visda2022-org · COSNet https://huggingface.co/papers/2410.24139 · FANet https://huggingface.co/papers/2407.09379 · EWSegNet https://huggingface.co/papers/2606.13587 · BMVC 2025 ZeroWaste detection https://huggingface.co/papers/2508.18799 · TUNI https://huggingface.co/papers/2509.10005 · CARL https://huggingface.co/papers/2504.19223 · HS3-Bench https://github.com/nickstheisen/hyperseg.

Syllabus and program: SJSU catalog entry https://catalog.sjsu.edu/preview_course_nopop.php?catoid=13&coid=120206 · MSADI requirements https://www.sjsu.edu/applied-data-science/msadi/graduation-requirements.php · Mobilint–SJSU MOU coverage https://zdnet.co.kr/view/?no=20250916094518 · SJSU syllabus policy S16-9 https://www.sjsu.edu/senate/docs/S16-9.pdf.

Presentation and architecture precedents: EdgeNeXt https://huggingface.co/papers/2206.10589 · MobileViT https://huggingface.co/papers/2110.02178 · ConvNeXt https://huggingface.co/papers/2201.03545 · R(2+1)D https://huggingface.co/papers/1711.11248 · HybridSN https://huggingface.co/papers/1902.06701 · ChannelViT https://huggingface.co/papers/2309.16108 · DOFA https://huggingface.co/papers/2403.15356 · LESSViT code https://github.com/uiuctml/LESSViT · HyperspectralViTs https://huggingface.co/papers/2410.17248.

Hardware-aware design and quantization: AutoNeural https://huggingface.co/papers/2512.02924 · Bondarenko et al. (clipped softmax, gated attention) https://huggingface.co/papers/2306.12929 · I-ViT https://huggingface.co/papers/2207.01405 · FQ-ViT https://github.com/megvii-research/FQ-ViT · I-Segmenter https://huggingface.co/papers/2509.10334 · SLAB/PRepBN https://huggingface.co/papers/2405.11582 · ShuffleNetV2 https://huggingface.co/papers/1807.11164 · Mobilint SDK tutorial https://github.com/mobilint/mblt-sdk-tutorial · vision toolkit and NPU-vs-GPU table https://github.com/mobilint/mblt-vision-python · mblt-tracker https://github.com/mobilint/mblt-tracker.
