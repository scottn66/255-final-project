# SSFT-Wave — Session 1 Plan (DATA 255)

Group-facing map (short notes, numbered): [`bulletins/README.md`](bulletins/README.md) — **1.xx plan** · **2.xx board**. Checkboard: [`BOARD.md`](BOARD.md). GitHub: [scottn66/255-final-project](https://github.com/scottn66/255-final-project).

**Project:** reimplement a tiny dual-path spectral–spatial fusion transformer from scratch, replace integer band-index embeddings with wavelength-aware PE, and test whether the model degrades less than a vanilla ViT when bands are dropped or swapped.

**One-sentence success criterion:** a ~0.5M dual-path model that degrades less than a vanilla ViT when bands are dropped or swapped.

**This session (after approval):** write `PLAN.md` into the repo, run one tiny spike (attention + λ-PE + variable-C dummy forward), stop. No training. No Indian Pines download. No multi-GB archives.

**Sources actually read (do not invent layers):**

- SSFT — Musiat, Ebert, Wasenmüller. *SSFT: A Lightweight Spectral–Spatial Fusion Transformer for Generic Hyperspectral Classification*. arXiv:2604.15828 (17 Apr 2026). [abs](https://arxiv.org/abs/2604.15828) / [pdf](https://arxiv.org/pdf/2604.15828) / [html](https://arxiv.org/html/2604.15828). Implement **§3 only**. There is **no official code**. This is **not** SSFTT (TGRS 2022, `zgr6010/HSI_SSFTT`).
- LESSViT — Si, Wan, Wang, Do, Zhao (UIUC). *LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift*. arXiv:2605.18541 (18 May 2026). [abs](https://arxiv.org/abs/2605.18541) / [pdf](https://arxiv.org/pdf/2605.18541) / [project](https://uiuctml.github.io/LESSViT/). **Steal ideas, do not reproduce the paper.**

---

## Locked defaults (load-bearing, not TBD)

| Decision | Lock | Why |
|---|---|---|
| Input unit | **Labeled-center patches**, not whole-cube | Indian Pines / Pavia U are pixel-labeled scenes. SSFT §3 is cube→one class (HSI-Benchmark object/patch classification). We keep SSFT’s *modules and shapes*, but feed `P×P` patches with the **center pixel’s class**. |
| Patch size | **16×16** | `s=8` then yields `H/s=W/s=2`, `N=4` spatial tokens. Fusion is non-degenerate. If a later dataset patch is `<8`, set `s = patch_size` so `N=1`. |
| Split | **Spatial block split only** | Random-pixel splits leak 8-connected neighbors and fake 99% OA. Prefer a correct split over a clever attention variant. |
| Width | **D=64**, **4 heads**, **1 spectral MHSA + 1 spatial conv block + 1 cross-attn fusion** | SSFT §3.6. |
| Downsample | **s=8** (max-pool, both branches) | SSFT §3.6. |
| λ-PE v1 | **Sinusoidal 1D on λ in nm, internally converted to microns (`λ/1000`)** | LESSViT §2.2 / App. B idea without SSRoPE. |
| Tied projection | **Yes, on every C-mixing layer** | LESSViT §2.1. Required so `C` can change at test time. |
| Aux heads | **Off in spike and first real train**; `λ_aux=0.05` later | SSFT §3.5 / §4.3. |
| Attention impl | **From scratch** (`nn.Linear` / `softmax` only). Ban `nn.MultiheadAttention`, `F.multi_head_attention_forward`, timm, HF ViT blocks. | Project rule. |
| Repo root | **`/Users/scottnelson/Desktop/255/ssft-wave/`** | Group project lives under DATA 255. Do not dump files onto Desktop. |

---

## A. Paper extract

### A.1 SSFT §3 — what we copy

**Task (paper).** Cube `X ∈ R^{H×W×C}` → class `y ∈ {1,…,K}` (SSFT §3, first paragraph). Two parallel encoders + fusion + MLP head.

**Spectral encoder `E_s` (SSFT §3.1).**

1. Spatial max-pool by `s` → `X^{(s)↓} ∈ R^{H/s × W/s × C}`.
2. Flatten spatial grid: `N = (H/s)·(W/s)` locations. Per location, a length-`C` sequence of **scalars**.
3. Learnable `ϕ(·)` maps **each scalar band value** → `D` dims: `E = ϕ(X̃^{(s)}) ∈ R^{N×C×D}`.
4. **Paper:** add learned **channel embedding** `P ∈ R^{C×D}`, broadcast over `N`: `Z = E + P`. **We change this** (see A.3).
5. **One** spectral self-attention block. Paper eq. (1):
   `Attn(Z) = softmax(Q K^⊤ / √D) V`, softmax over the **channel** axis.
   `W_Q, W_K, W_V ∈ R^{D×D}`. “In practice … multi-head” (SSFT §3.1). Residual + two-layer FFN.
6. Mean-pool the `C` tokens per location → `h^{(s)} ∈ R^{H/s × W/s × D}`.

**Spatial encoder `E_p` (SSFT §3.2).**

1. `1×1` conv `ψ`: `C → D` → `X^{(p)} ∈ R^{H×W×D}`. **Paper layer is C-specific. We change this** (see A.3).
2. Max-pool by `s` → `R^{H/s × W/s × D}`.
3. **One** conv block: `3×3` conv + BatchNorm + GELU → `h^{(p)} ∈ R^{H/s × W/s × D}`.

**Fusion `Φ` (SSFT §3.3).** Flatten both maps to `N` tokens: `S, P ∈ R^{N×D}`. **Spatial tokens query, spectral tokens key/value.** Paper eq. (4):
`CA(P,S) = softmax(Q_p K_s^⊤ / √D) V_s`. Multi-head in practice. Residual + two-layer FFN. Reshape back to `H/s × W/s × D`.

**Heads (SSFT §3.4).** Adaptive average pool → one `D`-vector. Two-layer MLP + GELU → `z ∈ R^K`. Optional aux heads on `h^{(s)}` and `h^{(p)}` (same design); discarded at inference.

**Loss (SSFT §3.5).** `L = L_main + λ_aux (L_aux^{(s)} + L_aux^{(p)})`. Paper uses `λ_aux = 0.05` when enabled (§4.3).

**Training recipe we will use later (SSFT §3.6, copied):**

- AdamW, lr `4e-4`, weight decay `1e-2`, batch size `8`
- ≤50 epochs, StepLR `γ=0.1` every 20 epochs
- Early stop patience 10 on **validation AA** (paper says val accuracy; we use AA because OA is a trap on Indian Pines)
- **No PCA / no band-reduction**
- Paper found **no augmentation best** on HSI-Benchmark (§4.3 / Tab. 5). Default: no aug. Optional later: H/V flip only (spatial, not spectral).

**Paper gaps (not specified in §3; we lock defaults rather than invent extra blocks):**

| Gap | Our lock |
|---|---|
| FFN hidden width | `4D = 256`, GELU (Transformer convention) |
| LayerNorm | **Pre-LN** before MHSA and FFN (paper omits LN; training without it is unstable) |
| Output projection `W_O` | Yes. Multi-head split/concat then `Linear(D,D)` |
| Head dim | `d_h = D / 4 = 16` |
| `ϕ` | shared `Linear(1, D)` — already channel-agnostic |
| Param count | Paper claims **516k**. Faithful D=64 reconstruction is **~0.16–0.23M** (see A.4). We do **not** add mystery layers to chase 516k. Session 2 prints `sum(p.numel())` and we report the honest number. |

**Ablation from the paper we will reuse later (SSFT §4.3, Tab. 4):** zeroing the spatial map before fusion drops overall much more than zeroing spectral (38 vs 71 vs 85 on their benchmark). Spatial path is not optional.

### A.2 LESSViT §2 — what we steal vs skip

**Steal (v1):**

1. **Tied / channel-agnostic first projection (LESSViT §2.1).** Shared weights across bands so `C` is not baked into `Conv2d(C, D, 1×1)` or `Linear(C·P², D)`. Paper: partition `C×H×W` into `C × (H/P) × (W/P)` patches of size `P×P`; shared `W ∈ R^{P²×D}` → tokens `R^{N×C×D}`.
2. **Wavelength-aware positional encoding (LESSViT §2.2 + Appendix B).** PE must **not** depend on packed channel index. Appendix B uses `θ_i(λ) = λ / τ_i^{(λ)}` with geometric periods (1D RoPE on central wavelength). **v1 = sinusoidal 1D additive PE on λ**, not RoPE.
3. **Cross-spectral test protocol (LESSViT §3.1), shrunk to Indian Pines / Pavia.** Train on one frozen band subset; test on ID, complementary/swapped subset, random ~50% drop, optional full-`C` expansion. Partition VNIR/SWIR at **1000 nm** as they do.

**Skip (reject if it creeps in):**

- LESS Attention, Kronecker / low-rank spatial–spectral product, AttenPool, CLS-token grid `(N+1)×(C+1)` (LESSViT §2.2, Alg. 1)
- SSRoPE / 1D RoPE on λ (Appendix B) — **later stretch, not v1**
- HyperMAE, 75% decoupled masking, hierarchical channel sampling (LESSViT §2.3)
- ViT-B, 12 encoder / 8 decoder blocks, D=768, 4×H200, 96-hour pretrain (Appendix C)
- SpectralEarth / EnMAP 202-band pretrain (~TB)
- DOFA wavelength-conditioned *weight generation* (mentioned in LESSViT §1; out of scope)

### A.3 Copy vs change

| Module | SSFT paper | SSFT-Wave (ours) |
|---|---|---|
| `ϕ` scalar→D | learnable, per-scalar | **copy:** shared `Linear(1,D)` |
| Spectral PE | learned `P ∈ R^{C×D}` indexed by band **index** (§3.1) | **change:** additive sinusoidal PE of `λ_nm` (LESSViT idea). Index-PE kept as an **ablation baseline**. |
| Spectral MHSA | 1 block, 4 heads, residual+FFN | **copy**, written from scratch |
| Spatial `1×1` | `Conv2d(C, D, 1)` — **fixed C** | **change:** tied `Conv2d(1, D, 1)` per band, then **masked mean** over C. `C` free at test. |
| Spatial `3×3+BN+GELU` | 1 block | **copy** |
| Fusion | spatial Q, spectral K/V, 4 heads | **copy**, from scratch |
| Head | adaptive pool + 2-layer MLP | **copy**; aux off until later |
| Forward API | `forward(X)` implicit fixed C | **`forward(cube, wavelengths, band_mask)`** — model never assumes a fixed C |

**Vanilla ViT baseline (the thing that should break):** patchify `16×16×C` with a **first linear that collapses bands**, `Linear(C·p·p, D)` or `Conv2d(C, D, kernel=p, stride=p)` with **fixed C**. Same D/heads budget, ~comparable params. At test, `C` cannot change; dropped bands are **zero-filled** to `C_train`. Swapped subset of the same length is a permutation of the feature slots the first layer memorized.

**Index-PE protocol lock (so the ablation is a real test):** SSFT-index uses packed positions `0..C-1` of the **current tensor**, not a `C_full` table of original sensor indices. Complementary subset of the same `C` therefore reuses the same embedding rows for **different wavelengths**. λ-PE is invariant to packing because wavelengths travel with bands. (If we used original-index PE, drop would be too easy for the “index” baseline and would not isolate the LESSViT claim.)

### A.4 Tensor shapes (labeled-center 16×16, D=64, s=8)

Let `B` = batch, `C` = current number of **unmasked** bands (or full C with a mask — see API), `K` = classes (IP=16, PU=9).

Input (NCHW, PyTorch): `cube: (B, C, 16, 16)`, `wavelengths: (C,)` in nm, `band_mask: (C,)` bool.

| Hop | Tensor | Shape |
|---|---|---|
| 0 input | `X` | `(B, C, 16, 16)` |
| 1a spectral pool | `X_s` | `(B, C, 2, 2)` |
| 1b flatten spatial | `X̃_s` | `(B, N=4, C)` scalars |
| 1c `ϕ` | `E` | `(B, 4, C, 64)` |
| 1d + λ-PE (broadcast N) | `Z` | `(B, 4, C, 64)` |
| 1e MHSA over C (batch over `B·N`) | `Z'` | `(B, 4, C, 64)` |
| 1f mean-pool C (masked) | `h^{(s)}` | `(B, 64, 2, 2)` |
| 2a tied 1×1 per band | | `(B, C, 64, 16, 16)` then masked-mean C → `(B, 64, 16, 16)` |
| 2b max-pool s=8 | | `(B, 64, 2, 2)` |
| 2c 3×3+BN+GELU | `h^{(p)}` | `(B, 64, 2, 2)` |
| 3 flatten | `S, P` | `(B, 4, 64)` each |
| 3 cross-attn (N as seq) | fused | `(B, 4, 64)` |
| 4 adaptive avg pool | | `(B, 64)` |
| 4 MLP | logits | `(B, K)` |

**Attention internals (from scratch):**

- `Q,K,V: (B*, S, 64) → view (B*, 4, S, 16)` where `S=C` (spectral) or `S=N` (fusion)
- scores `(B*, 4, S_q, S_k) = Q @ K^T / √16`
- optional key mask: `band_mask` → `-inf` on masked spectral keys (spike + later)
- softmax last dim ≈ 1
- concat heads → `(B*, S_q, 64)` → `W_O`

**1-pixel spectrum (spike / unit test):** `H=W=1`, `s=1`, `N=1`, sequence length `C`. MHSA over `C` must not crash.

**Honest param count (D=64, FFN 4D, no aux, Indian Pines C=200 for the *index* variant only):**

- `ϕ` Linear(1,64) ≈ 128
- index `P` (index variant only) = 200×64 = 12,800; λ-PE = 0
- each MHSA+FFN+2×LN ≈ 50k; two of them (spectral + fusion) ≈ 100k
- spatial: paper `1×1` C→D ≈ 13k **or** tied `1×1` ≈ 128; `3×3` ≈ 37k; BN ≈ 128
- head `64→256→K` ≈ 21k
- **SSFT-index ~180k; SSFT-Wave ~160k.** Paper 516k is ~3× this reconstruction. We report ours; we do not inflate.

### A.5 Three risks

1. **Shape mismatch on variable C.** Spatial `Conv2d(C_in, D, 1)` and ViT `Linear(C·P², D)` explode or silently reshape when `C` changes. Mitigation: every C-mixing op is tied or a reduction (masked mean). Unit test: same module, `C=32` then `C=20`, no rebuild. Fusion does **not** see `C` (spectral mean-pool already collapsed it); mask must be applied **before** that pool.
2. **Spatial-split leakage.** 16×16 patches centered in a train block still overlap a test block. Mitigation: **discard** (or assign to neither) any patch whose spatial support intersects a different-split cell. Never random-pixel split. State this in the writeup; 99% OA on Indian Pines with a pixel split is not a result.
3. **PE scale on λ in nm.** Raw 400–2500 in `sin(λ / 10000^{2i/D})` saturates low frequencies (LESSViT App. B scales `λ` by learned/geometric `τ_i`). Mitigation: convert to **microns** (`λ/1000` → ~0.4–2.5) before the standard Transformer sinusoid; expose `lambda_scale` (default 1000). Test: PE of 800 nm is closer to 810 nm than to 2100 nm in L2.

### A.6 Patch vs cube (decision)

**Locked: labeled-center 16×16 patches, spatial block split, s=8.**

SSFT’s paper is cube→class. Indian Pines is a 145×145 scene with 16 pixel classes (Alfalfa n=46, Oats n=20, Soybean-mintill n=2455). Whole-cube classification is the wrong task. We use SSFT as a **patch classifier** whose internal cube is 16×16×C.

---

## B. Build plan (later Grok Build sessions)

Repo:

```
ssft-wave/
  PLAN.md
  README.md
  requirements.txt          # torch, pytest, numpy; scipy optional for .mat
  src/
    attention.py            # MHSA, cross-attn, residual+FFN  (from scratch)
    pe.py                   # sinusoidal 1D λ-PE; index-PE baseline
    ssft.py                 # dual-path SSFT / SSFT-Wave
    vit.py                  # vanilla ViT, fixed C
    data.py                 # Indian Pines / Pavia U, wavelengths, spatial split
    train.py                # AdamW loop, early stop, band-drop eval
  tests/
    test_attention.py
    test_pe.py
    test_ssft.py
    test_data.py
  scripts/
    download_data.py        # small public .mat only
    run_banddrop.py
  data/                     # gitignored cubes + wavelength tables
```

### Phase 0 — Spike (this session, after approval)

- **In:** papers extract above.
- **Out:** `src/attention.py`, `src/pe.py`, `tests/test_attention.py`, `tests/test_pe.py`, a stub variable-C forward (function or tiny module), `requirements.txt`, `PLAN.md`.
- **Done when:** `pytest` green on (1) MHSA/cross-attn output shapes, (2) softmax rows ≈ 1, (3) 1-pixel spectrum of length C, (4) sinusoidal PE on a wavelengths vector, (5) dummy cube `2×32×16×16` then `2×20×16×16` through PE+spectral attention **without rebuilding the module**.
- **Size:** ~150–250 lines. No train, no download.
- **Est.:** 1 session.

### Phase 1 — Attention primitives complete + docs

- **In:** spike.
- **Out:** residual+FFN wrapper, key-padding/band mask, `W_O`, explicit ban-list comment in `attention.py`.
- **Done when:** tests cover mask (masked keys get ~0 attention mass) and `d_h` not dividing D raises.
- **Est.:** same session as 0 if time, else start of session 2.

### Phase 2 — SSFT-index forward on a dummy cube + param count  ← **session 2 default**

- **In:** attention primitives.
- **Out:** `src/ssft.py` with `pe_mode="index"`, dummy `B×C×16×16`, `print(n_params)`.
- **Done when:** forward `(B,K)` logits; param count reported vs paper 516k; unit test `test_ssft.py` for shapes at `C=32` and `C=32` again (index table sized at init — index variant **may** still be C-fixed; Wave variant is the one that must accept C change).
- **Est.:** 1 session.

### Phase 3 — λ-PE + `band_mask` API

- **In:** SSFT-index.
- **Out:** `pe_mode="wave"`, tied spatial `1×1`, `forward(cube, wavelengths, band_mask)`.
- **Done when:** same module runs `C=32` then `C=20`; permuting bands+wavelengths together ≈ invariant logits (same seed, eval mode, no dropout); permuting bands **without** wavelengths is not.
- **Est.:** 1 session.

### Phase 4 — Indian Pines loader + spatial split + wavelength table

- **In:** public Indian Pines `.mat` (~few MB; **not** HSI-Benchmark 240 GB, **not** SpectralEarth). Pavia U next.
- **Out:** `src/data.py`: cube, labels, `wavelengths_nm` length C, disjoint **block** split, 16×16 patch dataset, overlap-discard rule.
- **Wavelengths:**
  - Indian Pines: AVIRIS ~400–2500 nm, 10 nm sampling. Scene is 145×145×220 (often cited 224); drop water-absorption `[104–108], [150–163], 220` (1-indexed on the 220-band product) → **200 bands**. Store a 200-vector, not indices. Prefer Purdue calibration file if present; else `linspace(400, 2500, 220)` then drop the same slots, and **document the approximation**.
  - Pavia U: 610×340×103, ROSIS **430–860 nm**, ~4 nm. v1 table: `linspace(430, 860, 103)`.
- **Split algorithm:** tile the scene into a `G×G` grid (default G=5 for IP → 29×29 cells). Assign cells to train/val/test with a fixed seed targeting ~60/20/20 of **labeled** pixels, keeping rare classes present in train if possible. A patch is train iff its **center** is in a train cell **and** its 16×16 support does not intersect a val/test cell (else drop). Same for val/test. Log how many labeled pixels were dropped to the buffer.
- **Done when:** a test on a fake 32×32 labeled grid proves train/test patches do not share pixels; wavelength vector length equals C; **no random-pixel API**.
- **Est.:** 1 session.

### Phase 5 — Tiny overfit

- **In:** real loader + SSFT-Wave.
- **Out:** train on 1–8 patches, CE → ~0, train OA = 1.0.
- **Done when:** the loop (AdamW, one device, no val) works. Not a generalization result.
- **Est.:** half session.

### Phase 6 — Vanilla ViT baseline

- **In:** same patches / same split / same optimizer.
- **Out:** `src/vit.py`, fixed-C first layer, comparable param budget (print both).
- **Done when:** overfit smoke + ID eval on the spatial val split.
- **Est.:** 1 session.

### Phase 7 — Band-drop table (the actual experiment)

Train **one** frozen in-distribution subset; **do not** retune per test config.

**ID subset (lock):** following LESSViT §3.1, split at 1000 nm.

- Indian Pines: ~bands with `λ < 1000` vs `λ ≥ 1000`. Train on a **Pro-VNIR-style** mix: 80% of VNIR + 20% of SWIR, uniformly subsampled, frozen once, saved as `configs/ip_id_bands.json` (indices **and** nm).
- Complementary / swapped: the leftover bands (target similar `C` if possible).
- Random ~50% drop: RNG subset of the **ID** bands, 3 seeds.
- Optional full-C expansion: all 200 (or 103) at test.

**Models:** vanilla ViT (zero-fill to `C_id` when C shrinks; same-length permute when swapped) vs SSFT-index vs SSFT-Wave.

**Metrics:** **OA and macro-F1 and AA**. OA alone is a trap (Soybean-mintill 2455 vs Oats 20). Primary comparison = **relative drop** `(ID − OOD) / ID` on AA and macro-F1.

**Done when:** one table on Indian Pines, 3 seeds, plus a short “did Wave drop less?” sentence. Pavia U repeats the same table. **Do not claim SOTA.**

- **Est.:** 1–2 sessions (CPU-ok; 50 epochs × batch 8 × ~0.2M params is small).

### Phase 8 — Two ablations + writeup

Pick **two** (not more):

1. Fused vs spectral-only vs spatial-only (SSFT Tab. 4 protocol: zero the other map before fusion).
2. Index PE vs λ-PE (already in the main table; write it up as an ablation).
3. Tied embed vs fixed-C 1×1 (if needed to explain a Wave vs index gap).

Writeup: method (SSFT §3 + LESSViT PE/tied), spatial-split warning, table, limitations (tiny model, two scenes, no foundation-model comparison).

- **Est.:** 1 session.

### Later stretch (optional, not scheduled)

- 13-band EuroSAT-MS / Sentinel-2 (not RGB EuroSAT)
- 1D RoPE / SSRoPE (LESSViT App. B)
- Aux heads `λ_aux=0.05`
- Pavia Centre / Salinas as extra scenes

### Hard out of scope

- DATA 255 Lab 1 (240×240 ImageNet-20, ONNX, MXQ, Mobilint, MLA100, NPU)
- Fine-tuning SatMAE / Prithvi / SpectralGPT / AnySat / RemoteCLIP / LESSViT weights
- Downloading HSI-Benchmark (~240 GB) or SpectralEarth (~TB)
- Claiming SOTA on Indian Pines
- Wildfire detection (downstream later, not this build)
- Reproducing LESS Attention / HyperMAE / 96-hour pretrain
- Confusing SSFT with SSFTT 2022

---

## C. Spike spec (only code this session, after approval)

Files:

- `ssft-wave/src/attention.py`
  - `scaled_dot_product_attention(q, k, v, mask=None)`
  - `MultiHeadSelfAttention` (qkv from same x)
  - `MultiHeadCrossAttention` (q from x, kv from context)
  - No `nn.MultiheadAttention`.
- `ssft-wave/src/pe.py`
  - `sinusoidal_wavelength_pe(wavelengths_nm, d_model, lambda_scale=1000.0) -> (C, D)`
- `ssft-wave/src/spectral_stub.py` (or a function in tests)
  - `tied Linear(1,D)` + λ-PE + MHSA over C on `(B,C,H,W)`; mean-pool C
  - `band_mask` selects bands before attention
- Tests as in Phase 0 done-when.
- `requirements.txt`: `torch`, `pytest`, `numpy`. Stop if anything else is missing.

Dummy cube: `B=2, C=32, H=W=16`, fake `wavelengths = linspace(400, 2500, 32)`, random boolean mask keeping ≥8 bands; second forward `C=20` new wavelengths.

---

## D. Session 1 report

- Plan A+B lives in `ssft-wave/PLAN.md` (this file).
- Spike **proved** (17 pytest passed, torch 2.13.0 / pytest 9.1.1 / numpy 2.5.2, Python 3.12 venv):
  - MHSA and cross-attention output shapes; softmax rows ≈ 1; masked keys get ~0 mass.
  - 1-pixel spectrum of length C attends without crash.
  - Sinusoidal λ-PE on a wavelengths vector; 800 nm closer to 810 nm than to 2100 nm.
  - Dummy cube `2×32×16×16` with random `band_mask`, then `2×20×16×16`, **same `SpectralStub` module**, no rebuild. `φ` is `Linear(1, 64)`.
- Spike did **not** train, did **not** fetch Indian Pines, did **not** implement full SSFT (session 2).
- Param count: n/a this session (session 2).
- Next-session prompt (session 2):

```
Session 2 of SSFT-Wave (DATA 255). Repo: Desktop/255/ssft-wave. PLAN.md is the spec. BOARD.md + tickets/SSFT-2.md is the board.

Do this only:
1. Finish src/ssft.py as SSFT-index: D=64, s=8, 1 spectral MHSA (4 heads, from src/attention.py), 1 spatial block (1×1 then 3×3+BN+GELU), 1 cross-attn fusion (spatial Q, spectral K/V, 4 heads), adaptive pool + 2-layer MLP. Aux heads off.
2. Dummy cube B×C×16×16 (e.g. 2×32×16×16) → logits (B, K=16). Print exact param count vs paper 516k; do not add mystery layers.
3. tests/test_ssft.py: output shape; residual paths exist; index PE is C-sized.
4. Keep forward(cube, wavelengths=None, band_mask=None) signature even if index mode ignores wavelengths.
5. No Indian Pines download, no training, no LESS Attention.

Do not use nn.MultiheadAttention. Do not touch Lab 1 / NPU / foundation weights.
```

---

## Working rules for implementers

- Quote SSFT §3 / LESSViT §2; do not hallucinate extra blocks.
- Prefer small files and tests over notebooks.
- If a choice appears: spatial-split correctness > attention cleverness.
- `forward(cube, wavelengths, band_mask)` never assumes a fixed C in the Wave path.
