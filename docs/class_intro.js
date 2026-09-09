/**
 * SSFT-Wave — DATA 255 class introduction
 * Brief project proposal for class and professor.
 *
 * Run:  node class_intro.js
 * Out:  SSFT-Wave_class_intro.pptx
 */

const pptxgen = require("pptxgenjs");

const COLOR = {
  darkBg: "1B2420",
  cream: "F1EDE3",
  beige: "E8DFC8",
  cardCream: "F5EFE2",
  ink: "1B2420",
  body: "3B3F3A",
  muted: "7A7F76",
  mutedLight: "A7A79A",
  forest: "3B5D3A",
  forestDeep: "2C4A2B",
  moss: "7A9A6A",
  terracotta: "C26A33",
  terracottaSoft: "D48A5B",
  hairline: "C9C4B4",
  hairlineDark: "2A3530",
  titleOnDark: "EFE9D9",
  subtitleOnDark: "C9C4B4",
  white: "FFFFFF",
};

const FONT = {
  head: "Georgia",
  body: "Arial",
  mono: "Courier New",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "SSFT-Wave — DATA 255 Project Introduction";
pres.author = "SJSU DATA 255";
pres.subject = "Brief class introduction: problem, gap, data, method, evaluation, timeline";

const W = 13.333;
const H = 7.5;
const MARGIN_X = 0.85;

function addEyebrow(slide, label, pageStr, onDark) {
  const labelColor = onDark ? COLOR.mutedLight : COLOR.muted;
  slide.addText(label, {
    x: MARGIN_X,
    y: 0.42,
    w: 9.2,
    h: 0.32,
    fontFace: FONT.body,
    fontSize: 11,
    color: labelColor,
    charSpacing: 3.2,
    margin: 0,
  });
  slide.addText(pageStr, {
    x: W - MARGIN_X - 1.8,
    y: 0.42,
    w: 1.8,
    h: 0.32,
    fontFace: FONT.body,
    fontSize: 11,
    color: labelColor,
    charSpacing: 2,
    align: "right",
    margin: 0,
  });
}

function addFooterCite(slide, cite, onDark) {
  slide.addText(cite, {
    x: MARGIN_X,
    y: H - 0.42,
    w: W - 2 * MARGIN_X,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 10,
    color: onDark ? COLOR.mutedLight : COLOR.muted,
    margin: 0,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x,
    y,
    w,
    h,
    fill: { color: fill },
    line: { color: COLOR.hairline, width: 0.75 },
  });
}

// ======================================================================
// 01  Title
// ======================================================================
function slide1() {
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  addEyebrow(s, "SJSU  ·  DATA 255  ·  DEEP LEARNING FOR COMPUTER VISION", "01 / 09", true);

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X,
    y: 1.45,
    w: 2.55,
    h: 0.42,
    fill: { color: COLOR.darkBg },
    line: { color: COLOR.mutedLight, width: 0.75 },
  });
  s.addText("PROJECT INTRODUCTION", {
    x: MARGIN_X,
    y: 1.45,
    w: 2.55,
    h: 0.42,
    fontFace: FONT.body,
    fontSize: 10,
    color: COLOR.subtitleOnDark,
    charSpacing: 2.4,
    align: "center",
    valign: "middle",
    margin: 0,
  });

  s.addText("SSFT-Wave", {
    x: MARGIN_X,
    y: 2.1,
    w: 11.4,
    h: 1.05,
    fontFace: FONT.head,
    fontSize: 52,
    color: COLOR.titleOnDark,
    margin: 0,
  });
  s.addText("Does a tiny spectral–spatial transformer survive dropped or swapped bands better than a vanilla ViT?", {
    x: MARGIN_X,
    y: 3.25,
    w: 11.2,
    h: 1.15,
    fontFace: FONT.head,
    fontSize: 22,
    italic: true,
    color: COLOR.terracottaSoft,
    margin: 0,
  });
  s.addText("A class-scale reimplementation of SSFT §3, with wavelength-aware positional encoding instead of integer band-index embeddings. We do not claim SOTA, and we do not fine-tune a foundation model.", {
    x: MARGIN_X,
    y: 4.55,
    w: 10.8,
    h: 0.85,
    fontFace: FONT.body,
    fontSize: 15,
    color: COLOR.subtitleOnDark,
    margin: 0,
  });

  s.addShape(pres.shapes.LINE, {
    x: MARGIN_X,
    y: 5.7,
    w: W - 2 * MARGIN_X,
    h: 0,
    line: { color: COLOR.hairlineDark, width: 0.75 },
  });

  const meta = [
    { k: "COURSE", v: "DATA 255  ·  FALL 2026" },
    { k: "REPO", v: "github.com/scottn66/255-final-project" },
    { k: "PAPERS", v: "SSFT 2604.15828  ·  LESSViT 2605.18541" },
  ];
  meta.forEach((m, i) => {
    const x = MARGIN_X + i * 3.9;
    s.addText(m.k, {
      x,
      y: 5.9,
      w: 3.6,
      h: 0.26,
      fontFace: FONT.body,
      fontSize: 10,
      color: COLOR.mutedLight,
      charSpacing: 2,
      margin: 0,
    });
    s.addText(m.v, {
      x,
      y: 6.2,
      w: 3.7,
      h: 0.32,
      fontFace: FONT.body,
      fontSize: 13,
      bold: true,
      color: COLOR.titleOnDark,
      margin: 0,
    });
  });

  s.addNotes(
    "Open with the one-sentence success criterion. SSFT is Musiat et al. 2026, not SSFTT 2022. LESSViT is the source of two ideas only (tied embed + wavelength PE), not the model we reproduce."
  );
}

// ======================================================================
// 02  Problem
// ======================================================================
function slide2() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "PROBLEM STATEMENT", "02 / 09");

  s.addText("Hyperspectral cubes are not RGB with extra channels.", {
    x: MARGIN_X,
    y: 0.9,
    w: 11.6,
    h: 0.7,
    fontFace: FONT.head,
    fontSize: 28,
    color: COLOR.ink,
    margin: 0,
  });
  s.addText("Each band is a physical wavelength. Sensors differ in coverage, sampling, and how many channels they return. A model trained on a fixed C often cannot even run when bands are dropped, swapped, or expanded.", {
    x: MARGIN_X,
    y: 1.62,
    w: 11.6,
    h: 0.7,
    fontFace: FONT.body,
    fontSize: 15,
    color: COLOR.body,
    margin: 0,
  });

  const fails = [
    {
      n: "01",
      t: "Dropped bands",
      d: "Missing channels at test time. A first linear layer of size C·P² cannot change C without a rebuild or a hack (zero-fill).",
    },
    {
      n: "02",
      t: "Swapped / complementary bands",
      d: "Same count C, different wavelengths. Index embeddings reuse the same rows for different physics.",
    },
    {
      n: "03",
      t: "The 99% OA trap",
      d: "Random-pixel splits leak neighbors into train and test. On Indian Pines a 3×3 window already covers ~31% of test pixels at 5% train, ~86% at 25%. 3D-CNN OA can drop ~37% under a true spatial split.",
    },
  ];
  fails.forEach((f, i) => {
    const x = MARGIN_X + i * 3.9;
    card(s, x, 2.5, 3.7, 3.55, COLOR.white);
    s.addText(f.n, {
      x: x + 0.22,
      y: 2.68,
      w: 1.2,
      h: 0.35,
      fontFace: FONT.mono,
      fontSize: 13,
      color: COLOR.terracotta,
      margin: 0,
    });
    s.addText(f.t, {
      x: x + 0.22,
      y: 3.1,
      w: 3.25,
      h: 0.7,
      fontFace: FONT.head,
      fontSize: 18,
      color: COLOR.ink,
      margin: 0,
    });
    s.addText(f.d, {
      x: x + 0.22,
      y: 3.85,
      w: 3.25,
      h: 1.9,
      fontFace: FONT.body,
      fontSize: 14,
      color: COLOR.body,
      margin: 0,
    });
  });

  addFooterCite(
    s,
    "LESSViT §1–§2 (arXiv:2605.18541). Leakage numbers: Nalepa, Myller, Kawulok, arXiv:1811.03707 / IEEE GRSL 2019."
  );
  s.addNotes(
    "Problem is spectral configuration shift plus evaluation leakage. Do not spend this slide on wildfire or SOTA leaderboards."
  );
}

// ======================================================================
// 03  Gap
// ======================================================================
function slide3() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "RESEARCH GAP", "03 / 09");

  s.addText("Two recent papers, neither of which is our assignment as-is.", {
    x: MARGIN_X,
    y: 0.9,
    w: 11.6,
    h: 0.55,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  card(s, MARGIN_X, 1.65, 5.55, 3.55, COLOR.white);
  s.addText("SSFT  ·  arXiv:2604.15828", {
    x: 1.1,
    y: 1.82,
    w: 5.1,
    h: 0.3,
    fontFace: FONT.mono,
    fontSize: 12,
    color: COLOR.forest,
    margin: 0,
  });
  s.addText("A 516k dual-path classifier we can actually reimplement.", {
    x: 1.1,
    y: 2.18,
    w: 5.05,
    h: 0.7,
    fontFace: FONT.head,
    fontSize: 18,
    color: COLOR.ink,
    margin: 0,
  });
  s.addText("§3 factorizes spectral MHSA and a tiny spatial CNN, then fuses with cross-attention (spatial Q, spectral K/V). No official code. The spectral PE is a learned table P ∈ R^{C×D} indexed by band index — so C is baked in. Not SSFTT (TGRS 2022).", {
    x: 1.1,
    y: 2.95,
    w: 5.05,
    h: 1.95,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  card(s, 7.0, 1.65, 5.45, 3.55, COLOR.white);
  s.addText("LESSViT  ·  arXiv:2605.18541", {
    x: 7.25,
    y: 1.82,
    w: 5.0,
    h: 0.3,
    fontFace: FONT.mono,
    fontSize: 12,
    color: COLOR.terracotta,
    margin: 0,
  });
  s.addText("The robustness recipe we will steal, not reproduce.", {
    x: 7.25,
    y: 2.18,
    w: 5.0,
    h: 0.7,
    fontFace: FONT.head,
    fontSize: 18,
    color: COLOR.ink,
    margin: 0,
  });
  s.addText("§2.1 tied / channel-agnostic projection so C can change. §2.2 / App. B wavelength-aware PE on λ in nm, not packed index. The rest (LESS Attention, HyperMAE, ViT-B, 4×H200, 96-hour pretrain) is out of scope for this class.", {
    x: 7.25,
    y: 2.95,
    w: 5.0,
    h: 1.95,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X,
    y: 5.4,
    w: W - 2 * MARGIN_X,
    h: 1.35,
    fill: { color: COLOR.beige },
  });
  s.addText("THE GAP", {
    x: 1.1,
    y: 5.52,
    w: 2.2,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.terracotta,
    charSpacing: 2,
    margin: 0,
  });
  s.addText("SSFT is small enough to train from scratch, but its first layers assume a fixed C. LESSViT is sensor-flexible, but it is a foundation-model paper. Nobody has shown — at class scale — whether swapping index PE for λ-PE on a ~0.5M dual-path model is enough to degrade less than a vanilla ViT under band drop/swap.", {
    x: 1.1,
    y: 5.82,
    w: 11.2,
    h: 0.8,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  addFooterCite(s, "Musiat, Ebert, Wasenmüller, SSFT §3. Si, Wan, Wang, Do, Zhao, LESSViT §2. Do not cite zgr6010/HSI_SSFTT.");
  s.addNotes(
    "If asked about SOTA: SSFT reports strong HSI-Benchmark numbers at 516k params; we are not chasing that leaderboard. If asked about LESSViT compute: skip it."
  );
}

// ======================================================================
// 04  Novelty
// ======================================================================
function slide4() {
  const s = pres.addSlide();
  s.background = { color: COLOR.beige };
  addEyebrow(s, "NOVELTY  ·  WHAT IS ACTUALLY NEW HERE", "04 / 09");

  s.addText("A controlled, honest class experiment — not a new backbone.", {
    x: MARGIN_X,
    y: 0.9,
    w: 11.6,
    h: 0.6,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  const items = [
    { k: "From scratch", v: "We write Q, K, V, scaled dot-product, multi-head split/concat, residual+FFN, and cross-attention ourselves. No nn.MultiheadAttention, no timm, no HuggingFace ViT blocks, no pretrained weights." },
    { k: "One surgical graft", v: "Copy SSFT §3. Steal two LESSViT ideas (tied projection, wavelength PE) — not LESS Attention, not SSRoPE. Sinusoidal λ-PE stands in for their rotary encoding. The graft plus the band-drop protocol is the class experiment, not a new operator." },
    { k: "The claim we will test", v: "SSFT-Wave should lose less AA / macro-F1 than a fixed-C ViT when bands are dropped or swapped. Relative drop (ID − OOD) / ID is the headline, not raw OA." },
  ];
  items.forEach((it, i) => {
    const y = 1.7 + i * 1.45;
    s.addShape(pres.shapes.RECTANGLE, {
      x: MARGIN_X,
      y,
      w: 0.12,
      h: 1.28,
      fill: { color: i === 2 ? COLOR.terracotta : COLOR.forest },
    });
    s.addText(it.k, {
      x: 1.2,
      y,
      w: 11.1,
      h: 0.38,
      fontFace: FONT.head,
      fontSize: 18,
      color: COLOR.ink,
      margin: 0,
    });
    s.addText(it.v, {
      x: 1.2,
      y: y + 0.4,
      w: 11.1,
      h: 0.8,
      fontFace: FONT.body,
      fontSize: 15,
      color: COLOR.body,
      margin: 0,
    });
  });

  s.addNotes("Deep-research check: tied embed and λ-PE are already in LESSViT §2; SSRoPE is rotary, not additive sinusoids. Do not claim a new mechanism. Claim the graft onto SSFT fusion plus a class-scale protocol. Paper 516k vs our D=64 reconstruction ~0.16–0.23M — report ours.");
}

// ======================================================================
// 05  Dataset
// ======================================================================
function slide5() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "DATASET", "05 / 09");

  s.addText("Public, small, and we store wavelengths in nanometers.", {
    x: MARGIN_X,
    y: 0.9,
    w: 11.6,
    h: 0.55,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  // Two dataset cards
  card(s, MARGIN_X, 1.6, 5.7, 3.55, COLOR.white);
  s.addText("PRIMARY", {
    x: 1.1,
    y: 1.75,
    w: 2.2,
    h: 0.26,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.terracotta,
    charSpacing: 2,
    margin: 0,
  });
  s.addText("Indian Pines", {
    x: 1.1,
    y: 2.05,
    w: 5.2,
    h: 0.42,
    fontFace: FONT.head,
    fontSize: 24,
    color: COLOR.ink,
    margin: 0,
  });
  s.addText("AVIRIS  ·  145×145  ·  16 classes  ·  Purdue 220-band cube, commonly 200 after dropping water-absorption bands  ·  ~400–2500 nm  ·  10,249 labeled pixels (Oats 20 vs Soybean-mintill 2455).", {
    x: 1.1,
    y: 2.55,
    w: 5.2,
    h: 1.15,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });
  s.addText("Labeled-center 16×16 patches. Spatial block split (5×5 cells). Discard any patch whose support crosses a split boundary.", {
    x: 1.1,
    y: 3.75,
    w: 5.2,
    h: 1.1,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  card(s, 7.05, 1.6, 5.4, 3.55, COLOR.white);
  s.addText("SECOND SCENE", {
    x: 7.3,
    y: 1.75,
    w: 3.2,
    h: 0.26,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.forest,
    charSpacing: 2,
    margin: 0,
  });
  s.addText("Pavia University", {
    x: 7.3,
    y: 2.05,
    w: 4.9,
    h: 0.42,
    fontFace: FONT.head,
    fontSize: 24,
    color: COLOR.ink,
    margin: 0,
  });
  s.addText("ROSIS  ·  9 urban classes  ·  103 bands  ·  ~430–860 nm (sources vary 850/860). Captured 610×610; after discarding no-data strips the usual cube is 610×340×103. Same patch and split rules.", {
    x: 7.3,
    y: 2.55,
    w: 4.9,
    h: 1.35,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });
  s.addText("Will not download: HSI-Benchmark (~240 GB) or SpectralEarth (~TB).", {
    x: 7.3,
    y: 4.05,
    w: 4.9,
    h: 0.75,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X,
    y: 5.35,
    w: W - 2 * MARGIN_X,
    h: 1.4,
    fill: { color: COLOR.beige },
  });
  s.addText("NON-NEGOTIABLE", {
    x: 1.1,
    y: 5.48,
    w: 3,
    h: 0.26,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.terracotta,
    charSpacing: 2,
    margin: 0,
  });
  s.addText("Every sample carries a wavelengths_nm vector, not just channel indices. Random-pixel splits are forbidden: neighboring pixels of the same field leak across the cut and manufacture 99% OA. Spatial (disjoint-region) split only.", {
    x: 1.1,
    y: 5.8,
    w: 11.2,
    h: 0.75,
    fontFace: FONT.body,
    fontSize: 15,
    color: COLOR.body,
    margin: 0,
  });

  addFooterCite(s, "GIC / UPV-EHU Hyperspectral Remote Sensing Scenes; Purdue AVIRIS Indian Pines. SSFT itself is cube→class; we adapt modules to labeled-center patches.");
  s.addNotes("GIC lists 224 bands then 200 after water-absorption drop; Purdue’s Site 3 release is a 220-band cube, and 220−20=200. Prefer ‘220-band product, commonly 200 bands.’ Pavia 610×340 is after discarding no-data strips from 610×610.");
}

// ======================================================================
// 06  Approach
// ======================================================================
function slide6() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "PROPOSED APPROACH", "06 / 09");

  s.addText("SSFT §3, then one Wave-path change.", {
    x: MARGIN_X,
    y: 0.88,
    w: 11.6,
    h: 0.5,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  // Pipeline boxes
  const boxes = [
    { t: "Input", d: "16×16×C\nλ in nm\nband_mask" },
    { t: "Spectral", d: "pool s=8\nφ: scalar→64\n1 MHSA, 4 heads\nmean over C" },
    { t: "Spatial", d: "tied 1×1\npool s=8\n3×3+BN+GELU" },
    { t: "Fusion", d: "spatial Q\nspectral K/V\n4-head CA" },
    { t: "Head", d: "adaptive pool\n2-layer MLP\nK logits" },
  ];
  boxes.forEach((b, i) => {
    const x = MARGIN_X + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y: 1.55,
      w: 2.18,
      h: 2.15,
      fill: { color: i === 0 || i === 4 ? COLOR.beige : COLOR.white },
      line: { color: COLOR.hairline, width: 0.75 },
    });
    s.addText(b.t, {
      x: x + 0.1,
      y: 1.65,
      w: 1.98,
      h: 0.38,
      fontFace: FONT.head,
      fontSize: 15,
      color: COLOR.forest,
      margin: 0,
    });
    s.addText(b.d, {
      x: x + 0.1,
      y: 2.08,
      w: 1.98,
      h: 1.45,
      fontFace: FONT.body,
      fontSize: 13,
      color: COLOR.body,
      margin: 0,
    });
    if (i < boxes.length - 1) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 2.17,
        y: 2.52,
        w: 0.16,
        h: 0.06,
        fill: { color: COLOR.terracotta },
      });
    }
  });

  // Two columns: copy / change
  card(s, MARGIN_X, 3.9, 5.7, 2.55, COLOR.white);
  s.addText("COPY FROM SSFT §3.6", {
    x: 1.1,
    y: 4.05,
    w: 5.2,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.forest,
    charSpacing: 1.6,
    margin: 0,
  });
  s.addText("D=64  ·  s=8  ·  4 heads  ·  one block each  ·  AdamW 4e-4, wd 1e-2, batch 8, ≤50 epochs, early stop. No PCA. Aux heads off for the first train (λ_aux=0.05 later).", {
    x: 1.1,
    y: 4.42,
    w: 5.2,
    h: 1.75,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  card(s, 7.05, 3.9, 5.4, 2.55, COLOR.white);
  s.addText("CHANGE (LESSViT IDEAS)", {
    x: 7.3,
    y: 4.05,
    w: 5.0,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.terracotta,
    charSpacing: 1.6,
    margin: 0,
  });
  s.addText("Index table P → sinusoidal PE(λ/1000). Spatial Conv2d(C,D,1) → tied Conv2d(1,D,1) + masked mean. API: forward(cube, wavelengths, band_mask). C is never a module dimension.", {
    x: 7.3,
    y: 4.42,
    w: 4.95,
    h: 1.75,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  s.addNotes("Paper param count 516k; we will print ours and not inflate layers. Spatial Q / spectral KV is SSFT eq. (4).");
}

// ======================================================================
// 07  Evaluation
// ======================================================================
function slide7() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "EVALUATION PLAN", "07 / 09");

  s.addText("Train once on a frozen band subset. Test four ways.", {
    x: MARGIN_X,
    y: 0.88,
    w: 11.6,
    h: 0.5,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  const tests = [
    { t: "In-distribution", d: "Same ID bands used in training." },
    { t: "Complementary", d: "Leftover / swapped wavelengths, similar C (LESSViT-style split at 1000 nm)." },
    { t: "~50% drop", d: "Random subset of ID bands, 3 seeds. Wave path shrinks C; ViT zero-fills." },
    { t: "Full-C (opt.)", d: "All 200 / 103 bands at test — channel expansion." },
  ];
  tests.forEach((t, i) => {
    const x = MARGIN_X + (i % 4) * 2.95;
    card(s, x, 1.5, 2.8, 1.7, COLOR.white);
    s.addText(String(i + 1).padStart(2, "0"), {
      x: x + 0.15,
      y: 1.6,
      w: 0.7,
      h: 0.28,
      fontFace: FONT.mono,
      fontSize: 12,
      color: COLOR.terracotta,
      margin: 0,
    });
    s.addText(t.t, {
      x: x + 0.15,
      y: 1.92,
      w: 2.5,
      h: 0.4,
      fontFace: FONT.head,
      fontSize: 14,
      color: COLOR.ink,
      margin: 0,
    });
    s.addText(t.d, {
      x: x + 0.15,
      y: 2.35,
      w: 2.5,
      h: 0.7,
      fontFace: FONT.body,
      fontSize: 12,
      color: COLOR.body,
      margin: 0,
    });
  });

  // Models + metrics
  card(s, MARGIN_X, 3.4, 5.7, 2.85, COLOR.white);
  s.addText("THREE MODELS", {
    x: 1.1,
    y: 3.55,
    w: 5.2,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.forest,
    charSpacing: 1.8,
    margin: 0,
  });
  s.addText("Vanilla ViT — bands collapsed in a fixed-C first linear layer.\nSSFT-index — learned P on packed 0…C−1.\nSSFT-Wave — λ-PE + tied stem.\nSame optimizer, same spatial split, 3 seeds. No retune per test config.", {
    x: 1.1,
    y: 3.95,
    w: 5.2,
    h: 2.05,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  card(s, 7.05, 3.4, 5.4, 2.85, COLOR.white);
  s.addText("METRICS", {
    x: 7.3,
    y: 3.55,
    w: 5.0,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLOR.terracotta,
    charSpacing: 1.8,
    margin: 0,
  });
  s.addText("OA, average accuracy (AA), and macro-F1. Headline: relative drop (ID − OOD) / ID on AA and macro-F1. OA alone is invalid on Indian Pines. We will not claim SOTA.", {
    x: 7.3,
    y: 3.95,
    w: 4.95,
    h: 2.05,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLOR.body,
    margin: 0,
  });

  addFooterCite(s, "Relative-drop protocol from LESSViT §3.1. Random ~50% drop is our class operationalization. SSFT itself has no band-drop protocol.");
  s.addNotes("Ablations later: fused vs spectral-only vs spatial-only (SSFT Tab. 4); index vs λ-PE.");
}

// ======================================================================
// 08  Timeline
// ======================================================================
function slide8() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "PROJECT TIMELINE", "08 / 09");

  s.addText("Eight sprints. One ticket in progress. No mystery layers.", {
    x: MARGIN_X,
    y: 0.88,
    w: 11.6,
    h: 0.5,
    fontFace: FONT.head,
    fontSize: 26,
    color: COLOR.ink,
    margin: 0,
  });

  const phases = [
    {
      k: "A  BUILD",
      when: "Now",
      rows: [
        ["1", "Spike: attention + λ-PE on a fake cube", "Done"],
        ["2", "SSFT-index dummy forward + param count", "Ready"],
        ["3", "Wave API: tied 1×1, C=32 then C=20", "Next"],
      ],
    },
    {
      k: "B  DATA + LOOP",
      when: "After 3",
      rows: [
        ["4", "Indian Pines + spatial split + nm table", ""],
        ["5", "Tiny overfit (1–8 patches)", ""],
        ["6", "Vanilla ViT baseline, fixed C", ""],
      ],
    },
    {
      k: "C  EXPERIMENT",
      when: "Close",
      rows: [
        ["7", "Band-drop table, 3 seeds, OA/AA/F1", ""],
        ["8", "Two ablations + writeup", ""],
      ],
    },
  ];

  phases.forEach((p, i) => {
    const x = MARGIN_X + i * 3.95;
    s.addText(p.k, {
      x,
      y: 1.5,
      w: 3.7,
      h: 0.28,
      fontFace: FONT.body,
      fontSize: 12,
      color: COLOR.terracotta,
      charSpacing: 1.6,
      margin: 0,
    });
    s.addText(p.when, {
      x,
      y: 1.78,
      w: 3.7,
      h: 0.28,
      fontFace: FONT.head,
      fontSize: 16,
      color: COLOR.ink,
      margin: 0,
    });
    p.rows.forEach((r, j) => {
      const y = 2.22 + j * 1.22;
      card(s, x, y, 3.75, 1.12, COLOR.white);
      s.addText(r[0], {
        x: x + 0.12,
        y: y + 0.14,
        w: 0.38,
        h: 0.8,
        fontFace: FONT.mono,
        fontSize: 16,
        color: COLOR.forest,
        valign: "middle",
        margin: 0,
      });
      s.addText(r[1], {
        x: x + 0.52,
        y: y + 0.12,
        w: 3.08,
        h: r[2] ? 0.55 : 0.85,
        fontFace: FONT.body,
        fontSize: 13,
        color: COLOR.body,
        valign: "middle",
        margin: 0,
      });
      if (r[2]) {
        s.addText(r[2], {
          x: x + 0.52,
          y: y + 0.7,
          w: 3.08,
          h: 0.3,
          fontFace: FONT.body,
          fontSize: 12,
          bold: true,
          color: COLOR.terracotta,
          margin: 0,
        });
      }
    });
  });

  s.addNotes(
    "WIP=1. Follow-on planning after sprint 3, before the first download, to freeze the ID band JSON. Course Lab 1 / NPU is a different assignment."
  );
}

// ======================================================================
// 09  Close
// ======================================================================
function slide9() {
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  addEyebrow(s, "WHAT SUCCESS LOOKS LIKE", "09 / 09", true);

  s.addText("A ~0.5M dual-path model that degrades less than a vanilla ViT when bands are dropped or swapped.", {
    x: MARGIN_X,
    y: 1.35,
    w: 11.5,
    h: 1.5,
    fontFace: FONT.head,
    fontSize: 28,
    color: COLOR.titleOnDark,
    margin: 0,
  });

  const ins = [
    { t: "Will report", d: "Honest param count, spatial-split OA/AA/macro-F1, relative drop, two ablations." },
    { t: "Will not", d: "SOTA on Indian Pines. Foundation-model fine-tunes. HSI-Benchmark / SpectralEarth. Lab 1 NPU work. SSFTT 2022." },
  ];
  ins.forEach((it, i) => {
    const x = MARGIN_X + i * 5.9;
    s.addText(it.t, {
      x,
      y: 3.15,
      w: 5.5,
      h: 0.35,
      fontFace: FONT.body,
      fontSize: 13,
      color: COLOR.terracottaSoft,
      charSpacing: 1.8,
      margin: 0,
    });
    s.addText(it.d, {
      x,
      y: 3.55,
      w: 5.5,
      h: 1.3,
      fontFace: FONT.body,
      fontSize: 16,
      color: COLOR.titleOnDark,
      margin: 0,
    });
  });

  s.addShape(pres.shapes.LINE, {
    x: MARGIN_X,
    y: 5.2,
    w: W - 2 * MARGIN_X,
    h: 0,
    line: { color: COLOR.hairlineDark, width: 0.75 },
  });
  s.addText("Map  ·  github.com/scottn66/255-final-project  ·  bulletins/ 1.xx plan, 2.xx board\nNext pull  ·  Sprint 2 — SSFT-index on a dummy cube (no data download)", {
    x: MARGIN_X,
    y: 5.45,
    w: 11.6,
    h: 0.95,
    fontFace: FONT.body,
    fontSize: 15,
    color: COLOR.subtitleOnDark,
    margin: 0,
  });

  s.addNotes("Close on the success criterion. Invite questions on split leakage and why we refuse OA-only tables.");
}

slide1();
slide2();
slide3();
slide4();
slide5();
slide6();
slide7();
slide8();
slide9();

pres.writeFile({ fileName: "/Users/scottnelson/Desktop/255/ssft-wave/docs/SSFT-Wave_class_intro.pptx" })
  .then(() => console.log("Wrote SSFT-Wave_class_intro.pptx"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
