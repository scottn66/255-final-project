# Class introduction deck

[`SSFT-Wave_class_intro.pptx`](SSFT-Wave_class_intro.pptx) — 9 slides for DATA 255 (class + professor).

| Slide | Required outline item |
|-------|------------------------|
| 01 | Title |
| 02 | Problem statement |
| 03–04 | Research gap and novelty |
| 05 | Dataset |
| 06 | Proposed approach |
| 07 | Evaluation plan |
| 08 | Project timeline |
| 09 | Success criterion / will-not |

Regenerate:

```bash
cd docs && npm install && node class_intro.js
```

Sources: SSFT arXiv:2604.15828 §3; LESSViT arXiv:2605.18541 §2 / §3.1; GIC Indian Pines / Pavia U; Nalepa et al. arXiv:1811.03707 (spatial leakage); project `PLAN.md` and `bulletins/`.

A background research pass flagged: Indian Pines remainder 200 matches a **220**-band Purdue cube minus 20 water bands (GIC’s “224” is inconsistent); Pavia **610×340** is after discarding no-data strips; LESSViT PE is **SSRoPE**, so our sinusoid is a simpler stand-in; random 50% drop is a class operationalization, not a named LESSViT test. Those corrections are in this deck.
