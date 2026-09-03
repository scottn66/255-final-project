---
id: SSFT-2
status: ready
sprint: 2
owner: unassigned
blocked_by: [SSFT-1]
---

# SSFT-2 — SSFT-index dummy forward + param count

**This is the current sprint. Pull it next.** Set status to `in_progress` in this file **and** in `BOARD.md` before coding.

## Goal

Implement SSFT §3 as **index-PE** (learned `P ∈ R^{C×D}`), on a fake cube only. Print the real parameter count versus the paper’s 516k. Do **not** add mystery layers to chase 516k.

## Pickup prompt (paste into another AI)

```
Session 2 of SSFT-Wave (DATA 255). Repo: github.com/scottn66/255-final-project
Read bulletins/README.md, bulletins/2.20-sprint-2.md, PLAN.md, tickets/SSFT-2.md.

Do this only:
1. Finish src/ssft.py as SSFT-index: D=64, s=8, 1 spectral MHSA (4 heads, from src/attention.py), 1 spatial block (1×1 then 3×3+BN+GELU), 1 cross-attn fusion (spatial Q, spectral K/V, 4 heads), adaptive pool + 2-layer MLP. Aux heads off.
2. Dummy cube B×C×16×16 (e.g. 2×32×16×16) → logits (B, K=16). Print exact param count vs paper 516k; do not add mystery layers.
3. tests/test_ssft.py: output shape; residual paths exist; index PE is C-sized.
4. Keep forward(cube, wavelengths=None, band_mask=None) signature even if index mode ignores wavelengths.
5. No Indian Pines download, no training, no LESS Attention.

Do not use nn.MultiheadAttention. Do not touch Lab 1 / NPU / foundation weights.
When done: pytest -q green, mark SSFT-2 done on BOARD.md, append a 5-line sprint log.
```

## Acceptance criteria

- [ ] `src/ssft.py` exists; `pe_mode="index"` (or equivalent)
- [ ] Dummy `(B, C, 16, 16)` → `(B, K)` logits, K default 16
- [ ] Param count printed; honest number (PLAN.md expects ~0.16–0.23M at D=64, not 516k)
- [ ] `tests/test_ssft.py`: shapes; index PE table is C-sized
- [ ] Signature `forward(cube, wavelengths=None, band_mask=None)`
- [ ] Aux heads off
- [ ] `pytest -q` still green including old tests
- [ ] BOARD.md → `done` or `review`

## Out of this ticket

λ-PE wiring, tied spatial 1×1, Indian Pines, training, vanilla ViT, LESS Attention.
