---
id: SSFT-1
status: done
sprint: 1
owner: grok (session 1, 2026-08-25)
blocked_by: [SSFT-0]
---

# SSFT-1 — Attention primitives complete

## Goal

Residual+FFN, key mask, `W_O`, ban-list comment. Planned as “same session as 0 if time.”

## Done when (met)

- [x] Residual self-attn and cross-attn blocks (Pre-LN)
- [x] Masked keys get ~0 attention mass
- [x] `d_h` not dividing D raises
- [x] No `nn.MultiheadAttention(` in `attention.py`

Shipped inside the spike; no leftover work.
