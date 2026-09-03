---
id: SSFT-6
status: backlog
sprint: 6
owner: unassigned
blocked_by: [SSFT-5]
---

# SSFT-6 — Vanilla ViT baseline (fixed C)

## Goal

The model that should *break* under band drop/swap: first layer collapses bands, `C` baked in.

## Pickup prompt

```
SSFT-Wave ticket SSFT-6. Repo: Desktop/255/ssft-wave. PLAN.md A.3 vanilla ViT.

Do this only:
1. src/vit.py: Linear(C·p·p, D) or Conv2d(C, D, kernel=p, stride=p) with fixed C. D=64, 4 heads, comparable param budget (print vs SSFT).
2. Same patch size / split / optimizer as SSFT-Wave.
3. Overfit smoke + ID eval on the spatial val split.
4. At test, C cannot change: dropped bands are zero-filled to C_train.

No band-drop table yet (that is SSFT-7). pytest green. Mark SSFT-6 done on BOARD.md.
```

## Acceptance criteria

- [ ] Fixed-C first layer
- [ ] Param count printed next to SSFT
- [ ] Zero-fill protocol documented in the module docstring
