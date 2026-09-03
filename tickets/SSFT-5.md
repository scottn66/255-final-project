---
id: SSFT-5
status: backlog
sprint: 5
owner: unassigned
blocked_by: [SSFT-4, SSFT-3]
---

# SSFT-5 — Tiny overfit

## Goal

Prove the training loop. 1–8 patches, CE → ~0, train OA = 1.0. Not a generalization result.

## Pickup prompt

```
SSFT-Wave ticket SSFT-5. Repo: Desktop/255/ssft-wave.

Do this only:
1. src/train.py (or scripts/overfit.py): AdamW lr 4e-4, wd 1e-2, batch ≤8, SSFT-Wave.
2. Overfit 1–8 real (or frozen dummy-labeled) patches until train OA = 1.0 / CE ~0.
3. No val split required. No band-drop table. No full 50-epoch sweep.

pytest still green. Mark SSFT-5 done on BOARD.md.
```

## Acceptance criteria

- [ ] Loop runs on one device
- [ ] Train OA hits 1.0 on the tiny set
- [ ] Optimizer matches PLAN.md (AdamW 4e-4, wd 1e-2)
