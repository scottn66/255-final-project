---
id: SSFT-7
status: backlog
sprint: 7
owner: unassigned
blocked_by: [SSFT-5, SSFT-6]
---

# SSFT-7 — Band-drop table (the experiment)

## Goal

Train **one** frozen in-distribution band subset. Test ID / complementary / random ~50% drop / optional full-C. Compare vanilla ViT vs SSFT-index vs SSFT-Wave.

**Metrics:** OA **and** AA **and** macro-F1. Primary = relative drop `(ID − OOD) / ID` on AA and macro-F1. **Do not claim SOTA.**

## Pickup prompt

```
SSFT-Wave ticket SSFT-7. Repo: Desktop/255/ssft-wave. PLAN.md Phase 7.

Do this only:
1. Freeze one ID subset (LESSViT-style, split at 1000 nm; Pro-VNIR mix). Save configs/ip_id_bands.json (indices AND nm).
2. Train each of: vanilla ViT, SSFT-index, SSFT-Wave. Do not retune per test config.
3. Evaluate: ID, complementary/swapped, random ~50% drop (3 seeds), optional full-C.
4. Table: OA, AA, macro-F1, relative drop. Write one sentence: did Wave drop less?
5. Spatial split only. No SOTA language.

Mark SSFT-7 done on BOARD.md when the Indian Pines table exists. Pavia U may be a second pass.
```

## Acceptance criteria

- [ ] ID subset frozen on disk with wavelengths
- [ ] Three-model table, 3 seeds
- [ ] OA is not the only metric
- [ ] No SOTA claim
