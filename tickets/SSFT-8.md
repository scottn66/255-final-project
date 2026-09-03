---
id: SSFT-8
status: backlog
sprint: 8
owner: unassigned
blocked_by: [SSFT-7]
---

# SSFT-8 — Two ablations + writeup

## Goal

Pick **two** ablations, not more. Write the method + spatial-split warning + table + limitations.

Ablation menu:

1. Fused vs spectral-only vs spatial-only (SSFT Tab. 4: zero the other map before fusion)
2. Index PE vs λ-PE (already in the main table; write it up)
3. Tied embed vs fixed-C 1×1 (only if needed to explain a Wave vs index gap)

## Pickup prompt

```
SSFT-Wave ticket SSFT-8. Repo: Desktop/255/ssft-wave. PLAN.md Phase 8.

Do this only:
1. Run two ablations from the ticket menu. Do not add a third.
2. Short writeup: SSFT §3 + LESSViT PE/tied, spatial-split leakage warning, band-drop table, limitations (tiny model, two scenes, no foundation models).
3. Do not claim SOTA. Do not add wildfire detection.

Mark SSFT-8 done on BOARD.md.
```

## Acceptance criteria

- [ ] Exactly two ablations
- [ ] Writeup states random-pixel 99% OA is invalid
- [ ] Limitations section present
