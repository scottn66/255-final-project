---
id: SSFT-3
status: backlog
sprint: 3
owner: unassigned
blocked_by: [SSFT-2]
---

# SSFT-3 — λ-PE + tied spatial 1×1 + band_mask API

## Goal

Turn SSFT-index into SSFT-Wave: wavelength PE, channel-agnostic spatial stem, `C` free at test time.

## Pickup prompt

```
SSFT-Wave ticket SSFT-3. Repo: Desktop/255/ssft-wave. Read PLAN.md A.3 and BOARD.md.

Do this only:
1. pe_mode="wave": additive sinusoidal λ-PE from src/pe.py (nm / 1000 → microns).
2. Replace spatial Conv2d(C, D, 1) with tied Conv2d(1, D, 1) per band + masked mean over C.
3. forward(cube, wavelengths, band_mask) must run C=32 then C=20 on the same module.
4. Test: permute bands+wavelengths together ≈ invariant logits (eval, no dropout); permute bands without wavelengths is not.
5. No Indian Pines, no training.

pytest -q green. Mark SSFT-3 done on BOARD.md.
```

## Acceptance criteria

- [ ] Same module, `C=32` then `C=20`, no rebuild
- [ ] Tied spatial `1×1` (no `Conv2d(C, D, 1)` in the Wave path)
- [ ] Band+λ permutation invariance test
- [ ] Index-PE path still available for the later ablation
