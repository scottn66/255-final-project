---
id: SSFT-4
status: backlog
sprint: 4
owner: unassigned
blocked_by: [SSFT-3]
---

# SSFT-4 — Indian Pines loader + spatial split + wavelength table

## Goal

Public Indian Pines only (~few MB `.mat`). Store **wavelengths in nm**, not just indices. **Spatial block split.** Discard patches whose 16×16 support crosses split boundaries. No random-pixel API.

## Pickup prompt

```
SSFT-Wave ticket SSFT-4. Repo: Desktop/255/ssft-wave. PLAN.md Phase 4 is the spec.

Do this only:
1. src/data.py: Indian Pines cube, labels, wavelengths_nm of length C.
2. Water-absorption drop on the 220-band product: [104–108], [150–163], 220 (1-indexed) → 200 bands. Document if linspace(400,2500,220) is used instead of Purdue calibration.
3. Spatial block split (G=5 default). 16×16 labeled-center patches. Overlap-discard.
4. tests/test_data.py on a fake 32×32 grid: train/test patches share no pixels.
5. Do not add a random-pixel split helper. Do not download HSI-Benchmark or SpectralEarth. Pavia U is a follow-up, not this ticket unless IP is already done and tiny.

pytest -q green. Mark SSFT-4 done on BOARD.md.
```

## Acceptance criteria

- [ ] `wavelengths_nm.shape == (C,)`
- [ ] Block split + overlap discard; test proves no shared pixels
- [ ] No random-pixel split API
- [ ] Rare-class warning logged if a class is missing from train
