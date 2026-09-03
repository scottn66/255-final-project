---
id: SSFT-0
status: done
sprint: 1
owner: grok (session 1, 2026-08-25)
blocked_by: []
---

# SSFT-0 — Spike: attention + λ-PE + variable-C stub

## Goal

Prove we can do from-scratch attention and a wavelength-conditioned forward on a fake cube. No training. No Indian Pines.

## Done when (met)

- [x] `src/attention.py` MHSA + cross-attention from scratch
- [x] softmax rows ≈ 1; 1-pixel spectrum of length C
- [x] `src/pe.py` sinusoidal 1D PE on λ in nm
- [x] dummy `2×32×16×16` then `2×20×16×16` without rebuilding the module
- [x] `pytest` green (17 passed)

## Notes

Landed: `src/attention.py`, `src/pe.py`, `src/spectral_stub.py`, tests. PLAN.md written. Full SSFT is SSFT-2, not this ticket.
