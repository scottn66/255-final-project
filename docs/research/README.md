# Research notes (revised claim)

**Status:** pending Jeong / Mamta approval (Group 10 email, Week 5 NPU smoke-test ask).  
**Course:** DATA 255 · Fall 2026 · Group 10 (Scott Nelson, Surya Reddy, Tim Wada)

This folder is the **current** proposal framing. The earlier Indian Pines / band-drop plan remains in [`PLAN.md`](../../PLAN.md) and [`bulletins/`](../../bulletins/) until the instructor reply lands.

## One-sentence claim

Can a hyperspectral model that takes **physical wavelength as a runtime input** (instead of hard-coding one camera’s band layout) keep that flexibility after **INT8 quantization and ARIES compilation** — so **one quantized binary can serve more than one camera**?

Novelty is **not** inventing wavelength-aware PE or channel-agnostic embeddings (those appear in recent work such as LESSViT). The question is whether **sensor-agnosticism survives INT8 on an edge accelerator**.

## Problem

Plastic HSI sorting has to run at conveyor / line speed on an edge accelerator. Compilation freezes input shape, including band count. A failed band or a different sensor at a new site then means retrain + recompile.

## Gap

Sensor-agnostic models are usually validated in float32 on GPUs. Edge-deployment papers usually quantize a model fixed to one sensor from the start. We want to test whether sensor-agnosticism survives INT8 on Mobilint ARIES.

## Dataset

- **Primary:** SpectralWaste (IROS 2024) — industrial plastics sorting facility, published baselines.
- **Optional / to verify:** MWIR-4-Plastic, only if it fits after a Week 5 compile smoke test.

We are **not** leading with Indian Pines / Pavia as the main industrial claim (those remain in the older plan docs as prior framing).

## Approach

Small from-scratch spectral–spatial model (SSFT §3 spirit + LESSViT ideas: tied / channel-agnostic per-band embedding, wavelength-conditioned PE, factorized spectral and spatial paths fused by cross-attention), about **0.5–5M** parameters.

Train float32 → quantize INT8 → compile for **ARIES**, with the **wavelength vector declared as an input tensor**.

Existing spike code (`src/attention.py`, `src/pe.py`, `src/spectral_stub.py`) is architecture scaffolding only — not a trained industrial model.

## Evaluation

- Wavelength-shuffle / band-subset tests **before and after** quantization on the same model.
- Latency vs line speed on hardware.
- Band-index-encoding control that should show no wavelength sensitivity by construction.
- Baselines: SpectralWaste published results and a fixed-channel model compiled for one camera.

## Timeline (Weeks 5–16)

| Weeks | Focus |
|-------|--------|
| 5 | ARIES compile smoke test (runtime wavelength vector + many-band input) |
| 6–8 | Data pipeline + float baseline |
| 9 | Float measurement |
| 10–11 | Quantize and re-measure (**core result**) |
| 12 | Calibration-composition ablation |
| 13 | On-hardware latency |
| 14 | Controls / ablations |
| 15–16 | Write-up and presentation |

## Instructor questions (open)

1. Week 5 NPU / ARIES compile access for a smoke test.
2. Whether one industrial dataset + hardware/quantization study (instead of pretraining / multi-benchmark breadth) matches final-project expectations.

## Links

- Repo: https://github.com/scottn66/255-final-project
- Prior class intro deck (Sep 9): [`../SSFT-Wave_class_intro.pptx`](../SSFT-Wave_class_intro.pptx)
- Prior full plan: [`../../PLAN.md`](../../PLAN.md)
