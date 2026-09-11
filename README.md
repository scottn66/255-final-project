# DATA 255 final project (Group 10)

**Current claim (pending Jeong / Mamta):** can a hyperspectral model that takes physical wavelength as a runtime input keep that flexibility after **INT8 quantization and ARIES compilation**, so one quantized binary can serve more than one camera?

→ Start at **[`docs/research/`](docs/research/README.md)** (SpectralWaste plastics sorting + edge quantization).

Group: Scott Nelson, Surya Reddy, Tim Wada · SJSU DATA 255 Fall 2026.

## What changed

The Sep 9 class intro framed **SSFT-Wave** on academic HSI (Indian Pines band-drop vs a fixed-C ViT). Instructor feedback pushed us to a **narrower, hardware-facing claim**: industrial plastic HSI, and whether sensor-agnosticism survives INT8 on Mobilint ARIES. Details live in [`docs/research/`](docs/research/README.md).

## Still in the repo (prior framing)

| Path | Role |
|------|------|
| [`docs/research/`](docs/research/README.md) | **Current** proposal notes |
| [`PLAN.md`](PLAN.md) / [`bulletins/`](bulletins/README.md) | Earlier SSFT-Wave / Indian Pines plan + sprint board |
| [`docs/SSFT-Wave_class_intro.pptx`](docs/SSFT-Wave_class_intro.pptx) | Sep 9 class intro deck (historical) |
| [`src/`](src/) | From-scratch attention / λ-PE spike (architecture scaffolding) |

Do **not** treat Indian Pines download or full training as the next step until the Week 5 compile smoke test and instructor reply.

## Spike tests

```bash
source .venv/bin/activate   # python 3.12, torch + pytest + numpy
pytest -q
```
