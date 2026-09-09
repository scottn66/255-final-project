# SSFT-Wave

SJSU DATA 255 final project ([github.com/scottn66/255-final-project](https://github.com/scottn66/255-final-project)). Tiny dual-path spectral–spatial fusion transformer ([SSFT](https://arxiv.org/abs/2604.15828) §3) with wavelength-aware PE ([LESSViT](https://arxiv.org/abs/2605.18541) idea).

This is **not** SSFTT (TGRS 2022). No official SSFT code exists.

## Start here

| # | Open | What it is |
|---|------|------------|
| **[bulletins/](bulletins/README.md)** | Starter notes | Hierarchical map: **1.xx plan** · **2.xx board** |
| [1.00 Plan](bulletins/1.00-plan.md) | Goal, papers, success | Full spec: [PLAN.md](PLAN.md) |
| [2.00 Board](bulletins/2.00-board.md) | How we run sprints | Checkboard: [BOARD.md](BOARD.md) |
| **[2.20 Sprint 2](bulletins/2.20-sprint-2.md)** | **Pull this now** | Ticket: [SSFT-2](tickets/SSFT-2.md) |

Teammates and other AIs: read [AGENTS.md](AGENTS.md), then the bulletin index, then one ticket.

## How to join a sprint

1. Open [bulletins/README.md](bulletins/README.md) and [BOARD.md](BOARD.md).
2. Pull the `ready` card (right now: **SSFT-2** / [2.20](bulletins/2.20-sprint-2.md)).
3. Put your name on **Owner**, set status `in_progress`.
4. Paste the ticket’s pickup prompt into your AI, or implement it yourself.
5. Stop when the ticket’s acceptance criteria are met. `pytest -q`. Update the board.

WIP limit: **one implementation ticket** at a time.

## Class intro slides

Nine-slide briefing for class and professor (problem, gap, data, method, evaluation, timeline): [`docs/SSFT-Wave_class_intro.pptx`](docs/SSFT-Wave_class_intro.pptx).

## Spike (already done)

From-scratch attention + sinusoidal λ-PE + a C-agnostic spectral stub. See [2.10](bulletins/2.10-sprint-1.md).

```bash
source .venv/bin/activate   # python 3.12, torch + pytest + numpy
pytest -q
```
