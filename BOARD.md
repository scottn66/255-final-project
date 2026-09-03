# SSFT-Wave board

**Checkboard** (status of record). Starter notes: [`bulletins/README.md`](bulletins/README.md) — [1.xx plan](bulletins/1.00-plan.md) · [2.xx board](bulletins/2.00-board.md). Spec: [`PLAN.md`](PLAN.md). Rules: [`AGENTS.md`](AGENTS.md). One ticket = one sprint.

Last updated: **2026-09-03** · Active sprint: **[2.20](bulletins/2.20-sprint-2.md) / SSFT-2** · WIP limit: **1 implementation ticket**

---

## Bulletin (read this first)

| | |
|---|---|
| **Project** | SSFT-Wave — DATA 255 group research project |
| **Goal** | A ~0.5M dual-path spectral–spatial transformer that degrades *less* than a vanilla ViT when bands are dropped or swapped |
| **Now** | Session 1 spike is **done** (17 tests). Next is **[2.20 SSFT-index](bulletins/2.20-sprint-2.md)** (no data download, no training) |
| **Repo** | [github.com/scottn66/255-final-project](https://github.com/scottn66/255-final-project) |
| **Not this project** | Lab 1 / NPU / ONNX / ImageNet-20 · SSFTT 2022 · foundation-model fine-tunes · HSI-Benchmark / SpectralEarth |

Paste into Slack/Canvas as-is. Status lives in the checkboard below; change the checkboard when you start or finish a ticket.

---

## Checkboard

Move a card by editing **Status** and **Owner**. Do not start work that is still `backlog` if a `ready` / `in_progress` card exists.

| ID | Ticket | Sprint | Status | Owner | Blocked by |
|----|--------|--------|--------|-------|------------|
| [SSFT-0](tickets/SSFT-0.md) | Spike: from-scratch attention + λ-PE + variable-C stub | 1 | **done** | grok (session 1) | — |
| [SSFT-1](tickets/SSFT-1.md) | Attention primitives complete (mask, W_O, residual+FFN) | 1 | **done** | grok (session 1, landed in spike) | SSFT-0 |
| [SSFT-2](tickets/SSFT-2.md) | SSFT-index dummy forward + param count | **2 (current)** | **ready** | *unassigned* | SSFT-1 |
| [SSFT-3](tickets/SSFT-3.md) | λ-PE + tied spatial 1×1 + `band_mask` API | 3 | backlog | | SSFT-2 |
| [SSFT-4](tickets/SSFT-4.md) | Indian Pines loader, wavelengths, spatial block split | 4 | backlog | | SSFT-3 |
| [SSFT-5](tickets/SSFT-5.md) | Tiny overfit (1–8 patches) | 5 | backlog | | SSFT-4, SSFT-3 |
| [SSFT-6](tickets/SSFT-6.md) | Vanilla ViT baseline (fixed C) | 6 | backlog | | SSFT-5 |
| [SSFT-7](tickets/SSFT-7.md) | Band-drop table (the experiment) | 7 | backlog | | SSFT-5, SSFT-6 |
| [SSFT-8](tickets/SSFT-8.md) | Two ablations + writeup | 8 | backlog | | SSFT-7 |

Status values: `backlog` · `ready` · `in_progress` · `review` · `done` · `blocked`

### Lanes (same cards)

```
DONE          SSFT-0  SSFT-1
READY         SSFT-2          ← pull this next
IN PROGRESS   (empty)
REVIEW        (empty)
BACKLOG       SSFT-3  SSFT-4  SSFT-5  SSFT-6  SSFT-7  SSFT-8
BLOCKED       (none)
ICEBOX        EuroSAT-MS · SSRoPE · aux heads · extra scenes
WON'T DO      Lab 1/NPU · SatMAE/Prithvi/SpectralGPT · HSI-Benchmark (~240GB) · SpectralEarth · SOTA claims · SSFTT 2022
```

---

## How we run agile here

This is a **phase-kanban**, not Scrum-with-story-points theatre. PLAN.md phases *are* the backlog.

1. **One ticket in progress.** An AI or a person pulls SSFT-N, sets `in_progress` + their name, then implements *only* that ticket.
2. **Sprint = one ticket**, usually one coding session. Do not “also quickly do SSFT-3” in the same session.
3. **Definition of done** (all of these):
   - Ticket acceptance criteria checked
   - `pytest -q` green
   - No banned attention APIs (`nn.MultiheadAttention`, timm, HF ViT blocks)
   - No Indian Pines / multi-GB download unless the ticket says so
   - `BOARD.md` status → `done` (or `review`)
   - 5-line note in [Sprint log](#sprint-log)
4. **Review:** another human *or* another AI reads the diff against PLAN.md §3 / locked defaults. Reviewer does not expand scope.
5. **Blocked:** set status `blocked` and write the blocker in the ticket. Do not silently skip the spatial-split rule to unblock data work.
6. **Changing locked defaults** (patch size, split, D, s, PE): group decision, then edit PLAN.md first, then the ticket.

### Roles (fill names)

| Role | Who | Notes |
|------|-----|--------|
| Spec owner | | PLAN.md locked defaults. Does not invent extra layers. |
| Board keeper | | Moves cards; keeps this bulletin honest. |
| Implementer (current) | *unassigned* | Pulls SSFT-2. |
| Reviewer | | Different person or different AI than the implementer. |
| Writeup | | SSFT-8; can start an outline anytime, not the experiment. |

---

## Standup (copy/paste)

```
Date:
Yesterday: (ticket ID + what landed)
Today:     (ticket ID)
Blocked:   (none / …)
Tests:     pytest N passed / not run
```

---

## Sprint log

### Sprint 1 — 2026-08-25 — SSFT-0 + SSFT-1 — done

- PLAN.md written from SSFT §3 and LESSViT §2 (not SSFTT 2022).
- Spike: `src/attention.py`, `src/pe.py`, `src/spectral_stub.py`.
- 17 pytest passed. Same module ran C=32 then C=20. No training, no data download.
- Full SSFT not built (that is SSFT-2).

### Sprint 2 — not started — SSFT-2

- Pickup prompt: [`tickets/SSFT-2.md`](tickets/SSFT-2.md)

---

## Icebox (later, not scheduled)

- 13-band EuroSAT-MS / Sentinel-2 (not RGB EuroSAT)
- 1D RoPE / SSRoPE (LESSViT Appendix B)
- Aux heads `λ_aux=0.05`
- Pavia Centre / Salinas as extra scenes
