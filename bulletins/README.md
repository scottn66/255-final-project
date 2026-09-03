# Bulletin index

Starter notes for **SSFT-Wave** (DATA 255 final project).  
Spec: [`../PLAN.md`](../PLAN.md) · Checkboard: [`../BOARD.md`](../BOARD.md) · Tickets: [`../tickets/`](../tickets/) · AI/human rules: [`../AGENTS.md`](../AGENTS.md)

**How to read this:** `1.xx` is the **plan** (what we build). `2.xx` is the **board** (how we execute). Open the parent (`1.00` / `2.00`) first, then a child. Full detail stays in PLAN.md / BOARD.md — these notes are the map.

| Now | Next |
|-----|------|
| Spike **done** ([2.10](2.10-sprint-1.md)) | **[2.20 Sprint 2 — SSFT-index](2.20-sprint-2.md)** — start now |
| Planning session 1 **done** (PLAN.md) | Follow-on planning after 2.20, before any data download |

```
1.xx PLAN                         2.xx BOARD
├─ 1.00  Plan hub                 ├─ 2.00  Board hub
├─ 1.10  Locked defaults          ├─ 2.10  Sprint 1 spike     done
├─ 1.20  Copy vs change           ├─ 2.20  Sprint 2 SSFT-index  ← current
├─ 1.30  Experiment               └─ 2.30  Sprints 3–8        later
└─ 1.40  Risks + out of scope
```

---

## 1.xx Plan

| ID | Note | Spec |
|----|------|------|
| [1.00](1.00-plan.md) | Plan hub — goal, papers, success criterion | [PLAN.md](../PLAN.md) |
| [1.10](1.10-defaults.md) | Locked defaults (patch, split, D, s) | PLAN locked table |
| [1.20](1.20-copy-vs-change.md) | What we copy from SSFT vs steal from LESSViT | PLAN A.3 |
| [1.30](1.30-experiment.md) | Band-drop protocol, metrics, splits | PLAN Phase 7 |
| [1.40](1.40-risks.md) | Three risks + will-not-do | PLAN A.5 / out of scope |

## 2.xx Board

| ID | Note | Board / ticket |
|----|------|----------------|
| [2.00](2.00-board.md) | Board hub — WIP, DoD, how to pull a card | [BOARD.md](../BOARD.md) |
| [2.10](2.10-sprint-1.md) | Sprint 1 — spike (done) | [SSFT-0](../tickets/SSFT-0.md), [SSFT-1](../tickets/SSFT-1.md) |
| [2.20](2.20-sprint-2.md) | Sprint 2 — SSFT-index dummy forward (**pull this**) | [SSFT-2](../tickets/SSFT-2.md) |
| [2.30](2.30-later-sprints.md) | Sprints 3–8 (Wave API → data → experiment → writeup) | [SSFT-3](../tickets/SSFT-3.md) … [SSFT-8](../tickets/SSFT-8.md) |

---

## Room for a later planning session

Do **not** reopen architecture before 2.20 lands. A short planning session after Sprint 2 is the right time to freeze the Indian Pines wavelength table and the ID band subset ([1.30](1.30-experiment.md)). Until then, execute [2.20](2.20-sprint-2.md).
