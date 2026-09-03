# Working with this repo (humans and other AIs)

You are joining **SSFT-Wave**, a SJSU DATA 255 group research project. Other people and other models (Grok, Claude, Codex, Cursor, …) will also edit this tree. Follow this file so you do not fork the architecture or skip the board.

## Start every session here

1. Read [`bulletins/README.md`](bulletins/README.md) — `1.xx` is plan, `2.xx` is board.
2. Read [`BOARD.md`](BOARD.md) — what is `in_progress` / `ready`.
3. Read [`PLAN.md`](PLAN.md) for full spec if you are changing shapes or layers. **Do not invent layers.**
4. Open **one** ticket under [`tickets/`](tickets/) (and its `2.xx` bulletin). That ticket is the whole session.
5. Set the ticket and BOARD status to `in_progress` and put your name (or model + date) in **Owner** *before* writing code.
6. Implement only that ticket. Stop when its acceptance criteria are met.

If BOARD says a ticket is `in_progress` for someone else, **do not take it**. Pick nothing, or ask.

## Locked defaults (do not “improve”)

- Labeled-center **16×16** patches, **spatial block split only**, **D=64**, **s=8**, **4 heads**
- 1 spectral MHSA + 1 spatial conv block + 1 cross-attn fusion (spatial Q, spectral K/V)
- Attention **from scratch** (`nn.Linear`, `softmax`). Banned: `nn.MultiheadAttention`, `F.multi_head_attention_forward`, timm, HuggingFace ViT blocks
- Wave path: `forward(cube, wavelengths, band_mask)` — never assume a fixed C
- No PCA. Aux heads off until a ticket says otherwise
- Prefer a correct spatial split over a clever attention variant

This is **SSFT** (Musiat et al., arXiv:2604.15828), **not** SSFTT (TGRS 2022, `zgr6010/HSI_SSFTT`). There is no official SSFT code.

## Out of scope (stop if asked)

DATA 255 Lab 1 (ImageNet-20, ONNX, MXQ, NPU) · foundation-model fine-tunes (SatMAE, Prithvi, SpectralGPT, AnySat, RemoteCLIP, LESSViT weights) · HSI-Benchmark (~240 GB) · SpectralEarth (~TB) · claiming SOTA on Indian Pines · wildfire detection · LESS Attention / HyperMAE / 96-hour pretrain

## Definition of done

- Ticket acceptance criteria all checked
- `pytest -q` green from repo root (`.venv` is Python 3.12, `torch` / `pytest` / `numpy`)
- BOARD.md status updated; 5-line note in the sprint log
- No extra files the ticket did not ask for (especially notebooks)

## Handoff to the next AI or teammate

When you stop, leave:

1. BOARD.md current
2. Ticket file: what landed, what did not, exact next command
3. Tests passing (or the failing command pasted in the ticket)

Do not leave “almost done” as `done`.
