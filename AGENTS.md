# Working in this repo

DATA 255 Group 1: Scott Nelson, Surya Reddy, Tim Wada, Prakhar, Yash.

The project description is the paragraph at the top of [`README.md`](README.md). Owners, done-when, and dates are in [`docs/PLAN.md`](docs/PLAN.md). Take one work package. Do not invent a second project.

## Layout

- `README.md` — start here
- `docs/PLAN.md` — work packages and timeline
- `notebooks/` — look at the data, then a pipeline smoke test
- `scripts/get_data.sh` — download the zip and check md5
- `src/sw/` — dataset, metric, model
- `tests/` — `pytest -q`

## Rules

- Never commit data, zips, tiffs, or checkpoints. The loader reads `SW_ROOT`.
- One branch per work package. Small pull requests. One teammate reviews.
- Use the official split only. No pretrained weights.
- Attention stays from scratch (`nn.Linear` and softmax). Do not drop in `nn.MultiheadAttention` or a library ViT block.
