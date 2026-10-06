# Working in this repo

DATA 255 Group 1 (SJSU): Scott Nelson, Surya Reddy, Tim Wada, Prakhar, Yash.

We train a transformer we write ourselves on SpectralWaste (Zenodo 10880544) to segment six waste classes. The score is mean IoU over those six classes; background is excluded. Phase 2 is spectral–spatial fusion. Phase 3 is INT8 on the course NPU. Approved by Jeong in writing; confirmed verbally with Mamta.

Read [`README.md`](README.md) first and [`docs/PLAN.md`](docs/PLAN.md) for owners, done-when, and dates. Take one work package. Do not invent a second project.

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
- Wavelength encoding already in the tree is a building block, not the claim.
