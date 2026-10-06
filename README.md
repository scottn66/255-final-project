# SpectralWaste segmentation — DATA 255 Group 1

We are building a transformer from scratch that turns hyperspectral bands into tokens and labels each pixel as one of six waste classes. We train on SpectralWaste (one camera, [Zenodo 10880544](https://zenodo.org/records/10880544)) and score mean IoU over those six classes only; background is left out. The hyperspectral-only numbers to compare against are 52.8 (MiniNet-v2) and 54.3 (SegFormer-B0). Next we fuse spectral and spatial features, then try INT8 on the course NPU.

**Group:** Scott Nelson, Surya Reddy, Tim Wada, Prakhar, Yash. SJSU DATA 255, Fall 2026.

**Status:** Topic approved by Jeong in writing; confirmed verbally with Mamta. Data verified: 514 train / 167 val / 171 test.

## Start here

```bash
git clone https://github.com/scottn66/255-final-project.git
cd 255-final-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/get_data.sh
export SW_ROOT="$PWD/spectralwaste_segmentation"
```

On Colab, skip the virtualenv and run that download in the notebook. The unpacked set is about 24 GB, so download it there instead of through Drive. Then open `notebooks/01_look_at_the_data.ipynb` and `notebooks/02_pipeline_smoke_test.ipynb`.

## Repo map

- [`docs/PLAN.md`](docs/PLAN.md) — work packages, owners, timeline
- `scripts/get_data.sh` — download the Zenodo zip and check its md5
- `notebooks/` — look at one cube, then a one-batch smoke test
- `src/sw/` — dataset, metric, and the model
- `tests/` — run `pytest -q`

## How we work

One branch per work package, small pull requests, one teammate review. Never commit the dataset, its images, zips, or checkpoints. The loader reads the data path from `SW_ROOT`.

Full plan: [`docs/PLAN.md`](docs/PLAN.md).
