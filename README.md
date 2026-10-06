# SpectralWaste segmentation — DATA 255 Group 1

**Group 1 (Scott, Surya, Tim, Prakhar, Yash).** We segment jam objects on a recycling line using the SpectralWaste hyperspectral dataset ([Casao et al., IROS 2024](https://arxiv.org/abs/2403.18033)). That's 224 bands from 900 to 1700 nm, 6 object classes plus background, and the official 514/167/171 split. We build a small spectral–spatial transformer from scratch in three phases: (1) a minimal transformer that segments the hyperspectral images, (2) spectral–spatial fusion with ablations, and (3) INT8 quantization for portable or NPU deployment, where we measure how much mIoU drops. We report mIoU over the 6 object classes. Our bar is the published hyperspectral-only results, MiniNet-v2 at 52.8 and SegFormer-B0 at 54.3. CMX-B0's 58.2 uses RGB and hyperspectral together, so we cite it as context, not as a target. Topic approved by Jeong; confirmed verbally with Mamta.

## Start here

```bash
git clone https://github.com/scottn66/255-final-project.git
cd 255-final-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/get_data.sh
export SW_ROOT="$PWD/data/spectralwaste_segmentation"
```

On Colab, skip the virtualenv and run that download in the notebook. The unpacked set is about 24 GB, so download it there instead of through Drive. Then open `notebooks/01_look_at_the_data.ipynb` and `notebooks/02_pipeline_smoke_test.ipynb`.

## Repo map

- [`docs/PLAN.md`](docs/PLAN.md) — work packages, owners, timeline
- `scripts/get_data.sh` — download the Zenodo zip and check its md5
- `notebooks/` — look at one cube, then a one-batch smoke test
- `src/sw/` — dataset loader, FG-only mIoU, and from-scratch attention / wavelength-encoding building blocks
- `tests/` — run `pytest -q`

## How we work

One branch per work package, small pull requests, one teammate review. Never commit the dataset, its images, zips, or checkpoints. The loader reads the data path from `SW_ROOT`.

Full plan: [`docs/PLAN.md`](docs/PLAN.md).
