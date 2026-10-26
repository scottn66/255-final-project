# Training on Colab

The unpacked SpectralWaste set is about 24 GB, and each epoch reads every cube from disk. Keep the cubes on the Colab local disk (`/content`). Google Drive is a poor place for those reads. Drive is the right place for checkpoints: a Colab GPU session can die, and the local disk dies with it.

## Setup

1. Runtime → Change runtime type → GPU.
2. Mount Drive for the run directory only:

```python
from google.colab import drive
drive.mount("/content/drive")
```

3. Clone, install, and download onto local disk. Colab already has a current torch; do not pin an older one.

```bash
git clone https://github.com/scottn66/255-final-project.git
cd 255-final-project
pip install -q -r requirements.txt
bash scripts/get_data.sh /content/data
export SW_ROOT=/content/data/spectralwaste_segmentation
```

`SW_ROOT` is the folder that contains `hyper/` and `labels_hyper_lt/`. If the set is already unpacked somewhere else, copy that folder to `/content`. Do not train against a Drive path.

## Recommended run

`--num-workers 2` overlaps the tiff reads with the GPU step. Checkpoints and the metrics log go to Drive.

```bash
PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --num-workers 2 --device cuda --patience 8 --out-dir /content/drive/MyDrive/sw_runs/cnn
```

That command is full resolution (256). A faster first score keeps every other pixel, so 256 becomes 128:

```bash
PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --num-workers 2 --device cuda --patience 8 --downsample 2 --out-dir /content/drive/MyDrive/sw_runs/cnn_ds2
```

The learning rate follows cosine annealing over `--epochs`. `--no-cosine` keeps it constant. `--amp` turns on mixed precision (`torch.autocast` and `GradScaler`) on CUDA only.

## Resume

`last.pt` is written at the end of every epoch. It holds the model, the AdamW state, the scheduler, the finished epoch, and the best validation foreground mIoU. `best.pt` is the best weights only, so resume from `last.pt`.

`--epochs` is the total budget, not a count of extra epochs. `last.pt` records the last epoch that finished, and resume starts at the next one. A checkpoint whose epoch is 12, resumed with `--epochs 30`, runs epochs 13 through 30. The metrics files in `--out-dir` are appended.

```bash
PYTHONPATH=src python -m sw.train --resume /content/drive/MyDrive/sw_runs/cnn/last.pt --epochs 30 --batch-size 4 --num-workers 2 --device cuda --patience 8 --out-dir /content/drive/MyDrive/sw_runs/cnn
```
