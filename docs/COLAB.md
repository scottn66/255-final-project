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

## Transformer: first runs, in order

`notebooks/06_transformer_gates.ipynb` runs everything in this section in order with Runtime → Run all: setup, data, smoke, memory, overfit, and the half-resolution screening runs. It stops at the first failed gate.

If the transformer stays background-only on more than a handful of cubes, use
`notebooks/07_input_blind_ab.ipynb` on `wp6-colab-gates` (see
[`docs/INPUT-BLIND.md`](INPUT-BLIND.md)): train-only `--standardize`,
`--pe-init-std 0.02`, and `lr 3e-4`. That notebook’s default clone branch is
`wp6-colab-gates`.

Run all asks once for the Drive sign-in. Keep its monitor cell running: an idle notebook can be recycled. After a disconnect, Run all again:
- Gates 1 and 2 are skipped once `gates.json` on Drive records them for the commit.
- Screening runs resume from `last.pt`.

The steps below are the same gates by hand.

Until the WP6 pull requests are merged, clone their branch instead of `main`: `git clone -b wp6-colab-gates https://github.com/scottn66/255-final-project.git`. Train in full precision first (no `--amp`).

1. Smoke. It must print `BandTransformer parameters: 488,487 (0.488M)`.

```bash
PYTHONPATH=src python -m sw.train --model transformer --smoke --device cuda --out-dir /content/sw_smoke
```

2. Memory and time at full resolution, batch 8, in a notebook cell so the peak is readable. Run it from the repo folder with `SW_ROOT` set in the notebook's environment. If it runs out of memory, try batch 4.

```python
import os, sys, json, torch
os.chdir("/content/255-final-project"); sys.path.insert(0, "src")
os.environ["SW_ROOT"] = "/content/data/spectralwaste_segmentation"
from sw.train import main
torch.cuda.reset_peak_memory_stats()
main(["--model", "transformer", "--device", "cuda", "--batch-size", "8", "--epochs", "1",
      "--max-train-steps", "25", "--max-train-images", "200", "--max-val-images", "8",
      "--out-dir", "/content/sw_memcheck"])
print("peak allocated", torch.cuda.max_memory_allocated() / 2**30, "GiB;",
      "peak reserved", torch.cuda.max_memory_reserved() / 2**30, "GiB")
row = json.loads(open("/content/sw_memcheck/metrics.jsonl").read().splitlines()[-1])
print("seconds per step (upper bound)", row["train_seconds"] / 25)
```

`train_seconds` includes the data loader's start-up and the first batches' reads, so with few steps it overstates the per-step time; 25 steps keeps that small. A full epoch at batch 8 is 65 steps (514 training images) plus validation. "Reserved" is what the allocator holds and is closer to what `nvidia-smi` shows; the CUDA context adds a few hundred MB on top.

3. Overfit four training cubes: no augmentation, a constant learning rate, half resolution, one optimizer step per epoch. No 30-epoch run before this passes.

   The four cubes are `sw.gates.OVERFIT_STEMS`; together they hold all six object classes. Link them into a small `SW_ROOT` first. `--max-train-images 4` alone takes the first four training images, which hold only background, film and bag, so basket and cardboard could not be judged.

```python
import sys; sys.path.insert(0, "src")
from sw import gates
gates.build_subset_root("/content/data/spectralwaste_segmentation", "/content/sw_overfit_root")
```

```bash
PYTHONPATH=src python -m sw.train --model transformer --data-root /content/sw_overfit_root --downsample 2 \
  --no-aug --no-cosine --max-train-images 4 --max-val-images 4 --batch-size 4 --epochs 600 --num-workers 0 \
  --device cuda --out-dir /content/sw_gates/tf_overfit
```

   Judge it with `gates.judge_overfit(gates.read_metrics("/content/sw_gates/tf_overfit/metrics.jsonl"), pixels=gates.class_pixels("/content/sw_overfit_root"))`:

   - **Pass (the head as it is now, without the spectral skip):** within 600 epochs, at least one epoch has film, basket, cardboard and bag *all* above 0.9 training IoU. The first such epoch counts, so a later loss spike does not hide a pass. Filament and video tape are reported but not judged: they are 1 to 2 pixels wide and the head blurs boundaries by about a pixel, so they can stay low without anything being wrong.
   - **Pass (once the head has the spectral skip):** `train_fg_miou` above 0.95, all six object classes included.
   - **Still rising at epoch 600:** resume to 900 (`--resume /content/sw_gates/tf_overfit/last.pt --epochs 900`) before calling it a failure.
   - **Fail:** the loss is flat, or no epoch has all four large classes above 0.9. Run the CNN (`--model cnn`) on the same cubes and flags. If the CNN fits, look at the transformer (token order, tokenizer input scale, the head). If it does not fit either, look at the labels, the class weights and the loss.

   Why 600 epochs: measured on real cubes on Nov 16, on Apple MPS (so not a Colab number). The transformer first passed at epochs 337, 447 and 433 (seeds 0, 1, 2); MiniNetSeg (one seed) passed at epoch 43. Training FG mIoU stayed under 0.1 for the first 94 to 131 epochs, while the model predicted almost only background; that is expected. Learning rates 3e-4 and 3e-3 were both slower than 1e-3. One seed's loss spiked around epoch 550 (0.10 to 1.13) at the constant learning rate, then recovered. That spike is a reason to land gradient clipping and warmup (the optimiser-recipe PR) before the long runs. The earlier rule (above 0.9 at 200 or 400 epochs on the first four images) came from synthetic data. On the real images it could judge only two classes, and whether it passed depended on the seed.

Each `metrics.jsonl` row has `train_fg_miou` and `train_per_class_iou` (from the predictions the training steps already make: train mode, before each step's update, so they lag a little), `train_seconds` (the training pass) and `epoch_seconds` (training plus validation). `metrics.csv` keeps its original columns. The transformer may use `--batch-size 1`; the CNN needs 2 or more (BatchNorm). A transformer `--crop-size` must be a multiple of the patch (8).

## Resume

`last.pt` is written at the end of every epoch. It holds the model, the AdamW state, the scheduler, the finished epoch, and the best validation foreground mIoU. `best.pt` is the best weights only, so resume from `last.pt`.

`--epochs` is the total budget, not a count of extra epochs. `last.pt` records the last epoch that finished, and resume starts at the next one. A checkpoint whose epoch is 12, resumed with `--epochs 30`, runs epochs 13 through 30. The metrics files in `--out-dir` are appended.

```bash
PYTHONPATH=src python -m sw.train --resume /content/drive/MyDrive/sw_runs/cnn/last.pt --epochs 30 --batch-size 4 --num-workers 2 --device cuda --patience 8 --out-dir /content/drive/MyDrive/sw_runs/cnn
```
