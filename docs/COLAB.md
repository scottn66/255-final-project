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

## Optimiser recipe

Three flags, all off by default, so a command without them trains exactly as before:

- `--warmup-steps W` (default 0): linear warmup. Optimizer step k, counted from 1 over the whole run, uses the scheduled rate times `min(1, k / W)`. W is in optimizer steps, not epochs: at batch 4 an epoch of the full training split is 129 steps. With cosine on, the two multiply: cosine sets the rate once per epoch, warmup scales it per step. The step count is in `last.pt`, so a resumed run continues the ramp.
- `--clip-grad-norm C` (default: no clipping): `torch.nn.utils.clip_grad_norm_` on all model parameters before each optimizer step (after `scaler.unscale_` under `--amp`). Each `metrics.jsonl` row then has `grad_norm_max`, the largest finite norm of that epoch before clipping (`null` if no step had one), and `grad_nonfinite_steps`, the number of steps whose norm was NaN or infinite. Under `--amp` a few such steps are normal: `GradScaler` skips them. They still count as warmup steps.
- `--cosine-floor F` (default 0): with cosine on, the rate anneals from `--lr` down to `--lr` times F, not to 0. It bounds the cosine schedule, not the warmup ramp. The last epoch trains just above the floor; the floor itself is the rate after the last epoch. With `--no-cosine` it is ignored and a line says so.

Measured on Nov 18 on the four overfit cubes of Gate 2 (Apple MPS, the Gate 2 flags, 800 epochs of one step, seeds 0, 1, 2), without the flags and with `--clip-grad-norm 1.0 --warmup-steps 20`. A wiring check on four training images, not a result.

| seed | first epoch with all four large classes above 0.9: no flags / recipe | largest one-epoch loss increase: no flags / recipe |
| --- | --- | --- |
| 0 | 337 / 382 | +6.64 at epoch 762 (0.52 to 7.16) / +0.14 at epoch 160 |
| 1 | 447 / 378 | +0.37 at epoch 115 / +0.18 at epoch 119 |
| 2 | 433 / 442 | +0.24 at epoch 155 / +0.46 at epoch 137 |

- Without the flags, seed 0 fell apart at epoch 760 (loss 0.07 to 7.16 in three epochs) and had not recovered by epoch 800 (training FG mIoU 0.04). Continuing that same run from its epoch-755 checkpoint with `--clip-grad-norm 1.0` gave no spike (loss at most 0.076 through epoch 800); continuing it without the flag gave the same collapse again.
- The recipe did not make the fit faster or slower in any one direction: later for seeds 0 and 2, earlier for seed 1. All six runs pass Gate 2 within 600 epochs.
- Clipping does not remove the loss jumps of 0.1 to 0.5 in epochs 100 to 200, while the model leaves the all-background stage. After the first passing epoch, no recipe run rose by more than 0.04 in one epoch.

Fix `--epochs` once for a cosine run and do not extend it. A resume keeps the schedule stored in `last.pt`: with a larger `--epochs` the rate climbs back up after the old last epoch, and `sw.train` prints a `WARNING` line. `--epochs` is not part of a run's identity in `sw.gates`, so `sw.gates` does not stop this. To train longer with cosine, start a new run. A resume with a different `--cosine-floor` (or a different `--lr` with a floor) is refused.

A run that `sw.gates` started without these flags resumes as before. Adding one to a run already started is a different run: `sw.gates` refuses it, so use a new folder.

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

## Long runs across sessions, for 2 to 4 people (`notebooks/09_long_run.ipynb`)

The screens train 30 epochs. Notebook 09 trains hundreds to thousands, survives the Colab session limit, and lets a few people share one folder of runs. Set `RUNNER_NAME` in the first code cell, then Runtime → Run all.

**One run, many sessions.** The recipe is warmup (5 epochs), then cosine over `EPOCHS_TARGET` (default 1000) down to 1 % of the peak rate, `--lr 3e-4` for the transformers (`1e-3` for the CNN, passed explicitly), `--standardize`, `--clip-grad-norm 1.0`, early stopping off, optional `--pe-init-std`. `sw.train` gained these flags, all off or equal to today's behaviour by default:

| Flag | Default | Effect |
|---|---|---|
| `--save-every N` | 1 | write `last.pt` every N epochs (always on the last epoch and on a stop) |
| `--keep-last K` | 0 | also keep the K newest `ckpt_eNNNNN.pt`, so a fork can start from an earlier epoch |
| `--eval-every N` | 1 | full val FG mIoU and per-class IoU every N epochs (and on the last or a stop); other rows log null |
| `--max-minutes M` | off | stop cleanly after a saved epoch once another would not fit in M minutes; `sw.gates` then writes no `DONE.json` |
| `--init-from CKPT`, `--init-optimizer` | off | start a new run from the weights (and AdamW moments) of a checkpoint: a fork |

Checkpoints are written to a temporary file and renamed, so a disconnect cannot leave a half-written file. `last.pt` also holds the random state (python, numpy, torch, CUDA, the train loader's shuffle), so a resumed run follows the stream an unbroken run would have. Log rows past the checkpoint's epoch are dropped on resume. Everything goes to `ROOT/<RUN_TAG>/` on Drive. The time budget (`TIME_BUDGET_MINUTES`, default 600) ends a session before Colab does. Open the notebook in the next session and Run all: the run resumes. If the previous session's VM is gone, its lock goes stale within 10 minutes; the notebook waits for that when the lock is yours.

**Fix `EPOCHS_TARGET` when a run starts.** The cosine schedule is stored in `last.pt`; the notebook refuses a changed target. To train longer, fork the run (below). This is the rule from "Optimiser recipe" above.

**Sharing a folder.** One person creates `SW_runs/long_run/` in Drive and shares it with the others; each adds a shortcut to it in **My Drive** named `SW_runs`, so `ROOT` is the same folder for everyone. Each person downloads the data to their own `/content/data`; nothing shares the cubes, and nobody trains from Drive.

- **Claim.** Starting `ROOT/<RUN_TAG>` writes `LOCK.json` (owner, session start, heartbeat). The training process refreshes the heartbeat every minute, so closing your tab does not make a live run look dead. A heartbeat newer than 10 minutes refuses a second claim: pick another `RUN_TAG`, or set `AUTO_NEW_TAG = True` to take `<tag>-2`. An older heartbeat is *stale* and can be taken over. `FORCE_TAKEOVER = True` takes a live lock; use it only for your own lost session. If another session takes the lock, the old process exits.
- **Race.** Drive cannot create a file atomically across machines. A claim is written, waited on, and read back, which catches two people starting in the same second but not every race. Give each person their own tags (the default is `<runner>-<model>`).
- **Registry.** Each run writes `metadata.json`: git sha, flags, seed, learning rate, owner, a fingerprint of the data (split counts and a hash of the file names), the parent run and epoch if it is a fork, and one entry per session. A run folder started on other flags or other data is refused. The Leaderboard cell scans every folder under `ROOT` and prints epoch reached (of the target), best val FG mIoU and its epoch, per-class IoU at the best epoch, owner, and state (`running`, `stale`, `paused`, `done`, `failed`); the last plot draws every run's val curve together. Runs with different data fingerprints are flagged.
- **Fork.** Set `FORK_FROM` (a run's tag), optionally `FORK_EPOCH` (needs `--keep-last` in the parent) and `FORK_OPTIMIZER`, a new `RUN_TAG`, and a different `LR`, `SEED` or `AUGMENT`. The notebook copies the checkpoint into the new folder (`init.pt`, weights only unless `FORK_OPTIMIZER`) and starts a fresh schedule of `EPOCHS_TARGET` epochs from those weights. The model flags must match the parent's. A re-run of the cell keeps the fork it started from. The new run's own `--lr` applies even when the moments are taken.
- **Soup (experimental).** `SOUP_RUNS` lists two or more forks of the same parent checkpoint. The last cell averages their `best.pt` weights, scores the average and each ingredient on the full val split, and saves the average as a new run (`SOUP_TAG`) only if it beats every ingredient. This is a trick we have not tested here; treat a saved soup as a hypothesis, not a result.
- **Writes.** Every file the notebook or `sw.longrun` writes to Drive (checkpoints, `LOCK.json`, `metadata.json`, `ARGS.json`, `RUNNING.json`, `DONE.json`, rewritten logs, `band_stats.pt`) goes through a temporary file and a rename.

**Not supported: data-parallel training across Colab VMs.** One run trains on one VM. Several people share a folder of independent runs, forks and a soup, not one gradient step. Splitting one run's batches across two Colab VMs needs a network path between them that Colab does not give you.

**Band statistics.** `--standardize` caches `band_stats.pt` in the run folder, so a run never reads another run's statistics; a fork copies its parent's. The data fingerprint in `metadata.json` stops a folder from continuing on different data.

**Crashes.** The monitor cell raises, with the end of `train.log`, when the process exited without finishing, when the lock went stale, when the lock says `failed`, or when the log stops growing; it never waits on a dead run. A crash writes `failed` and the error into `LOCK.json`, and the next Run all resumes from `last.pt`.
