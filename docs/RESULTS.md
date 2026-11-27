# Results

| run | model | resolution | epochs | best val FG mIoU | bg | film | basket | cardboard | video_tape | filament | bag | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_ds2 | MiniNetSeg | half (ds2) | 30 | 0.391 (ep29) | 0.868 | 0.552 | 0.495 | 0.578 | 0.286 | 0.000 | 0.436 | first CNN run, half resolution; val noisy ep10-11 |
| tf_ds2_std_lr3e4 | BandTransformer (488k) | half (ds2) | 30 (resumed after ep 13) | 0.299 (ep28) | 0.838 | 0.451 | 0.399 | 0.612 | 0.000 | 0.000 | 0.335 | screening; run by Yash. Numbers and notes copied from `docs/RESULTS.md` on branch `yash/wp6-transformer-screen-results`; no log of the run is in this repo and the commit is not recorded. His notes: validation only, not a test score. Per-band standardization (train-only stats), lr 3e-4, no cosine. Overfit gate passed (ep 387 of 600). Video tape and filament never predicted. Run on Colab T4; runtime dropped once, resumed from ep 13 checkpoint. |
| tf_fused_std | BandTransformer 2+6 (488,487) | half (ds2) | 30 | 0.286 (ep30) | 0.847 | 0.437 | 0.367 | 0.622 | 0.000 | 0.000 | 0.288 | screening; `notebooks/08_screen_three_arms.ipynb`, commit 318358e, batch 4, constant lr, `--standardize`, default seed (0), Colab L4; `--lr 3e-4`; the best epoch is the last one |
| tf_collapsed_std | BandTransformer 0+8, collapsed tokenizer: the control (491,191) | half (ds2) | 30 | 0.298 (ep28) | 0.837 | 0.465 | 0.375 | 0.618 | 0.000 | 0.000 | 0.327 | screening; `notebooks/08_screen_three_arms.ipynb`, commit 318358e, batch 4, constant lr, `--standardize`, default seed (0), Colab L4; `--lr 3e-4`; ended at 0.289 at epoch 30 |
| cnn_std | MiniNetSeg (571,879) | half (ds2) | 30 | 0.403 (ep25) | 0.860 | 0.543 | 0.517 | 0.599 | 0.302 | 0.000 | 0.458 | screening; `notebooks/08_screen_three_arms.ipynb`, commit 318358e, batch 4, constant lr, `--standardize`, default seed (0), Colab L4; `--lr 1e-3`; ended at 0.277 at epoch 30 |

IoU values are fractions (0.391 = 39.1%). 52.8 (MiniNet-v2) and 54.3 (SegFormer-B0) are published hyperspectral-only percentages at full resolution: context, not targets.

`tf_fused_std`, `tf_collapsed_std` and `cnn_std` are like-for-like with each other: one notebook run, one commit, the same flags apart from the model and its learning rate. The numbers are the ones stored in the notebook's outputs. They are one seed and 30 epochs at half resolution, so they are screening rows. The best epoch is a noisy pick: `cnn_std` scored 0.403 at epoch 25 and 0.277 at epoch 30.

`cnn_ds2` is not like-for-like with those three. It used raw input and a cosine schedule. `tf_ds2_std_lr3e4` is a teammate's reported run. Its reported flags match `tf_fused_std`, but its commit and batch size are not recorded, it ran on a T4 and was resumed once, and no log is in this repo. Treat it as not verified like-for-like.

`notebooks/09_fullres_fused_vs_control.ipynb` runs the fused model and the control at full resolution. No full-resolution row is in this table yet.
