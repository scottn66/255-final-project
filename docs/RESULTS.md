# Results

| run | model | resolution | epochs | best val FG mIoU | bg | film | basket | cardboard | video_tape | filament | bag | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_ds2 | MiniNetSeg | half (ds2) | 30 | 0.391 (ep29) | 0.868 | 0.552 | 0.495 | 0.578 | 0.286 | 0.000 | 0.436 | first CNN run, half resolution; val noisy ep10-11 |
| tf_fused_std | BandTransformer 2+6 | half (ds2) | 30 | 0.299 (ep28) | not reported | not reported | not reported | not reported | 0.000 | 0.000 | not reported | screening; run by Yash; score, epoch and the two 0.000 classes as reported by Yash, not checked against a log; flags reported as `--standardize --lr 3e-4 --no-cosine`; commit not recorded |

IoU values are fractions (0.391 = 39.1%). 52.8 (MiniNet-v2) and 54.3 (SegFormer-B0) are published hyperspectral-only percentages at full resolution: context, not targets.

`cnn_ds2` used raw input and a cosine schedule. `tf_fused_std` is reported with `--standardize` and a constant learning rate; no log of it is in the repo. The two rows are not a like-for-like comparison. `notebooks/08_screen_three_arms.ipynb` reruns the fused model, the control and the CNN with the same flags from one commit.
