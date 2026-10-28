# Results

| run | model | resolution | epochs | best val FG mIoU | bg | film | basket | cardboard | video_tape | filament | bag | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_ds2 | MiniNetSeg | half (ds2) | 30 | 0.391 (ep29) | 0.868 | 0.552 | 0.495 | 0.578 | 0.286 | 0.000 | 0.436 | first CNN run, half resolution; val noisy ep10-11 |

IoU values are fractions (0.391 = 39.1%). 52.8 (MiniNet-v2) and 54.3 (SegFormer-B0) are published hyperspectral-only percentages at full resolution: context, not targets.
