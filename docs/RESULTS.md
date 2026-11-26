# Results

| run | model | resolution | epochs | best val FG mIoU | bg | film | basket | cardboard | video_tape | filament | bag | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_ds2 | MiniNetSeg | half (ds2) | 30 | 0.391 (ep29) | 0.868 | 0.552 | 0.495 | 0.578 | 0.286 | 0.000 | 0.436 | first CNN run, half resolution; val noisy ep10-11 |
| tf_ds2_std_lr3e4 | BandTransformer (488k) | half (ds2) | 30 (resumed after ep 13) | 0.299 (ep28) | 0.838 | 0.451 | 0.399 | 0.612 | 0.000 | 0.000 | 0.335 | validation only, not a test score. Per-band standardization (train-only stats), lr 3e-4, no cosine. Overfit gate passed (ep 387 of 600). Video tape and filament never predicted. Run on Colab T4; runtime dropped once, resumed from ep 13 checkpoint. |
| tf_ds2_std_lr3e4_recipe_s0 | BandTransformer (488k) | half (ds2) | 30 | 0.296 (ep28) | 0.845 | 0.464 | 0.412 | 0.592 | 0.000 | 0.000 | 0.307 | validation only. Same config as tf_ds2_std_lr3e4 plus --warmup-steps 20 --clip-grad-norm 1.0, seed 0. Clipping was active in all 30 epochs (largest gradient norm per epoch 4.7 to 10.2, clip 1.0). No gain over the no-recipe runs. |
| tf_ds2_std_lr3e4_s1 | BandTransformer (488k) | half (ds2) | 30 | 0.294 (ep26) | 0.844 | 0.406 | 0.386 | 0.683 | 0.000 | 0.000 | 0.290 | validation only. Control: same config, no recipe, seed 1. Shows run-to-run spread: the three transformer runs end within 0.005 of each other (0.294 to 0.299), while last-10-epoch validation swings by a stdev of 0.016 to 0.033. Video tape and filament were never predicted correctly in any of the 90 validation epochs across the three runs. |

IoU values are fractions (0.391 = 39.1%). 52.8 (MiniNet-v2) and 54.3 (SegFormer-B0) are published hyperspectral-only percentages at full resolution: context, not targets.
