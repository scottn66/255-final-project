# MiniNetSeg INT8 probe (draft lab toolkit)

This is not a course submission and not a Canvas upload.
Do not merge it. Do not push it to main.
Scott Nelson has to read and accept every line before anyone uses it.
Jeong treats LLM-generated course code as cheating. This branch stays off main until that review.

Written against main `7ea45560347c4c8a6412c321b6162d814bab0dbc` (merge of PR #12). If main has moved, re-read `src/sw/models/cnn.py` and `src/sw/models/transformer.py` before trusting this kit.

## Purpose

Check whether the already-trained MiniNetSeg CNN can go through the Lab 2 Qubee path: ONNX export, then `mxq_compile`, then an NPU score. The network is the class in `src/sw/models/cnn.py`: 571,879 parameters, no pretrained weights. The scored run is cnn_ds2 (downsample 2, about 128×128). Best reported val foreground mIoU is 0.391. Filament IoU on that run is 0.000.

The weights are not in git. They were saved to Google Drive `MyDrive/sw_runs/cnn_ds2/best.pt`. Every script takes a path and stops if the file is missing. Do not commit `best.pt`, cubes, or compiled models.

A first compile may fail. That is a result. This directory does not change MiniNetSeg, and it has no transformer INT8 path. `SpatialPE`, `SpatialBlock`, and `SegHead` stay as they are on main. SpectralBlock and SpectralPool landed in PR #12; this probe does not touch them.

Nothing here books the NPU or runs `mxq_compile`. Computer 3 is where those commands run, later, after review.

## Layout contract

MiniNetSeg is already NCHW. `forward` takes `(N, in_channels, H, W)` and returns logits `(N, num_classes, H, W)`. Defaults are `in_channels=224` and `num_classes=7`.

The ONNX file is static batch 1:

| Tensor | Default shape | Dtype |
| --- | --- | --- |
| input `cube` | `(1, 224, 128, 128)` NCHW | float32 |
| output `logits` | `(1, 7, 128, 128)` NCHW | float32 |

128 is downsample 2 on a 256 cube. Do not switch the export to 256×256. The scripts warn if you pass 256, and the calibration writer refuses a spatial size that the stride did not produce.

`--nhwc` is off unless you pass it to both `export_onnx.py` and `convert_mxq.py`. It wraps the CNN so the file's input is `(1, 128, 128, 224)` and the logits are `(1, 128, 128, 7)`. That layout is unconfirmed against Jeong's announcement 5928848 scripts. The Qubee 0.11 quick start says a compile without CPU offload wants calibration arrays shaped `(H, W, C)`. This probe still defaults to NCHW, matching `MiniNetSeg.forward` and the ONNX file. A collapsed foreground mIoU is usually this layout or the calibration set, not the weights.

There is no uint8 cube contract. At the runtime boundary, feed the same float32 array the ONNX file takes and call `maccel.Model.infer`. Do not call `infer_to_float` for the score (that path wants manual int8 scaling). `sample_dtype="float"` in the compile call is the 0.12 default for debug samples, not a second input type.

## Preprocess (same for export, calibration, and score)

From `src/sw/data.py` and `prepare_sample` in `src/sw/train.py`:

1. Read `hyper/<split>/<stem>.tiff` with the loader. On disk the cube is uint16, band-first, real files 224×256×256.
2. Divide by 65535. The tensor is float32 in `[0, 1]`. No per-band mean or standard deviation.
3. Inference geometry is `prepare_sample(..., train=False, crop_size=None, downsample=2)`: `cube[:, ::2, ::2]`, no flip, no crop. 256 becomes 128.
4. Add a batch axis: `(1, 224, 128, 128)`.

Training flips are augmentation. They are not part of export, calibration, or the val score. The cnn_ds2 command in `docs/COLAB.md` did not set `--crop-size`.

## Risky ops (leave them)

MiniNetSeg uses depthwise / grouped conv, dilated depthwise, unfused BatchNorm, bilinear upsample, and Concat skips, plus a residual Add inside the dilation block. Lab 1/2 on this course only scored dense Conv, ReLU, MaxPool, GAP, Flatten, Gemm, and (Lab 2) residual Add on Qubee 0.12.

`export_onnx.py` turns constant folding and ONNX optimize off so BatchNorm is not fused on the way out. Folding can be a later, explicit change. It is a commented TODO in that file, not something this probe applies. Do not edit `src/sw/models/cnn.py` so the compiler accepts the graph.

## Export

On a machine that has the checkpoint and this repo (PyTorch, no NPU):

```bash
PYTHONPATH=src python experiments/cnn_int8_pilot/export_onnx.py \
  --checkpoint /path/to/sw_runs/cnn_ds2/best.pt \
  --out /tmp/mininet_seg_ds2.onnx
```

Defaults are `--height 128 --width 128 --bands 224`. Opset 13 is a probe default for the legacy exporter (`dynamo=False`). PyTorch 2.9+ warns that this exporter is old. It stays on so the dynamo optimizer cannot fold the graph. It is not a Computer 3 confirmation. The script exits if `--checkpoint` is missing or the state dict is not MiniNetSeg.

`MiniNetSeg.forward` ends with a Python `if` on the spatial size. The tracer freezes that branch for the dummy input. Four stride-2 stages divide H and W by 16, and the decoder restores the input, so a 128 export does not take the fallback upsample. The ONNX file is for that H and W only. Do not edit `cnn.py` to remove the warning.

## Compile on Computer 3

Lab 2 notes, not a reading of the machine today:

* Docker image `mobilint/qbcompiler:0.12-cpu-ubuntu22.04`
* Wheel `qubee-0.12.0.0+aries2-py3-none-any.whl`
* `mxq_compile(..., quantization_mode=2, quantization_method=1, percentile=0.999, quantization_output=0, backend="onnx", device="cpu")`
* `use_random_calib=False`, `cpu_offload=False`
* If the NPU foreground mIoU drops more than 0.1 versus FP32, recompile with `--percentile 0.9999`

Build calibration from the **train** split only, about 100–200 cubes, then compile inside the container. Pasteable commands are in `lab_pc_commands.txt`.

```bash
PYTHONPATH=src python experiments/cnn_int8_pilot/convert_mxq.py \
  --materialize-calib \
  --data-root "$SW_ROOT" \
  --calib-dir /tmp/cnn_ds2_calib \
  --calib-count 150

# inside the compiler image, after qubee imports:
PYTHONPATH=src python experiments/cnn_int8_pilot/convert_mxq.py \
  --onnx /tmp/mininet_seg_ds2.onnx \
  --calib-dir /tmp/cnn_ds2_calib \
  --out /tmp/mininet_seg_ds2.mxq
```

`qubee` is absent in CI, so the second command exits before `mxq_compile`. That is expected off Computer 3.

`--materialize-calib` uses `SpectralWasteDataset(..., split="train")` and `preprocess_train_cube`. It writes one float32 `.npy` per selected train stem. It refuses a `val` or `test` path component, refuses JPEGs and tiffs, and refuses a count outside 100–200 unless `--allow-short-calib` (tests only).

TODO: Qubee 0.12 also accepts a `.txt` or `.json` manifest of npy paths. This script only accepts a flat directory of `.npy`. The tiff loader is the source of truth for the cube values. The npy container is the 0.12 `calib_data_path` description (`docs.mobilint.com` compiler v0.12.0.0). The `(H, W, C)` rank from the 0.11 quick start is the open layout question above, not something this writer guesses.

## Score (document only; do not run the NPU from this PR)

Compare **val** foreground mIoU, three ways. Foreground mIoU is classes 1–6, background left out, via `sw.metrics.ConfusionMeter` / `fg_miou`. Logits must be NCHW `(N, 7, H, W)` before that helper. If you exported with `--nhwc`, permute the logits back before scoring.

| Run | What it tells you |
| --- | --- |
| FP32 PyTorch, loaded `best.pt`, `model.eval()` | The reference. cnn_ds2 reported 0.391. Filament IoU 0.000 can show up again and is not, by itself, a layout failure. |
| FP32 ONNX Runtime, same preprocessed val cubes | Must land within about 0.1 of the PyTorch score. A larger gap means the export is wrong. |
| INT8 on the NPU, `maccel` 0.30.1, `Model.infer` on float32 | The deployment number. CPU INT8 is not this number. An ONNX Runtime INT8 session is not this number. |

Latency, NPU only: batch 1, 20 warm-ups, then median and p90 over at least 200 timed iterations. Record the compiler version string you actually import and the hardware string the runtime prints (Lab 2's device node is `/dev/aries0` for `Accelerator(0)`). 

Lab 2 latency context, for the checklist only: a 0.12M dense fused net ran at 14.0 ms with about a 0.04 dB float-to-NPU drop; a 9.4M class model took 411 ms. 14 ms is not a MiniNetSeg forecast. This CNN is 0.57M and its graph is not that fused net.

Val is for the score. Train is for calibration. Do not calibrate on val or test.

## What to copy off the machine

Copy these back to Drive or the lab notes. Do not commit them.

* the `.mxq`
* the compile log
* the timing record (median, p90, warmup 20, n ≥ 200, batch 1)
* a metrics JSON with `fp32_pytorch_val_fg_miou`, `fp32_onnxruntime_val_fg_miou`, `int8_npu_val_fg_miou`, the latency block, `layout` (`nchw` or `nhwc`), `percentile`, and the note that CPU INT8 is not an NPU result
* the `qubee` version string and the `maccel` version string printed on that machine

Leave the dataset and `best.pt` where they are.

## Tests

`pytest -q tests/test_cnn_int8_pilot.py` builds a random MiniNetSeg checkpoint in a temp directory (nothing is committed), checks the ONNX shapes, and checks the Qubee flag defaults. It does not download SpectralWaste and it does not call `mxq_compile`. `onnx` is an extra import for the shape test; the rest of the repo does not require it.
