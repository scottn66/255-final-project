#!/usr/bin/env python3
"""Export the trained MiniNetSeg CNN to a static ONNX graph.

The graph is the model as it is in ``src/sw/models/cnn.py``. This script does
not fuse BatchNorm, does not rewrite depthwise or dilated convolutions, and
does not transpose to NHWC unless ``--nhwc`` is passed.

Preprocess this file assumes (it does not read a cube):

1. ``SpectralWasteDataset.__getitem__`` in ``src/sw/data.py`` loads
   ``hyper/<split>/<stem>.tiff`` as uint16, band-first ``(C, H, W)``, and
   divides by ``65535`` (``UINT16_MAX``). The tensor is float32 in ``[0, 1]``.
   There is no per-band mean or standard deviation.
2. ``prepare_sample(..., train=False, crop_size=None, downsample=2)`` in
   ``src/sw/train.py`` keeps every other pixel (``cube[:, ::2, ::2]``).
   A 256 cube becomes 128. Validation does not flip. The cnn_ds2 command in
   ``docs/COLAB.md`` left ``--crop-size`` unset, so the full 128 map is kept.
3. The ONNX input is that cube plus a batch axis: float32 NCHW
   ``(1, 224, 128, 128)``. The output is logits, float32 NCHW
   ``(1, 7, 128, 128)``.

``--nhwc`` is an explicit layout probe. It is off by default. It is not
confirmed against Jeong's announcement 5928848 scripts.

The checkpoint is not in git. cnn_ds2 weights were saved to Google Drive
``MyDrive/sw_runs/cnn_ds2/best.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sw.models.cnn import MiniNetSeg

DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_BANDS = 224
DEFAULT_OPSET = 13
CHECKPOINT_HINT = (
    "The cnn_ds2 weights are not in git. They were saved to Google Drive "
    "MyDrive/sw_runs/cnn_ds2/best.pt. Pass that file with --checkpoint. "
    "Do not commit best.pt."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MiniNetSeg to a static batch-1 ONNX file. NCHW unless --nhwc is set."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="best.pt from the cnn_ds2 run (Drive MyDrive/sw_runs/cnn_ds2/best.pt). Not in git.",
    )
    parser.add_argument("--out", required=True, help="Destination .onnx path. Do not commit it.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=(
            "ONNX opset for the legacy exporter. 13 is a probe default, "
            "not a version confirmed on Computer 3."
        ),
    )
    parser.add_argument(
        "--nhwc",
        action="store_true",
        help=(
            "Export an NHWC input and NHWC logits. Default off. "
            "Unconfirmed against Jeong's announcement 5928848 scripts. "
            "A collapsed metric is usually this flag or the calibration set."
        ),
    )
    return parser


def input_shape(bands: int, height: int, width: int, *, nhwc: bool) -> tuple[int, int, int, int]:
    """Static batch-1 shape. NCHW unless the explicit NHWC probe is on."""
    if nhwc:
        return (1, height, width, bands)
    return (1, bands, height, width)


def output_shape(num_classes: int, height: int, width: int, *, nhwc: bool) -> tuple[int, int, int, int]:
    if nhwc:
        return (1, height, width, num_classes)
    return (1, num_classes, height, width)


class _NhwcProbe(nn.Module):
    """Permute around MiniNetSeg. Used only when ``--nhwc`` is passed.

    MiniNetSeg itself stays NCHW. This wrapper is the whole layout switch.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        logits = self.model(cube.permute(0, 3, 1, 2).contiguous())
        return logits.permute(0, 2, 3, 1).contiguous()


def _state_dict(blob, path: Path) -> tuple[dict, dict]:
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        return blob["model"], blob
    if isinstance(blob, dict) and blob and all(torch.is_tensor(value) for value in blob.values()):
        return blob, {}
    raise SystemExit(
        f"unrecognized checkpoint layout in {path}. "
        "Expected best.pt or last.pt: a dict with a 'model' state_dict "
        "(see src/sw/train.py)."
    )


def load_mininet(path: Path, bands: int) -> tuple[MiniNetSeg, dict]:
    """Load MiniNetSeg weights. Fail if the file is missing or is not the CNN."""
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}\n{CHECKPOINT_HINT}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SystemExit(f"could not read checkpoint {path}: {exc}\n{CHECKPOINT_HINT}") from exc
    state, meta = _state_dict(blob, path)
    cfg = meta.get("cfg") or {}
    if cfg.get("model") not in (None, "cnn"):
        raise SystemExit(
            f"{path} records model={cfg.get('model')!r}. "
            "This probe exports MiniNetSeg only. There is no transformer INT8 path."
        )
    if any(key.startswith(("tokenizer.", "spectral.", "pos.", "spatial.")) for key in state):
        raise SystemExit(
            f"{path} looks like BandTransformer. "
            "This probe exports MiniNetSeg only. There is no transformer INT8 path."
        )
    if "stem.0.weight" not in state or "head.weight" not in state:
        raise SystemExit(
            f"{path} has no MiniNetSeg stem.0.weight / head.weight. "
            "Refusing to guess an architecture."
        )
    ckpt_bands = int(state["stem.0.weight"].shape[1])
    num_classes = int(state["head.weight"].shape[0])
    if ckpt_bands != bands:
        raise SystemExit(
            f"--bands is {bands} but the checkpoint stem takes {ckpt_bands} channels. "
            "cnn_ds2 used 224. Refusing to export a width the weights were not trained for."
        )
    model = MiniNetSeg(in_channels=ckpt_bands, num_classes=num_classes)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"checkpoint does not match MiniNetSeg (strict load failed): {exc}\n"
            "Do not edit src/sw/models/cnn.py to force a match."
        ) from exc
    model.eval()
    # The tracer rejects inplace ReLU. This flips the flag on the loaded copy
    # only. The class in cnn.py is unchanged, and the math is the same.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model, meta


def export_graph(
    model: nn.Module,
    dummy: torch.Tensor,
    out_path: Path,
    opset: int,
) -> None:
    """Write one ONNX file. BatchNorm stays unfused (``do_constant_folding=False``).

    TODO(qubee): MiniNetSeg contains depthwise / grouped conv, dilated
    depthwise, unfused BatchNorm, bilinear upsample, and Concat skips.
    Lab 1/2 on this course only scored dense Conv, ReLU, MaxPool, GAP,
    Flatten, Gemm, and (Lab 2) residual Add on Qubee 0.12. A first compile
    may fail. That is useful data. Do not fuse BatchNorm in this function,
    and do not rewrite MiniNetSeg so the compiler accepts it.
    """
    if dummy.ndim != 4 or dummy.shape[0] != 1 or dummy.dtype != torch.float32:
        raise SystemExit(
            f"ONNX dummy must be float32 NCHW or explicit NHWC with batch 1, got {tuple(dummy.shape)} {dummy.dtype}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False keeps the legacy TorchScript exporter. PyTorch 2.9+ warns
    # that this path is deprecated. The dynamo exporter defaults to optimize,
    # which can rewrite the graph. Constant folding stays off so BatchNorm is
    # not fused into Conv on the way out.
    #
    # MiniNetSeg.forward ends with a Python `if` on the spatial size. The
    # tracer freezes that branch for this dummy. Four stride-2 stages divide
    # H and W by 16, and the decoder restores the input size, so a 128 export
    # does not need the fallback upsample. The file is for this H and W only.
    # Do not edit cnn.py to silence the tracer warning.
    torch.onnx.export(
        model,
        (dummy,),
        f=str(out_path),
        input_names=["cube"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        external_data=False,
        do_constant_folding=False,
        optimize=False,
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=None,
        export_params=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.height < 1 or args.width < 1 or args.bands < 1:
        raise SystemExit("--height, --width, and --bands must be >= 1")
    if args.opset < 1:
        raise SystemExit(f"--opset must be >= 1, got {args.opset}")
    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out)
    model, meta = load_mininet(checkpoint, args.bands)
    cfg = meta.get("cfg") or {}
    if (args.height, args.width) != (DEFAULT_HEIGHT, DEFAULT_WIDTH):
        print(
            f"warning: cnn_ds2 was scored at about {DEFAULT_HEIGHT}x{DEFAULT_WIDTH} "
            f"(downsample 2 on 256). This export uses {args.height}x{args.width}.",
            file=sys.stderr,
        )
    if args.height % 16 != 0 or args.width % 16 != 0:
        print(
            "warning: four stride-2 stages divide H and W by 16. "
            "The ONNX trace freezes the shape check in MiniNetSeg.forward, "
            "so export the exact size you will score. Do not edit cnn.py.",
            file=sys.stderr,
        )
    downsample = cfg.get("downsample")
    if downsample not in (None, 2):
        print(
            f"warning: checkpoint cfg.downsample is {downsample}. "
            "The scored cnn_ds2 run used downsample 2.",
            file=sys.stderr,
        )
    module: nn.Module = model
    if args.nhwc:
        print(
            "warning: --nhwc is on. Input and logits are NHWC. "
            "This layout is unconfirmed against Jeong's announcement 5928848 scripts. "
            "Default is NCHW, which is MiniNetSeg.forward.",
            file=sys.stderr,
        )
        module = _NhwcProbe(model)
    shape = input_shape(args.bands, args.height, args.width, nhwc=args.nhwc)
    dummy = torch.zeros(*shape, dtype=torch.float32)
    export_graph(module, dummy, out_path, args.opset)
    num_classes = int(model.num_classes)
    expected_out = output_shape(num_classes, args.height, args.width, nhwc=args.nhwc)
    print(
        f"wrote {out_path}  input {shape} cube float32  output {expected_out} logits float32  "
        f"opset {args.opset}  layout {'nhwc' if args.nhwc else 'nchw'}  "
        f"classes {num_classes}"
    )
    print(
        "preprocess assumed: uint16 tiff / 65535, then prepare_sample "
        "train=False crop_size=None downsample=2, batch axis added. No NHWC transpose "
        "unless --nhwc was passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
