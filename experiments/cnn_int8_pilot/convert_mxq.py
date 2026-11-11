#!/usr/bin/env python3
"""Compile the MiniNetSeg ONNX file with the Lab 2 Qubee flags.

Documented compiler path from Lab 2 on this course. This script does not
know, and does not claim, the compiler version currently installed on
Computer 3:

* Docker image: ``mobilint/qbcompiler:0.12-cpu-ubuntu22.04``
* Wheel: ``qubee-0.12.0.0+aries2-py3-none-any.whl``
* Call ``mxq_compile`` with ``quantization_mode=2``, ``quantization_method=1``,
  ``percentile=0.999``, ``backend="onnx"``.
* The 0.12 quick start pairs those flags with ``quantization_output=0``.
  This script passes that too.
  https://docs.mobilint.com/compiler/v0.12.0.0/en/quick_start.html
* Runtime outside Docker, for the score step, is maccel-0.30.1.
  This file does not import maccel and does not book the NPU.

If the NPU foreground mIoU drops more than 0.1 versus FP32, recompile with
``--percentile 0.9999``. That fallback is not the default.

Calibration is training cubes only (about 100–200), preprocessed the same
way as ``export_onnx.py``. Never val or test. Not RGB JPEGs. Not uint8.

``--materialize-calib`` writes float32 ``.npy`` files. The value scale and
the stride come from the loader and ``prepare_sample``. The container is the
0.12 ``calib_data_path`` directory of pre-processed NumPy samples (compiler
docs, not a guess from the tiff loader). The array rank defaults to the ONNX
input, NCHW ``(1, bands, height, width)``.

TODO(layout): the Qubee 0.11 quick start says a compile without CPU offload
expects calibration arrays shaped ``(H, W, C)``. That note is not confirmed
against Jeong's announcement 5928848. ``--nhwc`` stays off unless you pass
it. Do not switch the rank silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Lab 2 flags. Names match the Qubee 0.12 quick start. Not a Computer 3 probe.
QUANTIZATION_MODE = 2
QUANTIZATION_METHOD = 1
PERCENTILE = 0.999
PERCENTILE_IF_DROP = 0.9999
METRIC_DROP_THRESHOLD = 0.1
BACKEND = "onnx"
QUANTIZATION_OUTPUT = 0
DEVICE = "cpu"
CALIB_COUNT = 150
CALIB_MIN = 100
CALIB_MAX = 200
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_BANDS = 224
DEFAULT_DOWNSAMPLE = 2

HELD_OUT_SPLITS = frozenset({"val", "test"})
# Raw cubes and RGB images are not calibration samples.
REJECTED_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff",
        ".pt",
        ".pth",
        ".onnx",
        ".mxq",
        ".zip",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize MiniNetSeg calibration cubes and/or call mxq_compile "
            "with the Lab 2 Qubee flags. Does not book the NPU."
        )
    )
    parser.add_argument("--onnx", default=None, help="Static ONNX file from export_onnx.py.")
    parser.add_argument("--out", default=None, help="Destination .mxq path. Do not commit it.")
    parser.add_argument(
        "--calib-dir",
        required=True,
        help="Directory of preprocessed training-cube .npy files. Not val/test. Not JPEGs.",
    )
    parser.add_argument(
        "--materialize-calib",
        action="store_true",
        help=(
            "Write calibration .npy from the train split (SW_ROOT or --data-root). "
            "Stops before compile unless --compile is also set."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Call mxq_compile. Implied when --materialize-calib is omitted.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset folder with hyper/ and labels_hyper_lt/. Default: SW_ROOT.",
    )
    parser.add_argument("--calib-count", type=int, default=CALIB_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument(
        "--allow-short-calib",
        action="store_true",
        help="Allow a calibration count outside 100–200. For a synthetic wiring check only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing .npy files in --calib-dir when materializing.",
    )
    parser.add_argument(
        "--nhwc",
        action="store_true",
        help=(
            "Write and expect NHWC calibration arrays (1, H, W, C). Default off. "
            "Unconfirmed against Jeong's announcement 5928848 scripts."
        ),
    )
    parser.add_argument("--quantization-mode", type=int, default=QUANTIZATION_MODE)
    parser.add_argument("--quantization-method", type=int, default=QUANTIZATION_METHOD)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--device", default=DEVICE)
    return parser


def path_uses_heldout_split(path: Path) -> bool:
    """True when a path component is the val or test split directory."""
    return any(part.lower() in HELD_OUT_SPLITS for part in Path(path).parts)


def preprocess_train_cube(
    cube: torch.Tensor,
    mask: torch.Tensor,
    *,
    downsample: int,
    height: int,
    width: int,
    bands: int,
    nhwc: bool,
) -> torch.Tensor:
    """Inference geometry for one training cube.

    ``cube`` is the loader tensor: float32 ``(C, H, W)`` in ``[0, 1]``,
    already divided by 65535 in ``SpectralWasteDataset.__getitem__``.
    This function does not rescale and does not read a file format of its own.

    It calls ``prepare_sample(..., train=False, crop_size=None)``, which is
    the cnn_ds2 validation path (stride ``downsample``, no flip, no crop).
    The returned tensor is float32 batch-1, NCHW unless ``nhwc`` is set.
    """
    from sw.train import prepare_sample

    if downsample < 1:
        raise SystemExit(f"downsample must be >= 1, got {downsample}")
    prepared, _mask = prepare_sample(
        cube,
        mask,
        train=False,
        crop_size=None,
        downsample=downsample,
    )
    if prepared.ndim != 3 or int(prepared.shape[0]) != bands:
        raise SystemExit(
            f"preprocessed cube has shape {tuple(prepared.shape)}; expected ({bands}, H, W). "
            "The loader keeps every band unless bands= is set. This probe uses --bands."
        )
    if prepared.dtype != torch.float32:
        raise SystemExit(f"preprocessed cube dtype is {prepared.dtype}; expected float32")
    finite = torch.isfinite(prepared)
    if not bool(finite.all()):
        raise SystemExit("preprocessed cube has a non-finite value")
    peak = float(prepared.max())
    floor = float(prepared.min())
    if floor < -1e-4 or peak > 1.0 + 1e-4:
        raise SystemExit(
            f"preprocessed cube range is [{floor}, {peak}]. "
            "The loader divides uint16 by 65535 before prepare_sample, so values sit in [0, 1]. "
            "Refusing a raw uint16 cube or a guessed scale."
        )
    if tuple(prepared.shape[-2:]) != (height, width):
        raise SystemExit(
            f"preprocessed spatial size is {tuple(prepared.shape[-2:])}, expected {(height, width)}. "
            "cnn_ds2 used downsample 2 on 256 cubes, which is 128. "
            "Do not point this probe at 256 if that is not the array the checkpoint saw."
        )
    batched = prepared.unsqueeze(0).contiguous()
    if nhwc:
        batched = batched.permute(0, 2, 3, 1).contiguous()
    return batched


def expected_calib_shape(bands: int, height: int, width: int, *, nhwc: bool) -> tuple[int, int, int, int]:
    if nhwc:
        return (1, height, width, bands)
    return (1, bands, height, width)


def _require_calib_count(count: int, allow_short: bool, what: str) -> None:
    if count < 1:
        raise SystemExit(f"{what} is empty. Calibration needs training cubes, about 100–200.")
    if allow_short:
        return
    if not (CALIB_MIN <= count <= CALIB_MAX):
        raise SystemExit(
            f"{what} has {count} samples; expected about {CALIB_MIN}–{CALIB_MAX} training cubes. "
            "Pass --allow-short-calib only for a synthetic wiring check."
        )


def materialize_calib(args: argparse.Namespace) -> int:
    """Write train-split calibration ``.npy`` files. Does not touch val or test."""
    from sw.data import SpectralWasteDataset

    dest = Path(args.calib_dir)
    if path_uses_heldout_split(dest):
        raise SystemExit(
            f"refusing to write calibration data under {dest}. "
            "val and test are held out. Calibrate on the train split only."
        )
    if args.nhwc:
        print(
            "warning: --nhwc calibration arrays are on. "
            "Unconfirmed against Jeong's announcement 5928848 scripts.",
            file=sys.stderr,
        )
    try:
        dataset = SpectralWasteDataset(
            args.data_root,
            split="train",
            random_flip=False,
            check_split_size=False,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"cannot read training cubes: {exc}") from exc
    if args.calib_count < 1:
        raise SystemExit("--calib-count must be >= 1")
    if len(dataset) < 1:
        raise SystemExit("train split is empty")
    if len(dataset) < args.calib_count and not args.allow_short_calib:
        raise SystemExit(
            f"train split has {len(dataset)} cubes; --calib-count is {args.calib_count}. "
            f"Use about {CALIB_MIN}–{CALIB_MAX} training cubes, or --allow-short-calib for a wiring check."
        )
    count = min(args.calib_count, len(dataset))
    _require_calib_count(count, args.allow_short_calib, "calibration set")
    dest.mkdir(parents=True, exist_ok=True)
    existing = [path for path in dest.iterdir()]
    if existing and not args.force:
        raise SystemExit(
            f"{dest} is not empty. Pass --force to replace calibration .npy files. "
            "This script will not mix a new set into an old directory."
        )
    if existing and args.force:
        strangers = [path.name for path in existing if path.suffix.lower() != ".npy" or path.is_dir()]
        if strangers:
            raise SystemExit(
                f"{dest} contains non-npy files ({strangers[:5]}). "
                "Refusing to delete them. Point --calib-dir at an empty directory."
            )
        for path in existing:
            path.unlink()
    generator = torch.Generator().manual_seed(args.seed)
    chosen = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    shape = expected_calib_shape(args.bands, args.height, args.width, nhwc=args.nhwc)
    for index in chosen:
        cube, mask = dataset[index]
        array = preprocess_train_cube(
            cube,
            mask,
            downsample=args.downsample,
            height=args.height,
            width=args.width,
            bands=args.bands,
            nhwc=args.nhwc,
        )
        if tuple(array.shape) != shape:
            raise SystemExit(f"internal shape {tuple(array.shape)} != {shape}")
        stem = dataset.pairs[index][0].stem
        np.save(dest / f"{stem}.npy", array.numpy())
    print(
        f"wrote {count} float32 npy files to {dest}  shape {shape}  "
        f"layout {'nhwc' if args.nhwc else 'nchw'}  split train  seed {args.seed}"
    )
    return count


def list_calib_npy(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(
            f"calibration directory not found: {directory}\n"
            "Build it with --materialize-calib from the train split, "
            "or pass a directory of preprocessed float32 .npy cubes. "
            "Do not pass RGB JPEGs or the raw hyper/ tree."
        )
    if path_uses_heldout_split(directory):
        raise SystemExit(
            f"refusing calibration path {directory}. "
            "A path component is val or test. Calibrate on training cubes only."
        )
    children = list(directory.iterdir())
    if any(path.is_dir() for path in children):
        raise SystemExit(
            f"{directory} contains a subdirectory. "
            "Pass a flat directory of .npy files, not hyper/train or a split tree."
        )
    rejected = [path.name for path in children if path.suffix.lower() in REJECTED_SUFFIXES]
    if rejected:
        raise SystemExit(
            f"{directory} contains raw cubes or RGB images ({rejected[:5]}). "
            "Calibration samples are preprocessed float32 .npy, not JPEGs and not uint16 tiffs."
        )
    other = [path.name for path in children if path.is_file() and path.suffix.lower() != ".npy"]
    if other:
        raise SystemExit(
            f"{directory} has files that are not .npy ({other[:5]}). "
            "Qubee 0.12 also accepts a separate .txt/.json manifest of npy paths. "
            "This probe does not parse a manifest. Leave a flat .npy directory, "
            "or extend the reader after checking the compiler docs on the machine."
        )
    files = sorted(path for path in children if path.suffix.lower() == ".npy")
    if not files:
        raise SystemExit(f"{directory} has no .npy files.")
    return files


def validate_calib_dir(args: argparse.Namespace) -> list[Path]:
    files = list_calib_npy(Path(args.calib_dir))
    _require_calib_count(len(files), args.allow_short_calib, "calibration directory")
    sample = np.load(files[0])
    expected = expected_calib_shape(args.bands, args.height, args.width, nhwc=args.nhwc)
    if sample.dtype != np.float32:
        raise SystemExit(
            f"{files[0].name} dtype is {sample.dtype}. Expected float32. "
            "Do not invent a uint8 cube contract."
        )
    if tuple(sample.shape) != expected:
        raise SystemExit(
            f"{files[0].name} shape {tuple(sample.shape)} != {expected}. "
            "A collapsed metric is usually this layout (or the calibration split), not the weights. "
            "Default is NCHW (1, bands, height, width). "
            "--nhwc is unconfirmed against Jeong's announcement 5928848."
        )
    return files


def mxq_call_kwargs(args: argparse.Namespace) -> dict:
    """Arguments for ``qubee.mxq_compile``. Mirrors the 0.12 quick-start call.

    ``sample_dtype="float"`` is the 0.12 default for saved debug samples.
    It is not a uint8 input contract. The score step should call
    ``maccel.Model.infer`` on the same float32 array the ONNX file takes.
    ``use_random_calib`` stays false. ``cpu_offload`` stays false.
    """
    if not args.onnx or not args.out:
        raise SystemExit("compile needs --onnx and --out")
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(
            f"ONNX file not found: {onnx_path}\n"
            "Run experiments/cnn_int8_pilot/export_onnx.py first. "
            "Do not point --onnx at a JPEG or a checkpoint."
        )
    if args.backend != BACKEND:
        print(
            f"warning: --backend {args.backend} is not the Lab 2 onnx backend.",
            file=sys.stderr,
        )
    if args.quantization_mode != QUANTIZATION_MODE or args.quantization_method != QUANTIZATION_METHOD:
        print(
            "warning: quantization mode/method differ from the Lab 2 pair "
            f"(mode {QUANTIZATION_MODE}, method {QUANTIZATION_METHOD}).",
            file=sys.stderr,
        )
    if args.percentile != PERCENTILE:
        print(
            f"warning: percentile {args.percentile} is not the default {PERCENTILE}. "
            f"Use {PERCENTILE_IF_DROP} only after the NPU foreground mIoU drops "
            f"more than {METRIC_DROP_THRESHOLD} versus FP32.",
            file=sys.stderr,
        )
    return {
        "model": str(onnx_path),
        "calib_data_path": str(Path(args.calib_dir)),
        "save_path": str(Path(args.out)),
        "backend": args.backend,
        "device": args.device,
        "quantization_method": args.quantization_method,
        "quantization_mode": args.quantization_mode,
        "percentile": args.percentile,
        "quantization_output": QUANTIZATION_OUTPUT,
        "sample_dtype": "float",
        "use_random_calib": False,
        "cpu_offload": False,
        "save_sample": False,
    }


def resolve_mxq_compile():
    """Import ``mxq_compile`` from the Lab 2 wheel. Absent here is a normal result."""
    try:
        from qubee import mxq_compile
    except ImportError as exc:
        raise SystemExit(
            "qubee is not installed in this environment, so mxq_compile was not called.\n"
            "Compile on Computer 3 inside Docker image "
            "mobilint/qbcompiler:0.12-cpu-ubuntu22.04, using wheel "
            "qubee-0.12.0.0+aries2-py3-none-any.whl.\n"
            "Those strings are the Lab 2 notes. They are not a confirmation of "
            "the compiler version installed on Computer 3 today.\n"
            "This probe does not book or touch the NPU.\n"
            f"import error: {exc}"
        ) from exc
    return mxq_compile


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    do_materialize = bool(args.materialize_calib)
    do_compile = bool(args.compile) or not do_materialize
    if do_materialize:
        materialize_calib(args)
    if not do_compile:
        print("materialize only; mxq_compile was not called.")
        return 0
    validate_calib_dir(args)
    kwargs = mxq_call_kwargs(args)
    compile_fn = resolve_mxq_compile()
    print(
        "calling mxq_compile  "
        f"mode {kwargs['quantization_mode']}  method {kwargs['quantization_method']}  "
        f"percentile {kwargs['percentile']}  backend {kwargs['backend']}  "
        f"device {kwargs['device']}  quantization_output {kwargs['quantization_output']}"
    )
    compile_fn(**kwargs)
    print(f"wrote {kwargs['save_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
