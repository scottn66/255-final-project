"""INT8 probe checks that do not need the SpectralWaste download or the NPU.

The real cnn_ds2 checkpoint stays on Drive. Export tests build a random
MiniNetSeg state dict in a temp file and delete it with the temp directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "experiments" / "cnn_int8_pilot"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))

import convert_mxq
import export_onnx


def test_export_defaults_are_the_cnn_ds2_contract():
    args = export_onnx.build_parser().parse_args(
        ["--checkpoint", "best.pt", "--out", "model.onnx"]
    )
    assert args.height == 128
    assert args.width == 128
    assert args.bands == 224
    assert args.opset == 13
    assert args.nhwc is False
    assert export_onnx.input_shape(224, 128, 128, nhwc=False) == (1, 224, 128, 128)
    assert export_onnx.output_shape(7, 128, 128, nhwc=False) == (1, 7, 128, 128)
    assert export_onnx.input_shape(224, 128, 128, nhwc=True) == (1, 128, 128, 224)


def test_missing_checkpoint_names_the_drive_file(tmp_path):
    missing = tmp_path / "best.pt"
    with pytest.raises(SystemExit, match="MyDrive/sw_runs/cnn_ds2/best.pt"):
        export_onnx.main(
            ["--checkpoint", str(missing), "--out", str(tmp_path / "model.onnx")]
        )
    assert not (tmp_path / "model.onnx").exists()


def test_transformer_checkpoint_is_refused(tmp_path):
    path = tmp_path / "transformer.pt"
    torch.save(
        {
            "model": {"tokenizer.proj.weight": torch.zeros(1)},
            "cfg": {"model": "transformer"},
        },
        path,
    )
    with pytest.raises(SystemExit, match="MiniNetSeg only"):
        export_onnx.main(
            ["--checkpoint", str(path), "--out", str(tmp_path / "model.onnx")]
        )


def _fake_checkpoint(path: Path) -> None:
    model = export_onnx.MiniNetSeg()
    torch.save({"model": model.state_dict(), "cfg": {"model": "cnn", "downsample": 2}}, path)


def _onnx_io(path: Path):
    onnx = pytest.importorskip("onnx")
    model = onnx.load(path)
    def dims(value):
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    return model, dims(model.graph.input[0]), dims(model.graph.output[0])


def test_random_checkpoint_exports_static_nchw(tmp_path):
    checkpoint = tmp_path / "best.pt"
    _fake_checkpoint(checkpoint)
    out = tmp_path / "mininet.onnx"
    assert export_onnx.main(
        ["--checkpoint", str(checkpoint), "--out", str(out)]
    ) == 0
    model, inputs, outputs = _onnx_io(out)
    assert inputs == [1, 224, 128, 128]
    assert outputs == [1, 7, 128, 128]
    assert model.graph.input[0].name == "cube"
    assert model.graph.output[0].name == "logits"
    # onnx TensorProto.FLOAT is 1. The file is not a uint8 cube.
    assert model.graph.input[0].type.tensor_type.elem_type == 1
    assert model.graph.output[0].type.tensor_type.elem_type == 1
    ops = {node.op_type for node in model.graph.node}
    # Risky ops stay in the graph. This probe does not fuse them away.
    assert "Conv" in ops
    assert "BatchNormalization" in ops
    assert "Relu" in ops
    assert "Concat" in ops
    assert "Add" in ops
    assert "Resize" in ops or "Upsample" in ops
    grouped = False
    dilated = False
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        for attr in node.attribute:
            if attr.name == "group" and attr.i > 1:
                grouped = True
            if attr.name == "dilations" and list(attr.ints) not in ([], [1, 1]):
                dilated = True
    assert grouped
    assert dilated
    assert not (tmp_path / "mininet.onnx.data").exists()


def test_nhwc_export_is_opt_in(tmp_path):
    checkpoint = tmp_path / "best.pt"
    _fake_checkpoint(checkpoint)
    out = tmp_path / "nhwc.onnx"
    assert export_onnx.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out),
            "--nhwc",
            "--height",
            "32",
            "--width",
            "32",
        ]
    ) == 0
    _model, inputs, outputs = _onnx_io(out)
    assert inputs == [1, 32, 32, 224]
    assert outputs == [1, 32, 32, 7]


def test_real_drive_checkpoint_when_the_env_points_at_it(tmp_path):
    raw = os.environ.get("CNN_DS2_CHECKPOINT", "")
    if not raw or not Path(raw).is_file():
        pytest.xfail(
            "cnn_ds2 best.pt is on Drive (MyDrive/sw_runs/cnn_ds2/best.pt), not in git"
        )
    out = tmp_path / "from_drive.onnx"
    assert export_onnx.main(["--checkpoint", raw, "--out", str(out)]) == 0


def test_qubee_flag_constants_match_lab2():
    assert convert_mxq.QUANTIZATION_MODE == 2
    assert convert_mxq.QUANTIZATION_METHOD == 1
    assert convert_mxq.PERCENTILE == 0.999
    assert convert_mxq.PERCENTILE_IF_DROP == 0.9999
    assert convert_mxq.METRIC_DROP_THRESHOLD == 0.1
    assert convert_mxq.BACKEND == "onnx"
    assert convert_mxq.QUANTIZATION_OUTPUT == 0
    assert convert_mxq.DEVICE == "cpu"
    args = convert_mxq.build_parser().parse_args(
        ["--onnx", "m.onnx", "--calib-dir", "calib", "--out", "m.mxq"]
    )
    assert args.quantization_mode == 2
    assert args.quantization_method == 1
    assert args.percentile == 0.999
    assert args.backend == "onnx"
    assert args.device == "cpu"
    assert args.nhwc is False
    assert args.height == 128 and args.width == 128 and args.bands == 224
    assert args.downsample == 2
    assert args.calib_count == 150
    assert args.materialize_calib is False


def test_compile_kwargs_keep_random_calib_off(tmp_path, monkeypatch):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"not-read")
    calib = tmp_path / "calib"
    calib.mkdir()
    array = np.zeros((1, 4, 8, 8), dtype=np.float32)
    np.save(calib / "train_00.npy", array)
    captured = {}

    def fake_resolve():
        def fake_compile(**kwargs):
            captured.update(kwargs)

        return fake_compile

    monkeypatch.setattr(convert_mxq, "resolve_mxq_compile", fake_resolve)
    assert convert_mxq.main(
        [
            "--onnx",
            str(onnx_path),
            "--calib-dir",
            str(calib),
            "--out",
            str(tmp_path / "m.mxq"),
            "--bands",
            "4",
            "--height",
            "8",
            "--width",
            "8",
            "--allow-short-calib",
        ]
    ) == 0
    assert captured["quantization_mode"] == 2
    assert captured["quantization_method"] == 1
    assert captured["percentile"] == 0.999
    assert captured["quantization_output"] == 0
    assert captured["backend"] == "onnx"
    assert captured["device"] == "cpu"
    assert captured["sample_dtype"] == "float"
    assert captured["use_random_calib"] is False
    assert captured["cpu_offload"] is False
    assert captured["model"] == str(onnx_path)
    assert captured["calib_data_path"] == str(calib)


def test_missing_qubee_does_not_compile(tmp_path):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"not-read")
    calib = tmp_path / "calib"
    calib.mkdir()
    np.save(calib / "train_00.npy", np.zeros((1, 224, 128, 128), dtype=np.float32))
    with pytest.raises(SystemExit, match="mxq_compile was not called"):
        convert_mxq.resolve_mxq_compile()
    with pytest.raises(SystemExit, match="does not book or touch the NPU"):
        convert_mxq.main(
            [
                "--onnx",
                str(onnx_path),
                "--calib-dir",
                str(calib),
                "--out",
                str(tmp_path / "m.mxq"),
                "--allow-short-calib",
            ]
        )


def test_calib_dir_rejects_jpeg_tiff_and_val(tmp_path):
    jpeg_dir = tmp_path / "jpeg"
    jpeg_dir.mkdir()
    (jpeg_dir / "frame.jpg").write_bytes(b"nope")
    with pytest.raises(SystemExit, match="RGB"):
        convert_mxq.list_calib_npy(jpeg_dir)

    tiff_dir = tmp_path / "tiffs"
    tiff_dir.mkdir()
    (tiff_dir / "cube.tiff").write_bytes(b"nope")
    with pytest.raises(SystemExit, match="tiff"):
        convert_mxq.list_calib_npy(tiff_dir)

    val_dir = tmp_path / "hyper" / "val"
    val_dir.mkdir(parents=True)
    np.save(val_dir / "a.npy", np.zeros((1, 224, 128, 128), dtype=np.float32))
    with pytest.raises(SystemExit, match="val"):
        convert_mxq.list_calib_npy(val_dir)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no .npy"):
        convert_mxq.list_calib_npy(empty)


def test_preprocess_is_stride_scale_and_nchw():
    cube = torch.zeros(4, 16, 16)
    cube[0, 0, 0] = 1.0
    cube[1, 2, 4] = 0.5
    mask = torch.zeros(16, 16, dtype=torch.int64)
    out = convert_mxq.preprocess_train_cube(
        cube,
        mask,
        downsample=2,
        height=8,
        width=8,
        bands=4,
        nhwc=False,
    )
    assert out.shape == (1, 4, 8, 8)
    assert out.dtype == torch.float32
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[0, 1, 1, 2].item() == 0.5
    nhwc = convert_mxq.preprocess_train_cube(
        cube,
        mask,
        downsample=2,
        height=8,
        width=8,
        bands=4,
        nhwc=True,
    )
    assert nhwc.shape == (1, 8, 8, 4)
    assert nhwc[0, 0, 0, 0].item() == 1.0


def test_preprocess_refuses_raw_uint16_scale():
    cube = torch.full((4, 8, 8), 1000.0)
    mask = torch.zeros(8, 8, dtype=torch.int64)
    with pytest.raises(SystemExit, match="65535"):
        convert_mxq.preprocess_train_cube(
            cube,
            mask,
            downsample=1,
            height=8,
            width=8,
            bands=4,
            nhwc=False,
        )


def _write_train_cube(root: Path, bands: int = 4, size: int = 16) -> None:
    import tifffile
    from PIL import Image

    hyper = root / "hyper" / "train"
    labels = root / "labels_hyper_lt" / "train"
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    cube = np.zeros((bands, size, size), dtype=np.uint16)
    cube[0, 0, 0] = 65535
    cube[1, 2, 4] = 32768
    tifffile.imwrite(hyper / "train_00.tiff", cube, photometric="minisblack")
    tifffile.imwrite(hyper / "train_01.tiff", cube, photometric="minisblack")
    mask = np.zeros((size, size), dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(labels / "train_00.png")
    Image.fromarray(mask, mode="L").save(labels / "train_01.png")
    # A val pair must not be read. If the writer ever walks val, the test below
    # still only checks the output directory, so keep val absent.


def test_materialize_writes_train_npy_only(tmp_path):
    root = tmp_path / "data"
    _write_train_cube(root)
    calib = tmp_path / "calib"
    assert convert_mxq.main(
        [
            "--materialize-calib",
            "--data-root",
            str(root),
            "--calib-dir",
            str(calib),
            "--calib-count",
            "2",
            "--bands",
            "4",
            "--height",
            "8",
            "--width",
            "8",
            "--allow-short-calib",
            "--seed",
            "0",
        ]
    ) == 0
    files = sorted(calib.glob("*.npy"))
    assert [path.name for path in files] == ["train_00.npy", "train_01.npy"]
    array = np.load(files[0])
    assert array.dtype == np.float32
    assert array.shape == (1, 4, 8, 8)
    assert array[0, 0, 0, 0] == pytest.approx(1.0)
    assert array.max() <= 1.0
    assert not any(calib.rglob("*.jpg"))
    assert not any(calib.rglob("*.tiff"))


def test_repo_does_not_contain_probe_weights():
    banned = []
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".venv", "site-packages", ".git"} for part in path.parts):
            continue
        if path.suffix.lower() in {".pt", ".pth", ".onnx", ".mxq", ".npy"}:
            banned.append(path)
    assert banned == []
