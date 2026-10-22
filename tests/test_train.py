"""Harness checks on a tiny synthetic dataset. No real SpectralWaste files."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw.train import config_from_args, main, prepare_sample, resolve_device


def _write_split(root, split, n_images, shape=(224, 32, 32)):
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    channels, height, width = shape
    for index in range(n_images):
        cube = np.zeros(shape, dtype=np.uint16)
        cube[0, 0, 0] = 0
        cube[0, 0, -1] = 65535
        cube[0, 0, 1] = 1000 * (index + 1)
        tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", cube, photometric="minisblack")
        mask = np.zeros((height, width), dtype=np.uint8)
        for cls in range(7):
            mask[:, cls::7] = cls
        Image.fromarray(mask, mode="L").save(labels / f"{split}_{index:02d}.png")


def _synthetic_root(root):
    # Enough images for --smoke: 4 train, 2 val, and a test split the harness ignores.
    _write_split(root, "train", 4)
    _write_split(root, "val", 2)
    _write_split(root, "test", 1)


def test_prepare_sample_keeps_cube_and_mask_aligned():
    cube = torch.zeros(4, 16, 16)
    mask = torch.zeros(16, 16, dtype=torch.int64)
    cube[0, 0, 0] = 1.0
    mask[0, 0] = 3
    cube[0, 0, -1] = 0.25
    mask[0, -1] = 5

    down_cube, down_mask = prepare_sample(
        cube, mask, train=False, crop_size=None, downsample=2
    )
    assert down_cube.shape == (4, 8, 8)
    assert down_mask.shape == (8, 8)
    # Stride keeps the origin pixel, so the markers still sit on the same corner.
    assert down_cube[0, 0, 0].item() == 1.0
    assert down_mask[0, 0].item() == 3

    torch.manual_seed(0)
    seen_left = set()
    for _ in range(20):
        flipped_cube, flipped_mask = prepare_sample(
            cube, mask, train=True, crop_size=None, downsample=1
        )
        assert flipped_cube.shape == cube.shape
        assert flipped_mask.shape == mask.shape
        # The cube's 1.0 and the mask's 3 move as a pair.
        ones = (flipped_cube[0] == 1.0).nonzero()
        threes = (flipped_mask == 3).nonzero()
        assert ones.tolist() == threes.tolist()
        seen_left.add(int(ones[0, 1]))
    assert seen_left == {0, 15}


def test_center_crop_after_downsample():
    cube = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    mask = torch.arange(16, dtype=torch.int64).reshape(4, 4)
    cropped_cube, cropped_mask = prepare_sample(
        cube, mask, train=False, crop_size=2, downsample=1
    )
    assert cropped_cube.shape == (1, 2, 2)
    assert cropped_mask.shape == (2, 2)
    # Center of a 4x4 is rows 1:3, cols 1:3.
    assert torch.equal(cropped_mask, mask[1:3, 1:3])


def test_resolve_device_cpu_and_missing_cuda():
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


def test_smoke_flag_shortens_the_run():
    from sw.train import _build_parser

    cfg = config_from_args(_build_parser().parse_args(["--smoke", "--device", "cpu"]))
    assert cfg.epochs == 1
    assert cfg.batch_size == 2
    assert cfg.max_train_steps == 2
    assert cfg.max_train_images == 4
    assert cfg.max_val_images == 2
    assert cfg.downsample == 4
    assert cfg.num_workers == 0
    assert cfg.weight_scheme == "paper"
    assert cfg.out_dir == "runs/smoke"
    # An explicit stride wins over the smoke default of 4.
    overridden = config_from_args(
        _build_parser().parse_args(["--smoke", "--downsample", "1", "--out-dir", "runs/custom"])
    )
    assert overridden.downsample == 1
    assert overridden.out_dir == "runs/custom"
    assert overridden.max_train_steps == 2
    explicit = config_from_args(
        _build_parser().parse_args(
            [
                "--max-train-images",
                "3",
                "--max-val-images",
                "1",
                "--max-train-steps",
                "5",
                "--no-cosine",
                "--amp",
                "--resume",
                "runs/cnn/last.pt",
            ]
        )
    )
    assert explicit.max_train_images == 3
    assert explicit.max_val_images == 1
    assert explicit.max_train_steps == 5
    assert explicit.cosine is False
    assert explicit.amp is True
    assert explicit.resume == "runs/cnn/last.pt"
    # An explicit step limit wins over the smoke default of 2.
    smoke_steps = config_from_args(
        _build_parser().parse_args(["--smoke", "--max-train-steps", "7"])
    )
    assert smoke_steps.max_train_steps == 7
    assert smoke_steps.max_train_images == 4
    assert smoke_steps.cosine is True


def test_harness_smoke_on_synthetic_cubes(tmp_path, capsys):
    root = tmp_path / "data"
    _synthetic_root(root)
    out = tmp_path / "run"
    from sw.models.cnn import MiniNetSeg

    result = main(
        [
            "--smoke",
            "--data-root",
            str(root),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--downsample",
            "2",
            "--seed",
            "0",
        ]
    )
    printed = capsys.readouterr().out
    assert "parameters:" in printed
    assert "class weights (paper):" in printed
    assert "Limited run" in printed
    assert "val_fg_miou" in printed
    assert result["epochs_ran"] == 1
    assert result["best_epoch"] == 1
    assert result["param_count"] >= 400_000
    assert math.isfinite(result["best_val_fg_miou"])
    assert (out / "best.pt").is_file()

    with (out / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 1
    assert rows[0]["steps"] == 2
    assert rows[0]["per_class_iou"]["background"] is not None
    assert "filament" in rows[0]["per_class_iou"]

    with (out / "metrics.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 1
    assert table[0]["epoch"] == "1"
    assert "iou_filament" in table[0]

    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["class_weight_scheme"] == "paper"
    assert len(checkpoint["class_weights"]) == 7
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "epoch", "best_val_fg_miou"):
        assert key in last
    assert last["epoch"] == 1
    assert last["scheduler"] is not None
    restored = MiniNetSeg()
    restored.load_state_dict(checkpoint["model"])
    restored.eval()
    with torch.no_grad():
        logits = restored(torch.zeros(1, 224, 16, 16))
    assert logits.shape == (1, 7, 16, 16)


def _adam_step(checkpoint) -> int:
    steps = []
    for slot in checkpoint["optimizer"]["state"].values():
        if "step" not in slot:
            continue
        value = slot["step"]
        steps.append(int(value.item() if torch.is_tensor(value) else value))
    assert steps
    return max(steps)


def test_resume_restores_optimizer_and_appends_the_log(tmp_path, capsys):
    root = tmp_path / "data"
    _synthetic_root(root)
    out = tmp_path / "run"
    common = [
        "--data-root",
        str(root),
        "--out-dir",
        str(out),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--downsample",
        "4",
        "--num-workers",
        "0",
        "--max-train-images",
        "4",
        "--max-val-images",
        "2",
        "--max-train-steps",
        "1",
        "--seed",
        "0",
    ]
    first = main([*common, "--epochs", "1"])
    assert first["epochs_ran"] == 1
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 1
    assert last["scheduler"]["last_epoch"] == 1
    assert last["best_val_fg_miou"] is not None
    assert _adam_step(last) == 1
    with (out / "metrics.jsonl").open(encoding="utf-8") as handle:
        first_rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["epoch"] for row in first_rows] == [1]

    capsys.readouterr()
    second = main([*common, "--epochs", "2", "--resume", str(out / "last.pt")])
    printed = capsys.readouterr().out
    assert "resumed" in printed
    assert "next epoch 2" in printed
    assert second["epochs_ran"] == 2
    last_again = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last_again["epoch"] == 2
    assert last_again["scheduler"]["last_epoch"] == 2
    # A fresh optimizer would be back at step 1. Step 2 means the state was restored.
    assert _adam_step(last_again) == 2
    with (out / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["train_loss"] == first_rows[0]["train_loss"]
    with (out / "metrics.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert [row["epoch"] for row in table] == ["1", "2"]


def test_size_one_tail_is_logged_and_cpu_amp_is_skipped(tmp_path, capsys):
    root = tmp_path / "data"
    _write_split(root, "train", 5)
    _write_split(root, "val", 2)
    from sw.train import TrainConfig, train

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(224, 7, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    train(
        TinyHead(),
        TrainConfig(
            data_root=str(root),
            out_dir=str(tmp_path / "run"),
            epochs=1,
            batch_size=4,
            num_workers=0,
            device="cpu",
            downsample=4,
            cosine=False,
            amp=True,
        ),
    )
    printed = capsys.readouterr().out
    assert "train loader drop_last: leaving out 1 image" in printed
    assert "full precision" in printed
    last = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert last["scheduler"] is None
    assert last["epoch"] == 1
    with (tmp_path / "run" / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows[0]["steps"] == 1


def test_batch_size_one_is_rejected():
    from sw.train import TrainConfig, train

    with pytest.raises(ValueError, match="BatchNorm"):
        train(torch.nn.Conv2d(1, 1, 1), TrainConfig(batch_size=1))
