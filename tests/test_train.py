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
    restored = MiniNetSeg()
    restored.load_state_dict(checkpoint["model"])
    restored.eval()
    with torch.no_grad():
        logits = restored(torch.zeros(1, 224, 16, 16))
    assert logits.shape == (1, 7, 16, 16)
