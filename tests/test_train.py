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


# --- Harness gates: no-aug, training-set metrics, batch rules (WP6 PR 7) ---


def _cfg(**overrides):
    from sw.train import TrainConfig

    base = dict(epochs=1, num_workers=0, device="cpu", max_val_images=1)
    base.update(overrides)
    return TrainConfig(**base)


def test_batch_size_defaults_and_rules(tmp_path):
    from sw.train import _build_parser, _check_cfg, train

    parse = lambda *flags: config_from_args(_build_parser().parse_args(list(flags)))
    assert parse().batch_size == 4
    assert parse("--smoke").batch_size == 2
    assert parse("--smoke", "--model", "transformer", "--batch-size", "1").batch_size == 1  # smoke keeps an explicit size
    assert parse("--batch-size", "8").batch_size == 8
    assert parse().augment is True
    assert parse("--no-aug").augment is False

    _check_cfg(_cfg(model="transformer", batch_size=1))  # no batch statistics: allowed
    with pytest.raises(ValueError, match="BatchNorm"):
        _check_cfg(_cfg(model="cnn", batch_size=1))
    with pytest.raises(ValueError, match=">= 1"):
        _check_cfg(_cfg(model="transformer", batch_size=0))

    # The model is checked too: a BatchNorm model is refused a batch of 1
    # even when the config says transformer.
    with pytest.raises(ValueError, match="BatchNorm"):
        train(
            torch.nn.Sequential(torch.nn.Conv2d(224, 7, 1), torch.nn.BatchNorm2d(7)),
            # An empty data folder: if the refusal ever moved after the data
            # load, this would fail on missing data, never train on SW_ROOT.
            _cfg(model="transformer", batch_size=1, out_dir=str(tmp_path / "bn"), data_root=str(tmp_path / "no-data")),
        )
    assert not (tmp_path / "bn").exists()  # refused before any folder is made


def test_transformer_crop_must_be_a_multiple_of_the_patch():
    from sw.train import _check_cfg

    with pytest.raises(ValueError, match="multiple of patch"):
        _check_cfg(_cfg(model="transformer", crop_size=30))
    _check_cfg(_cfg(model="transformer", crop_size=32))
    _check_cfg(_cfg(model="cnn", crop_size=30))  # the CNN has no patch grid


def _write_coded_split(root, split, n_images, seed, *, background_only=False):
    """Images whose labels and planned predictions are known pixel by pixel.

    Each image has its own random label map M. Channel 0 of the cube holds a
    prediction map P (P / 6), equal to M at about 60% of the pixels, so the
    expected IoUs are neither 0 nor 1. Channel 1 is a position ramp, so a
    crop that is off by one pixel, or a flip, shows. Returns [(P, M), ...].
    """
    rng = np.random.default_rng(seed)
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    pairs = []
    ramp = (np.arange(32 * 32).reshape(32, 32) * 60).astype(np.uint16)
    for index in range(n_images):
        mask = np.zeros((32, 32), np.int64) if background_only else rng.integers(0, 7, size=(32, 32))
        pred = np.where(rng.random((32, 32)) < 0.6, mask, rng.integers(0, 7, size=(32, 32)))
        if background_only:
            pred = np.zeros_like(mask)
        cube = np.zeros((224, 32, 32), dtype=np.uint16)
        cube[0] = np.round(pred / 6 * 65535).astype(np.uint16)
        cube[1] = ramp
        tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", cube, photometric="minisblack")
        Image.fromarray(mask.astype(np.uint8), mode="L").save(labels / f"{split}_{index:02d}.png")
        pairs.append((pred, mask))
    return pairs


class _CodedModel(torch.nn.Module):
    """Train mode: predicts the class coded in channel 0. Eval mode: predicts
    background everywhere. So a training metric read from an eval-mode or
    validation pass gives a different answer. Optionally sleeps per call and
    records every input, for the timing and augmentation tests."""

    def __init__(self, train_sleep=0.0, eval_sleep=0.0):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(()))  # something for AdamW to hold
        self.train_sleep, self.eval_sleep = train_sleep, eval_sleep
        self.train_inputs, self.eval_inputs = [], []

    def forward(self, x):
        import time

        if self.training:
            self.train_inputs.extend(image.detach().clone() for image in x)
            time.sleep(self.train_sleep)
            cls = torch.round(x[:, 0] * 6).long().clamp(0, 6)
        else:
            self.eval_inputs.extend(image.detach().clone() for image in x)
            time.sleep(self.eval_sleep)
            cls = torch.zeros(x.shape[0], *x.shape[2:], dtype=torch.long)
        logits = 10.0 * torch.nn.functional.one_hot(cls, 7).permute(0, 3, 1, 2).float()
        return logits + 0.0 * self.unused


def _expected(pairs, crop=None):
    from sw.metrics import confusion_matrix, fg_miou_from_cm

    cm = sum(confusion_matrix(p if crop is None else p[crop], m if crop is None else m[crop]) for p, m in pairs)
    return fg_miou_from_cm(cm)


def test_training_metrics_count_every_pixel_of_every_image(tmp_path):
    """The logged training metrics equal a count over every pixel of every
    training image, with masks aligned to the predictions, with or without
    flips (counts are flip-invariant only if cube and mask flip together),
    and when an epoch ends on --max-train-steps."""
    from sw.data import CLASS_NAMES
    from sw.train import train

    root = tmp_path / "data"
    pairs = _write_coded_split(root, "train", 4, seed=0)
    _write_coded_split(root, "val", 2, seed=1)
    score, iou = _expected(pairs)
    assert 0.2 < score < 0.9  # a non-trivial answer
    for name, extra in (("aug", {}), ("noaug", {"augment": False}), ("steps", {"max_train_steps": 2})):
        out = tmp_path / name
        train(_CodedModel(), _cfg(data_root=str(root), out_dir=str(out), epochs=2, batch_size=2, max_val_images=2, **extra))
        for line in (out / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["steps"] == 2
            assert row["train_fg_miou"] == pytest.approx(score, abs=1e-12), name
            for cls, value in zip(CLASS_NAMES, iou.tolist()):
                assert row["train_per_class_iou"][cls] == pytest.approx(value, abs=1e-12), (name, cls)
            assert row["val_fg_miou"] == 0.0  # the eval pass predicts background only

    # The CSV keeps its original columns, so a run started before this
    # change can still be resumed and appended to.
    with (tmp_path / "aug" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["epoch", "train_loss", "val_fg_miou", "steps", *[f"iou_{c}" for c in CLASS_NAMES]]


def test_training_score_is_null_without_foreground(tmp_path):
    from sw.train import train

    root = tmp_path / "data"
    _write_coded_split(root, "train", 2, seed=0, background_only=True)
    _write_coded_split(root, "val", 1, seed=1)
    train(_CodedModel(), _cfg(data_root=str(root), out_dir=str(tmp_path / "run"), batch_size=2))
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["train_fg_miou"] is None  # NaN in the matrix, null in the JSON
    assert row["train_per_class_iou"]["background"] == 1.0
    assert all(row["train_per_class_iou"][c] is None for c in ("film", "basket", "cardboard", "video_tape", "filament", "bag"))


def test_no_aug_shows_the_same_pixels_and_validation_is_never_augmented(tmp_path):
    from sw.data import SpectralWasteDataset
    from sw.train import train

    root = tmp_path / "data"
    pairs = _write_coded_split(root, "train", 4, seed=0)
    _write_coded_split(root, "val", 2, seed=1)
    base = SpectralWasteDataset(str(root), split="train", random_flip=False)
    raw = [torch.as_tensor(base[i][0]) for i in range(len(base))]
    val_base = SpectralWasteDataset(str(root), split="val", random_flip=False)
    val_raw = [torch.as_tensor(val_base[i][0]) for i in range(len(val_base))]

    def matches(cube, candidates):
        return any(cube.shape == c.shape and torch.equal(cube, c) for c in candidates)

    plain = _CodedModel()
    train(plain, _cfg(data_root=str(root), out_dir=str(tmp_path / "a"), epochs=3, batch_size=2, augment=False, max_val_images=2))
    assert len(plain.train_inputs) == 3 * 4 and all(matches(cube, raw) for cube in plain.train_inputs)

    # --no-aug with a crop takes the center crop, like validation: the
    # ramp channel pins the offset, and the logged metrics pin the mask crop.
    cropped = _CodedModel()
    out = tmp_path / "b"
    train(cropped, _cfg(data_root=str(root), out_dir=str(out), epochs=2, batch_size=2, crop_size=16, augment=False, max_val_images=2))
    centers = [cube[:, 8:24, 8:24] for cube in raw]
    assert len(cropped.train_inputs) == 2 * 4 and all(matches(cube, centers) for cube in cropped.train_inputs)
    score, _ = _expected(pairs, crop=(slice(8, 24), slice(8, 24)))
    for line in (out / "metrics.jsonl").read_text().splitlines():
        assert json.loads(line)["train_fg_miou"] == pytest.approx(score, abs=1e-12)

    # The default flips training images, and never validation images.
    flipped = _CodedModel()
    train(flipped, _cfg(data_root=str(root), out_dir=str(tmp_path / "c"), epochs=3, batch_size=2, max_val_images=2))
    assert not all(matches(cube, raw) for cube in flipped.train_inputs)
    assert flipped.eval_inputs and all(matches(cube, val_raw) for cube in flipped.eval_inputs)


def test_epoch_timing_fields(tmp_path):
    """train_seconds is this epoch's training pass; epoch_seconds adds the
    validation pass. Slow forwards make both measurable: 0.1 s per training
    call (2 per epoch) and 0.3 s per validation call (1 per epoch)."""
    from sw.train import train

    root = tmp_path / "data"
    _write_coded_split(root, "train", 4, seed=0)
    _write_coded_split(root, "val", 1, seed=1)
    out = tmp_path / "run"
    train(_CodedModel(train_sleep=0.1, eval_sleep=0.3), _cfg(data_root=str(root), out_dir=str(out), epochs=3, batch_size=2))
    for line in (out / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert 0.2 <= row["train_seconds"] < 0.5, row  # this epoch only, not cumulative
        assert row["epoch_seconds"] >= row["train_seconds"] + 0.3, row  # validation included
        assert row["epoch_seconds"] < 1.0, row


def test_transformer_keeps_a_one_image_tail_and_trains_on_one_image(tmp_path):
    from sw.train import TrainConfig, build_model, train

    root = tmp_path / "data"
    _write_split(root, "train", 5)
    _write_split(root, "val", 1)

    def steps(model_name, batch_size, images=None):
        cfg = _cfg(
            data_root=str(root), out_dir=str(tmp_path / f"{model_name}{batch_size}{images}"), model=model_name,
            batch_size=batch_size, max_train_images=images, depth_spectral=0, depth_spatial=1,
        )
        train(build_model(cfg), cfg)
        rows = (tmp_path / f"{model_name}{batch_size}{images}" / "metrics.jsonl").read_text().splitlines()
        return json.loads(rows[-1])["steps"]

    assert steps("transformer", 2) == 3  # 2 + 2 + 1: the tail of one image is kept
    assert steps("cnn", 2) == 2  # BatchNorm: the tail of one image is dropped
    assert steps("transformer", 1) == 5
    assert steps("transformer", 1, images=1) == 1  # one training image is enough
    with pytest.raises(RuntimeError, match="BatchNorm"):
        steps("cnn", 2, images=1)


def _write_learnable_split(root, split, n_images, seed):
    """Cubes whose 8 by 8 patches carry a class-specific 16-band shape.

    The band-token model can fit this quickly, so it stands in for the real
    overfit check (no augmentation, 4 images, train FG mIoU rising).
    """
    rng = np.random.default_rng(seed)
    shapes = np.random.default_rng(123).random((7, 16))
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(n_images):
        patch_classes = rng.integers(0, 7, size=(4, 4))
        mask = np.kron(patch_classes, np.ones((8, 8), dtype=np.int64)).astype(np.uint8)  # (32, 32)
        cube = 0.05 * rng.random((224, 32, 32)) + np.tile(shapes, (1, 14))[mask].transpose(2, 0, 1)[:224]
        tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", (cube / 1.1 * 65535).astype(np.uint16), photometric="minisblack")
        Image.fromarray(mask, mode="L").save(labels / f"{split}_{index:02d}.png")


def test_overfit_gate_runs_through_the_cli(tmp_path):
    """The plan's overfit check, shrunk to CPU size: --no-aug, 4 images, a
    small transformer, 120 epochs of one step. Measured across 44 runs on two
    machines (torch 2.14 and 2.15; seeds, data seeds and thread counts
    varied): the final train FG mIoU was 0.889 to 0.951. A rise measured from
    epoch 30 was not robust (3 of 36 runs under 0.4), so the rise is
    measured from epoch 1."""
    root = tmp_path / "data"
    _write_learnable_split(root, "train", 4, seed=0)
    _write_learnable_split(root, "val", 2, seed=1)
    out = tmp_path / "gate"
    main(
        [
            "--model", "transformer", "--depth-spectral", "1", "--depth-spatial", "1",
            "--no-aug", "--no-cosine", "--max-train-images", "4", "--max-val-images", "2",
            "--batch-size", "4", "--epochs", "120", "--num-workers", "0",
            "--data-root", str(root), "--out-dir", str(out), "--device", "cpu", "--seed", "0",
        ]
    )
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 120
    assert rows[-1]["train_fg_miou"] > 0.85
    assert rows[-1]["train_fg_miou"] - rows[0]["train_fg_miou"] > 0.5
    assert set(rows[-1]["train_per_class_iou"]) == set(rows[-1]["per_class_iou"])
