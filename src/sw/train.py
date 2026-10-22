"""Training harness for the SpectralWaste CNN control.

``train(model, cfg)`` runs AdamW and weighted cross-entropy, reports
foreground mIoU on the validation split after every epoch, and writes
``best.pt``, ``metrics.jsonl``, and ``metrics.csv`` under ``cfg.out_dir``.

Run it from the repo root:

    PYTHONPATH=src python -m sw.train --smoke
    PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --out-dir runs/cnn

``--smoke`` is two optimizer steps on a few images. It is a CPU-friendly
wiring check. The number it prints is not a result to put on a slide.

A full run reads every cube from disk each epoch (about 24 GB unzipped).
That is a Colab GPU job. ``SW_ROOT`` must point at the folder that contains
``hyper/`` and ``labels_hyper_lt/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sw.data import CLASS_NAMES, SpectralWasteDataset, class_weights
from sw.metrics import ConfusionMeter
from sw.models.cnn import MiniNetSeg, count_parameters


@dataclass
class TrainConfig:
    """Knobs for :func:`train`.

    ``data_root`` is the folder with ``hyper/`` and ``labels_hyper_lt/``.
    ``None`` reads the ``SW_ROOT`` environment variable.

    ``downsample`` is an integer stride applied first. ``1`` keeps every
    pixel. ``2`` keeps every other pixel, so 256 becomes 128, and class ids
    stay integers (no blending).

    ``crop_size`` is the square side length after downsampling. ``None``
    keeps the whole image. Training crops at a random offset; validation
    crops the center.

    ``device`` is ``"auto"`` (cuda, then mps, then cpu) or one of those names.

    The ``max_*`` fields are how ``--smoke`` shortens a run. ``None`` means
    the whole split and every batch. ``patience`` stops after that many
    epochs without a better validation foreground mIoU. ``None`` runs every
    epoch.
    """

    data_root: str | None = None
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    crop_size: int | None = None
    downsample: int = 1
    num_workers: int = 2
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs/cnn"
    weight_scheme: str = "paper"
    max_train_images: int | None = None
    max_val_images: int | None = None
    max_train_steps: int | None = None
    patience: int | None = None


def resolve_device(name: str) -> torch.device:
    """``auto`` picks cuda, then mps, then cpu. A named device must exist."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but CUDA is not available")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("device='mps' but MPS is not available")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sample(cube, mask, *, train: bool, crop_size: int | None, downsample: int):
    """Downsample, flip (training only), then crop. Cube and mask stay aligned.

    ``cube`` is ``(C, H, W)`` and ``mask`` is ``(H, W)``.
    """
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")
    if downsample > 1:
        if cube.shape[-2] < downsample or cube.shape[-1] < downsample:
            raise ValueError(
                f"downsample {downsample} is larger than the image "
                f"{tuple(cube.shape[-2:])}"
            )
        cube = cube[:, ::downsample, ::downsample]
        mask = mask[::downsample, ::downsample]
    if train:
        # Independent horizontal and vertical flips. Both tensors flip together.
        if torch.rand(()) < 0.5:
            cube = cube.flip(-1)
            mask = mask.flip(-1)
        if torch.rand(()) < 0.5:
            cube = cube.flip(-2)
            mask = mask.flip(-2)
    if crop_size is not None:
        cube, mask = _crop(cube, mask, crop_size, train=train)
    return cube.contiguous(), mask.contiguous()


def _crop(cube, mask, size: int, *, train: bool):
    _, height, width = cube.shape
    if size < 1:
        raise ValueError(f"crop_size must be >= 1, got {size}")
    if height < size or width < size:
        raise ValueError(f"crop_size {size} is larger than the image {height}x{width}")
    if train:
        top = int(torch.randint(0, height - size + 1, ()).item())
        left = int(torch.randint(0, width - size + 1, ()).item())
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    return cube[:, top : top + size, left : left + size], mask[top : top + size, left : left + size]


class TransformedSplit(Dataset):
    """A SpectralWaste split plus the harness crop, flip, and downsample."""

    def __init__(self, base, indices, crop_size, downsample, train: bool):
        self.base = base
        self.indices = list(indices)
        self.crop_size = crop_size
        self.downsample = downsample
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        cube, mask = self.base[self.indices[index]]
        return prepare_sample(
            cube,
            mask,
            train=self.train,
            crop_size=self.crop_size,
            downsample=self.downsample,
        )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, cfg: TrainConfig, shuffle: bool, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    # A trailing batch of 1 image breaks BatchNorm in train mode.
    drop_last = shuffle and len(dataset) % cfg.batch_size == 1 and len(dataset) > cfg.batch_size
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_seed_worker,
        pin_memory=pin_memory,
    )


def _split(cfg: TrainConfig, name: str, limit: int | None, train: bool) -> TransformedSplit:
    base = SpectralWasteDataset(cfg.data_root, split=name, random_flip=False)
    if len(base) == 0:
        raise RuntimeError(f"{name} split is empty under {cfg.data_root or 'SW_ROOT'}")
    n = len(base) if limit is None else min(len(base), limit)
    if n < 1:
        raise RuntimeError(f"{name} split has no images to load")
    return TransformedSplit(base, range(n), cfg.crop_size, cfg.downsample, train=train)


def _iou_payload(iou) -> dict:
    """Per-class IoU as JSON-safe floats. NaN (class absent) becomes null."""
    payload = {}
    for name, value in zip(CLASS_NAMES, iou.tolist()):
        value = float(value)
        payload[name] = None if math.isnan(value) else value
    return payload


def _evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    meter = ConfusionMeter()
    with torch.no_grad():
        for cubes, masks in loader:
            logits = model(cubes.to(device))
            meter.update(logits, masks)
    return meter.fg_miou(), _iou_payload(meter.per_class_iou())


def _check_cfg(cfg: TrainConfig) -> None:
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.epochs}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {cfg.downsample}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be > 0, got {cfg.lr}")
    if cfg.weight_scheme not in ("paper", "inverse"):
        raise ValueError(f"weight_scheme must be 'paper' or 'inverse', got {cfg.weight_scheme!r}")


def train(model: nn.Module, cfg: TrainConfig) -> dict:
    """Train ``model`` and return the best validation foreground mIoU.

    The model is moved to ``cfg.device``. Loss is weighted cross-entropy
    with :func:`sw.data.class_weights` (default scheme ``"paper"``).
    Seed the RNG before you construct ``model`` if the initial weights
    should follow ``cfg.seed``. This function seeds again so the flips,
    crops, and shuffle follow ``cfg.seed`` on their own.
    """
    _check_cfg(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _split(cfg, "train", cfg.max_train_images, train=True)
    val_ds = _split(cfg, "val", cfg.max_val_images, train=False)
    if len(train_ds) < 2:
        raise RuntimeError(
            "Need at least 2 training images. BatchNorm cannot train on a batch of 1."
        )
    pin_memory = device.type == "cuda"
    train_loader = _loader(train_ds, cfg, shuffle=True, pin_memory=pin_memory)
    val_loader = _loader(val_ds, cfg, shuffle=False, pin_memory=pin_memory)

    weights = class_weights(scheme=cfg.weight_scheme).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = count_parameters(model)
    limited = cfg.max_train_steps is not None or cfg.max_train_images is not None
    print(f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    print(
        f"device {device}  epochs {cfg.epochs}  batch {cfg.batch_size}  "
        f"lr {cfg.lr}  weight_decay {cfg.weight_decay}"
    )
    print(
        f"train images {len(train_ds)}  val images {len(val_ds)}  "
        f"downsample {cfg.downsample}  crop {cfg.crop_size}  out {out_dir}"
    )
    rounded = " ".join(f"{value:.3f}" for value in weights.detach().cpu().tolist())
    print(f"class weights ({cfg.weight_scheme}): {rounded}")
    if limited:
        print("Limited run: the val FG mIoU below checks the wiring. It is not a score to report.")

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_fg_miou", "steps", *[f"iou_{name}" for name in CLASS_NAMES]]
    best = -math.inf
    best_epoch = None
    stall = 0
    epochs_ran = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            steps = 0
            for cubes, masks in train_loader:
                if cubes.shape[0] < 2:
                    # BatchNorm variance is undefined for a single image.
                    continue
                cubes = cubes.to(device)
                masks = masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(cubes), masks)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * cubes.shape[0]
                seen += cubes.shape[0]
                steps += 1
                if cfg.max_train_steps is not None and steps >= cfg.max_train_steps:
                    break
            if steps == 0:
                raise RuntimeError("no training steps ran; check batch_size and the split size")
            train_loss = loss_sum / seen
            score, per_class = _evaluate(model, val_loader, device)
            epochs_ran = epoch
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_fg_miou": None if math.isnan(score) else score,
                "steps": steps,
                "per_class_iou": per_class,
            }
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()
            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_fg_miou": "" if math.isnan(score) else f"{score:.6f}",
                "steps": steps,
            }
            for name in CLASS_NAMES:
                value = per_class[name]
                row[f"iou_{name}"] = "" if value is None else f"{value:.6f}"
            writer.writerow(row)
            csv_file.flush()
            shown = "nan" if math.isnan(score) else f"{score:.4f}"
            print(f"epoch {epoch:02d}  steps {steps:4d}  train_loss {train_loss:.4f}  val_fg_miou {shown}")

            if math.isfinite(score) and score > best:
                best = score
                best_epoch = epoch
                stall = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_fg_miou": score,
                        "per_class_iou": per_class,
                        "class_weight_scheme": cfg.weight_scheme,
                        "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
                        "cfg": asdict(cfg),
                    },
                    out_dir / "best.pt",
                )
                print(f"saved {out_dir / 'best.pt'}  (epoch {epoch}, val FG mIoU {score:.4f})")
            else:
                stall += 1
                if cfg.patience is not None and stall >= cfg.patience:
                    print(f"early stop at epoch {epoch} (patience {cfg.patience})")
                    break

    result = {
        "best_val_fg_miou": None if best_epoch is None else best,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "param_count": n_params,
        "out_dir": str(out_dir),
    }
    if result["best_val_fg_miou"] is None:
        best_text = "none"
    else:
        best_text = f"{result['best_val_fg_miou']:.4f}"
    print(f"done  best val FG mIoU {best_text}  epoch {best_epoch}  log {jsonl_path}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SpectralWaste CNN control (MiniNetSeg)."
    )
    parser.add_argument("--data-root", default=None, help="Dataset folder. Default: the SW_ROOT environment variable.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=None, help="Square crop after downsampling. Default: full image.")
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Integer stride. 2 turns 256 into 128. Default: 1, or 4 with --smoke.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--out-dir", default=None, help="Default: runs/cnn, or runs/smoke with --smoke.")
    parser.add_argument(
        "--weight-scheme",
        default="paper",
        choices=["paper", "inverse"],
        help="paper is the SpectralWaste median-frequency scheme. inverse is mean-1 inverse frequency.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop after this many epochs without a better val FG mIoU.",
    )
    parser.add_argument("--in-channels", type=int, default=224, help="Bands in the cube. Real data is 224.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two train steps on 4 train images and 2 val images (1 epoch, batch 2). CPU wiring check, not a score.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Turn CLI flags into a :class:`TrainConfig`. ``--smoke`` overrides the length of the run."""
    smoke = bool(args.smoke)
    if smoke and args.downsample is None:
        downsample = 4
    elif args.downsample is None:
        downsample = 1
    else:
        downsample = args.downsample
    if smoke and args.num_workers is None:
        num_workers = 0
    elif args.num_workers is None:
        num_workers = 2
    else:
        num_workers = args.num_workers
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif smoke:
        out_dir = "runs/smoke"
    else:
        out_dir = "runs/cnn"
    return TrainConfig(
        data_root=args.data_root,
        epochs=1 if smoke else args.epochs,
        batch_size=2 if smoke else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        crop_size=args.crop_size,
        downsample=downsample,
        num_workers=num_workers,
        seed=args.seed,
        device=args.device,
        out_dir=out_dir,
        weight_scheme=args.weight_scheme,
        max_train_images=4 if smoke else None,
        max_val_images=2 if smoke else None,
        max_train_steps=2 if smoke else None,
        patience=args.patience,
    )


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    cfg = config_from_args(args)
    # Seed before the constructor so a re-run with the same seed starts
    # from the same weights. train() seeds again for the data pipeline.
    set_seed(cfg.seed)
    model = MiniNetSeg(in_channels=args.in_channels)
    return train(model, cfg)


if __name__ == "__main__":
    main()
