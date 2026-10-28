"""Training harness for SpectralWaste.

``--model cnn`` (the default) is ``MiniNetSeg``. ``--model transformer``
is the band-token scaffold in ``sw.models.transformer``. That forward
raises ``NotImplementedError`` until the TODOs in
``docs/TRANSFORMER-TODO.md`` are filled in. Leaving ``--model`` unset
keeps the CNN path.

``train(model, cfg)`` runs AdamW and weighted cross-entropy, reports
foreground mIoU on the validation split after every epoch, and writes
``best.pt``, ``last.pt``, ``metrics.jsonl``, and ``metrics.csv`` under
``cfg.out_dir``. ``last.pt`` is the resume file (model, optimizer,
scheduler, epoch, best foreground mIoU). ``best.pt`` is the best weights.

Run it from the repo root:

    PYTHONPATH=src python -m sw.train --smoke
    PYTHONPATH=src python -m sw.train --model transformer --smoke
    PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --out-dir runs/cnn

``--smoke`` is two optimizer steps on a few images. It is a CPU-friendly
wiring check. The number it prints is not a result to put on a slide.

A full run reads every cube from disk each epoch (about 24 GB unzipped).
That is a Colab GPU job. Put the cubes on local disk and the checkpoints
on Drive. The commands are in ``docs/COLAB.md``. ``SW_ROOT`` must point at
the folder that contains ``hyper/`` and ``labels_hyper_lt/``.
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
from sw.models.transformer import BandTransformer


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

    The ``max_*`` fields shorten a run. ``None`` means the whole split and
    every batch. ``--smoke`` sets them when the CLI flags are left unset.
    ``patience`` stops after that many epochs without a better validation
    foreground mIoU. ``None`` runs every requested epoch.

    ``cosine`` anneals the learning rate with ``CosineAnnealingLR`` over
    ``epochs``. ``amp`` turns on CUDA mixed precision. On cpu or mps it is
    ignored and a line is printed. ``resume`` is a ``last.pt`` path; training
    continues at the next epoch and the metrics files are appended.

    ``model`` is ``"cnn"`` (``MiniNetSeg``) or ``"transformer"``
    (``BandTransformer``). The depth, width, head, MLP, patch, and group
    fields size the transformer. A CNN run stores them and leaves them unused.
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
    cosine: bool = True
    amp: bool = False
    resume: str | None = None
    model: str = "cnn"
    depth_spectral: int = 2
    depth_spatial: int = 6
    dim: int = 64
    heads: int = 4
    mlp_ratio: float = 4.0
    patch: int = 8
    group: int = 16


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


def _train_drop_last(n_images: int, batch_size: int) -> bool:
    """True when the last training batch would contain a single image.

    BatchNorm cannot estimate a variance from one image. ``drop_last`` on the
    loader leaves that image out. A tail of 2 or more images is kept.
    """
    if n_images < batch_size:
        return False
    return n_images % batch_size == 1


def _loader(dataset, cfg: TrainConfig, shuffle: bool, pin_memory: bool, drop_last: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
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
    if cfg.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 so BatchNorm does not train on a batch of 1, got {cfg.batch_size}"
        )
    if cfg.downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {cfg.downsample}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be > 0, got {cfg.lr}")
    if cfg.weight_scheme not in ("paper", "inverse"):
        raise ValueError(f"weight_scheme must be 'paper' or 'inverse', got {cfg.weight_scheme!r}")
    for name in ("max_train_images", "max_val_images", "max_train_steps"):
        value = getattr(cfg, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if cfg.resume is not None and not Path(cfg.resume).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {cfg.resume}")
    if cfg.model not in ("cnn", "transformer"):
        raise ValueError(f"model must be 'cnn' or 'transformer', got {cfg.model!r}")
    if cfg.model == "transformer":
        if cfg.depth_spectral < 0 or cfg.depth_spatial < 0:
            raise ValueError(
                f"depths must be >= 0, got spectral {cfg.depth_spectral}, "
                f"spatial {cfg.depth_spatial}"
            )
        if cfg.dim < 1 or cfg.heads < 1 or cfg.dim % cfg.heads != 0:
            raise ValueError(f"dim={cfg.dim} must be divisible by heads={cfg.heads}")
        if cfg.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {cfg.mlp_ratio}")
        if cfg.patch < 1:
            raise ValueError(f"patch must be >= 1, got {cfg.patch}")
        if cfg.group < 1:
            raise ValueError(f"group must be >= 1, got {cfg.group}")


def _log_mode(path: Path, resume: bool) -> str:
    """Append a metrics file on resume. A new run replaces it."""
    if resume and path.is_file() and path.stat().st_size > 0:
        return "a"
    return "w"


def _save_last(path, model, optimizer, scheduler, scaler, epoch, best, best_epoch, stall, cfg, weights, use_amp):
    """Resume checkpoint. ``best.pt`` stays the best weights only."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "best_val_fg_miou": None if best_epoch is None else best,
        "best_epoch": best_epoch,
        "stall": stall,
        "class_weight_scheme": cfg.weight_scheme,
        "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
        "cfg": asdict(cfg),
    }
    if use_amp:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"saved {path}  (epoch {epoch})")


def _load_resume(path, model, optimizer, scheduler, scaler, device, use_amp) -> dict:
    """Restore a ``last.pt`` and return where the next epoch should start."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt or "epoch" not in ckpt:
        raise KeyError(f"{path} is missing 'model' or 'epoch'")
    if "optimizer" not in ckpt:
        raise KeyError(
            f"{path} has weights but no optimizer. Resume from last.pt "
            "(best.pt is the best weights only)."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is None:
        if ckpt.get("scheduler") is not None:
            print("cosine is off for this run; not restoring the scheduler")
    elif ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("checkpoint has no scheduler; cosine restarts from the base learning rate")
    if use_amp and scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    elif use_amp:
        print("checkpoint has no GradScaler state; mixed precision starts fresh")
    finished = int(ckpt["epoch"])
    raw_best = ckpt.get("best_val_fg_miou")
    if raw_best is None:
        best = -math.inf
        best_epoch = None
    else:
        best = float(raw_best)
        best_epoch = ckpt.get("best_epoch")
    shown = "none" if best_epoch is None else f"{best:.4f}"
    print(
        f"resumed {path}  finished epoch {finished}  "
        f"next epoch {finished + 1}  best val FG mIoU {shown}"
    )
    return {
        "start_epoch": finished + 1,
        "best": best,
        "best_epoch": best_epoch,
        "stall": int(ckpt.get("stall", 0)),
    }


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
    drop_last = _train_drop_last(len(train_ds), cfg.batch_size)
    if drop_last:
        print(
            "train loader drop_last: leaving out 1 image "
            "so BatchNorm does not train on a batch of 1"
        )
    pin_memory = device.type == "cuda"
    train_loader = _loader(train_ds, cfg, shuffle=True, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = _loader(val_ds, cfg, shuffle=False, pin_memory=pin_memory, drop_last=False)

    weights = class_weights(scheme=cfg.weight_scheme).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    if cfg.amp and not use_amp:
        print(f"amp applies to cuda only; device is {device.type}, training in full precision")
    # enabled=False is a no-op, so cpu and mps take the same step as a plain backward.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -math.inf
    best_epoch = None
    stall = 0
    start_epoch = 1
    if cfg.resume:
        restored = _load_resume(cfg.resume, model, optimizer, scheduler, scaler, device, use_amp)
        start_epoch = restored["start_epoch"]
        best = restored["best"]
        best_epoch = restored["best_epoch"]
        stall = restored["stall"]

    n_params = count_parameters(model)
    limited = cfg.max_train_steps is not None or cfg.max_train_images is not None
    schedule = "cosine" if scheduler is not None else "constant"
    print(f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    print(
        f"device {device}  epochs {start_epoch}-{cfg.epochs}  batch {cfg.batch_size}  "
        f"lr {cfg.lr}  weight_decay {cfg.weight_decay}  schedule {schedule}  amp {'on' if use_amp else 'off'}"
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
    if start_epoch > cfg.epochs:
        print(
            f"checkpoint already finished epoch {start_epoch - 1}; "
            f"--epochs is {cfg.epochs}. Nothing to run."
        )
        return _result(best, best_epoch, start_epoch - 1, n_params, out_dir, jsonl_path)

    fieldnames = ["epoch", "train_loss", "val_fg_miou", "steps", *[f"iou_{name}" for name in CLASS_NAMES]]
    epochs_ran = start_epoch - 1
    jsonl_mode = _log_mode(jsonl_path, cfg.resume is not None)
    csv_mode = _log_mode(csv_path, cfg.resume is not None)

    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jsonl_file, csv_path.open(
        csv_mode, newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, cfg.epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            steps = 0
            for cubes, masks in train_loader:
                if cubes.shape[0] < 2:
                    raise RuntimeError(
                        f"training batch has {cubes.shape[0]} image(s); "
                        "BatchNorm needs at least 2. The train loader should have dropped this tail."
                    )
                cubes = cubes.to(device)
                masks = masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = criterion(model(cubes), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:02d}  steps {steps:4d}  lr {current_lr:.6g}  "
                f"train_loss {train_loss:.4f}  val_fg_miou {shown}"
            )

            improved = math.isfinite(score) and score > best
            if improved:
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
            if scheduler is not None:
                scheduler.step()
            _save_last(
                out_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best,
                best_epoch,
                stall,
                cfg,
                weights,
                use_amp,
            )
            if not improved and cfg.patience is not None and stall >= cfg.patience:
                print(f"early stop at epoch {epoch} (patience {cfg.patience})")
                break

    return _result(best, best_epoch, epochs_ran, n_params, out_dir, jsonl_path)


def _result(best, best_epoch, epochs_ran, n_params, out_dir, jsonl_path) -> dict:
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
        description="Train a SpectralWaste model. The default is the CNN control (MiniNetSeg)."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "transformer"],
        default="cnn",
        help="cnn is MiniNetSeg. transformer is the band-token scaffold.",
    )
    parser.add_argument(
        "--depth-spectral",
        type=int,
        default=2,
        help="Spectral blocks (transformer). Default 2.",
    )
    parser.add_argument(
        "--depth-spatial",
        type=int,
        default=6,
        help="Spatial blocks (transformer). Default 6.",
    )
    parser.add_argument("--dim", type=int, default=64, help="Token width D (transformer). Default 64.")
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Attention heads (transformer). dim must be divisible by heads. Default 4.",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="MLP hidden width is dim * mlp_ratio (transformer). Default 4.",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=8,
        help=(
            "Tokenizer patch and stride (transformer). "
            "At --downsample 2 a 256 cube is 128, so patch 8 is a 16x16 grid (256 tokens)."
        ),
    )
    parser.add_argument(
        "--group",
        type=int,
        default=16,
        help="Bands per tokenizer group (transformer). 224 / 16 = 14 groups.",
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
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default: runs/cnn, runs/transformer with --model transformer, or runs/smoke with --smoke.",
    )
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
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Use only the first N training images. Default: all, or 4 with --smoke.",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Use only the first N validation images. Default: all, or 2 with --smoke.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Stop each epoch after this many optimizer steps. Default: all, or 2 with --smoke.",
    )
    parser.add_argument(
        "--no-cosine",
        action="store_true",
        help="Constant learning rate. The default is CosineAnnealingLR over --epochs.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Mixed precision on CUDA (autocast + GradScaler). Ignored on cpu and mps.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="last.pt from an interrupted run. Continues at the next epoch and appends the metrics log.",
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
    elif args.model == "transformer":
        out_dir = "runs/transformer"
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
        max_train_images=4 if smoke and args.max_train_images is None else args.max_train_images,
        max_val_images=2 if smoke and args.max_val_images is None else args.max_val_images,
        max_train_steps=2 if smoke and args.max_train_steps is None else args.max_train_steps,
        patience=args.patience,
        cosine=not args.no_cosine,
        amp=bool(args.amp),
        resume=args.resume,
        model=args.model,
        depth_spectral=args.depth_spectral,
        depth_spatial=args.depth_spatial,
        dim=args.dim,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        patch=args.patch,
        group=args.group,
    )


def build_model(cfg: TrainConfig, in_channels: int = 224) -> nn.Module:
    """``cnn`` builds ``MiniNetSeg``. ``transformer`` builds ``BandTransformer``.

    The transformer size fields live on ``cfg``. ``in_channels`` is the CLI
    ``--in-channels`` value (real cubes are 224) and is the tokenizer's
    band count. CNN runs ignore the transformer size fields.
    """
    if cfg.model == "cnn":
        return MiniNetSeg(in_channels=in_channels)
    if cfg.model == "transformer":
        return BandTransformer(
            depth_spectral=cfg.depth_spectral,
            depth_spatial=cfg.depth_spatial,
            dim=cfg.dim,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            num_classes=7,
            n_bands=in_channels,
            group=cfg.group,
            patch=cfg.patch,
        )
    raise ValueError(f"model must be 'cnn' or 'transformer', got {cfg.model!r}")


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    cfg = config_from_args(args)
    # Seed before the constructor so a re-run with the same seed starts
    # from the same weights. train() seeds again for the data pipeline.
    set_seed(cfg.seed)
    model = build_model(cfg, in_channels=args.in_channels)
    return train(model, cfg)


if __name__ == "__main__":
    main()
