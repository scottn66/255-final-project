"""Training harness for SpectralWaste.

``--model cnn`` (the default) is ``MiniNetSeg``. ``--model transformer``
is the band-token transformer in ``sw.models.transformer`` (488,487
parameters at the defaults). Leaving ``--model`` unset keeps the CNN path.

``train(model, cfg)`` runs AdamW and weighted cross-entropy, reports
foreground mIoU on the validation split after every epoch, and writes
``best.pt``, ``last.pt``, ``metrics.jsonl``, and ``metrics.csv`` under
``cfg.out_dir``. ``last.pt`` is the resume file (model, optimizer,
scheduler, epoch, best foreground mIoU). ``best.pt`` is the best weights.

Each ``metrics.jsonl`` row also has the training-set FG mIoU and per-class
IoU of that epoch and its wall time; ``metrics.csv`` keeps its original
columns so older runs can still be resumed.

The optimiser recipe is three flags, all off by default: ``--warmup-steps``
(linear warmup, counted in optimizer steps), ``--clip-grad-norm`` (clip the
global gradient norm before each step) and ``--cosine-floor`` (the cosine
schedule stops at that fraction of ``--lr``, not at 0). Why: on four real
cubes at a constant learning rate with no clipping, one seed's loss jumped
from 0.10 to 1.13 near epoch 550 (``sw.gates``).

The overfit-one-batch check (no augmentation, 4 training cubes that hold
all six object classes, linked into a small root by
``sw.gates.build_subset_root``; the pass rule is ``sw.gates.judge_overfit``
and ``docs/COLAB.md``) is::

    PYTHONPATH=src python -m sw.train --model transformer --data-root <subset root> \
        --downsample 2 --no-aug --no-cosine --max-train-images 4 --max-val-images 4 \
        --batch-size 4 --epochs 600

Run it from the repo root:

    PYTHONPATH=src python -m sw.train --smoke
    PYTHONPATH=src python -m sw.train --model transformer --smoke
    PYTHONPATH=src python -m sw.train --epochs 30 --batch-size 4 --out-dir runs/cnn

The band-collapsed control for the fusion claim is the same transformer
with ``--tokenizer collapsed --depth-spectral 0 --depth-spatial 8``
(491,191 parameters against 488,487). Give it its own ``--out-dir``.

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
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from sw.data import (
    CLASS_NAMES,
    HYPER_LT_TRAIN_PIXEL_COUNTS,
    SpectralWasteDataset,
    class_weights,
)
from sw.metrics import NUM_CLASSES, ConfusionMeter, fg_miou_from_cm
from sw.models.cnn import MiniNetSeg, count_parameters
from sw.models.transformer import BandTransformer, init_seg_head_prior_bias
from sw.standardize import BandStandardizer, load_or_compute_band_stats


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

    ``warmup_steps`` (W) ramps the learning rate linearly over the first W
    optimizer steps of the run: step k (counting from 1) uses the scheduled
    rate times ``min(1, k / W)``. The scheduled rate is ``lr``, or with
    ``cosine`` the cosine value of the current epoch, so the two multiply:
    cosine moves once per epoch, warmup once per step. ``0`` is no warmup.
    The step count is stored in ``last.pt``, so a resume continues the ramp.

    ``clip_grad_norm`` (C) rescales the gradient of all parameters to a
    global L2 norm of at most C before each optimizer step
    (``torch.nn.utils.clip_grad_norm_``). ``None`` is no clipping. With it
    on, each ``metrics.jsonl`` row has ``grad_norm_max``, the largest
    finite norm of the epoch before clipping (``null`` if there was none),
    and ``grad_nonfinite_steps``, the number of steps whose norm was NaN
    or infinite.

    ``cosine_floor`` (F) is the lowest point of the cosine schedule as a
    fraction of ``lr`` (``eta_min = lr * F``). ``0`` anneals to zero. It
    bounds the schedule, not the warmup ramp, and does nothing without
    ``cosine``. The last epoch trains just above the floor; the floor is
    the rate after it.

    Long runs over several sessions (``docs/COLAB.md``), all off by default:
    ``save_every`` (N) writes ``last.pt`` every N epochs and always on the
    last epoch or a stop; ``keep_last`` (K) also keeps the K newest
    ``ckpt_eNNNNN.pt`` copies of it. ``eval_every`` (N) validates every N
    epochs (and on the last and a stop); other epochs log ``val_fg_miou``
    null, and ``patience`` counts validations. ``max_minutes`` stops cleanly,
    after a saved epoch, once another epoch would not fit in that many
    minutes from the start of this call; the result then has
    ``stopped: "time_budget"`` and a later ``--resume`` carries on.
    ``last.pt`` also holds the RNG states (python, numpy, torch, cuda and the
    train loader's shuffle), restored on resume.

    ``init_from`` starts a NEW run (epoch 1, fresh schedule and warmup) from the
    weights in that checkpoint; ``init_optimizer`` also takes its AdamW
    moments, with this run's ``lr`` and ``weight_decay``. A ``resume`` wins
    over it. This is how a variant forks from another run's state.

    A resume keeps the cosine schedule stored in ``last.pt``. A different
    floor is refused. A larger ``epochs`` prints a warning: the schedule
    keeps its old length and the rate climbs back up after it, so fix
    ``epochs`` once for a cosine run.

    ``model`` is ``"cnn"`` (``MiniNetSeg``) or ``"transformer"``
    (``BandTransformer``). The depth, width, head, MLP, patch, and group
    fields size the transformer. A CNN run stores them and leaves them unused.

    ``augment=False`` turns off the training flips and makes a training crop
    a center crop, so every epoch sees exactly the same pixels. It is for
    the overfit-one-batch check.

    ``standardize`` applies per-band mean/std from the TRAIN split only
    (cached at ``stats_path``, default ``{out_dir}/band_stats.pt``). Val and
    test use the same affine. See ``docs/INPUT-BLIND.md``.

    ``pe_init_std`` scales ``IndexBandPE`` after its default ``N(0, 1)`` draw
    (transformer only). ``1.0`` leaves today's initialisation alone.

    ``head_prior_bias`` sets the SegHead final conv bias to ``log`` train
    class priors (transformer only; default off).

    ``tokenizer`` is ``"band"`` (``BandTokenizer``, the default) or
    ``"collapsed"`` (``CollapsedTokenizer``: a 1x1 conv mixes all bands
    before the patch embedding). Transformer only. ``"collapsed"`` needs
    ``depth_spectral=0``; with ``depth_spatial=8`` it is the
    parameter-matched control for the default model (491,191 against
    488,487 parameters; FLOPs and step time are not matched). ``group`` and
    ``pe_init_std`` have no effect with it. A config saved before this
    field existed has no ``tokenizer`` entry and means ``"band"``.

    Default ``lr`` stays ``1e-3`` for both models. For the transformer on
    raw (non-standardised) cubes, ``3e-4`` is the recommended starting
    point; see ``docs/INPUT-BLIND.md``.

    Batch size: the CNN has BatchNorm, so it needs at least 2 images per
    training batch. The transformer has no batch statistics and may train
    on batches of 1. A transformer ``crop_size`` must be a multiple of
    ``patch``: the tokenizer would drop the remainder anyway.
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
    augment: bool = True
    standardize: bool = False
    stats_path: str | None = None
    pe_init_std: float = 1.0
    head_prior_bias: bool = False
    tokenizer: str = "band"
    warmup_steps: int = 0
    clip_grad_norm: float | None = None
    cosine_floor: float = 0.0
    save_every: int = 1
    keep_last: int = 0
    eval_every: int = 1
    max_minutes: float | None = None
    init_from: str | None = None
    init_optimizer: bool = False


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

    def __init__(
        self,
        base,
        indices,
        crop_size,
        downsample,
        train: bool,
        standardizer: BandStandardizer | None = None,
    ):
        self.base = base
        self.indices = list(indices)
        self.crop_size = crop_size
        self.downsample = downsample
        self.train = train
        self.standardizer = standardizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        cube, mask = self.base[self.indices[index]]
        cube, mask = prepare_sample(
            cube,
            mask,
            train=self.train,
            crop_size=self.crop_size,
            downsample=self.downsample,
        )
        if self.standardizer is not None:
            cube = self.standardizer(cube)
        return cube, mask


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


def _split(
    cfg: TrainConfig,
    name: str,
    limit: int | None,
    train: bool,
    standardizer: BandStandardizer | None = None,
) -> TransformedSplit:
    base = SpectralWasteDataset(cfg.data_root, split=name, random_flip=False)
    if len(base) == 0:
        raise RuntimeError(f"{name} split is empty under {cfg.data_root or 'SW_ROOT'}")
    n = len(base) if limit is None else min(len(base), limit)
    if n < 1:
        raise RuntimeError(f"{name} split has no images to load")
    return TransformedSplit(
        base,
        range(n),
        cfg.crop_size,
        cfg.downsample,
        train=train,
        standardizer=standardizer,
    )


def _resolve_standardizer(cfg: TrainConfig) -> BandStandardizer | None:
    """Build a train-only band standardizer, or ``None`` when the flag is off."""
    if not cfg.standardize:
        return None
    stats_path = cfg.stats_path
    if stats_path is None:
        stats_path = str(Path(cfg.out_dir) / "band_stats.pt")
    # Stats from the raw train cubes (no aug / crop). Limits follow max_train_images
    # so a smoke or subset run does not scan the whole split.
    base = SpectralWasteDataset(cfg.data_root, split="train", random_flip=False)
    if len(base) == 0:
        raise RuntimeError(f"train split is empty under {cfg.data_root or 'SW_ROOT'}")
    n = len(base) if cfg.max_train_images is None else min(len(base), cfg.max_train_images)
    if n < 1:
        raise RuntimeError("train split has no images to compute band stats")
    subset = Subset(base, list(range(n)))
    mean, std = load_or_compute_band_stats(stats_path, subset, split="train")
    print(f"standardize: train band stats -> {stats_path}  (C={mean.numel()})")
    return BandStandardizer(mean, std)


def _iou_payload(iou) -> dict:
    """Per-class IoU as JSON-safe floats. NaN (class absent) becomes null."""
    payload = {}
    for name, value in zip(CLASS_NAMES, iou.tolist()):
        value = float(value)
        payload[name] = None if math.isnan(value) else value
    return payload


def _count_pairs(counts: torch.Tensor, logits: torch.Tensor, masks: torch.Tensor) -> None:
    """Add this batch's (true, predicted) pixel pairs to ``counts`` in place.

    ``counts`` is a flat ``NUM_CLASSES ** 2`` tensor on the training device,
    so the training metric needs no copy to the CPU and no GPU sync inside
    the step; it is read once per epoch. Entry ``t * NUM_CLASSES + p``
    counts pixels of true class ``t`` predicted as ``p`` (the layout of
    ``sw.metrics.confusion_matrix``).

    Labels are not range-checked here, since that check would force a sync:
    an out-of-range label already fails in the loss, and it would also
    make ``counts`` longer than ``NUM_CLASSES ** 2``, which fails when the
    epoch reshapes it.
    """
    pred = logits.argmax(dim=1)
    pairs = masks.reshape(-1).long() * NUM_CLASSES + pred.reshape(-1)
    if counts.device.type == "mps":  # count on the CPU there, to be safe
        counts += torch.bincount(pairs.cpu(), minlength=NUM_CLASSES**2).to(counts.device)
    else:
        counts += torch.bincount(pairs, minlength=NUM_CLASSES**2)


def _evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    meter = ConfusionMeter()
    with torch.no_grad():
        for cubes, masks in loader:
            logits = model(cubes.to(device))
            meter.update(logits, masks)
    return meter.fg_miou(), _iou_payload(meter.per_class_iou())


def _has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


def _check_tokenizer(cfg: TrainConfig) -> None:
    """The ``tokenizer`` rules, worded for the CLI.

    ``_check_cfg`` and ``build_model`` both call this, so ``main`` (which
    builds the model before ``train`` checks the config) reports a bad
    combination with the flag names.
    """
    if cfg.tokenizer not in ("band", "collapsed"):
        raise ValueError(f"tokenizer must be 'band' or 'collapsed', got {cfg.tokenizer!r}")
    if cfg.tokenizer != "band" and cfg.model != "transformer":
        raise ValueError("--tokenizer applies only to --model transformer")
    if cfg.tokenizer == "collapsed" and cfg.depth_spectral > 0:
        raise ValueError(
            f"--tokenizer collapsed needs --depth-spectral 0, got {cfg.depth_spectral}: with one "
            "band group a spectral block is a per-token MLP, not attention over bands. "
            "The parameter-matched control is --depth-spectral 0 --depth-spatial 8 "
            "(matched against the default 2+6 model)"
        )


def _check_cfg(cfg: TrainConfig) -> None:
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.epochs}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    # Only the CNN has BatchNorm, which cannot estimate a variance from one image.
    if cfg.model != "transformer" and cfg.batch_size < 2:
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
    if cfg.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {cfg.warmup_steps}")
    if cfg.clip_grad_norm is not None and not cfg.clip_grad_norm > 0:
        raise ValueError(f"clip_grad_norm must be > 0 (leave it unset for no clipping), got {cfg.clip_grad_norm}")
    if not 0.0 <= cfg.cosine_floor <= 1.0:
        raise ValueError(f"cosine_floor must be between 0 and 1, got {cfg.cosine_floor}")
    if cfg.resume is not None and not Path(cfg.resume).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {cfg.resume}")
    if cfg.save_every < 1 or cfg.eval_every < 1 or cfg.keep_last < 0:
        raise ValueError(
            f"save_every and eval_every must be >= 1 and keep_last >= 0, got "
            f"{cfg.save_every}, {cfg.eval_every}, {cfg.keep_last}"
        )
    if cfg.max_minutes is not None and not cfg.max_minutes > 0:
        raise ValueError(f"max_minutes must be > 0 (leave it unset for no limit), got {cfg.max_minutes}")
    if cfg.init_from is not None and not Path(cfg.init_from).is_file():
        raise FileNotFoundError(f"init checkpoint does not exist: {cfg.init_from}")
    if cfg.init_optimizer and cfg.init_from is None:
        raise ValueError("--init-optimizer needs --init-from")
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
        if cfg.crop_size is not None and cfg.crop_size % cfg.patch != 0:
            raise ValueError(
                f"crop_size {cfg.crop_size} must be a multiple of patch {cfg.patch}; "
                "the tokenizer drops the remainder and the head only pads it"
            )
        if cfg.pe_init_std <= 0:
            raise ValueError(f"pe_init_std must be > 0, got {cfg.pe_init_std}")
    if cfg.head_prior_bias and cfg.model != "transformer":
        raise ValueError("--head-prior-bias applies only to --model transformer")
    _check_tokenizer(cfg)


def _log_mode(path: Path, resume: bool) -> str:
    """Append a metrics file on resume. A new run replaces it."""
    if resume and path.is_file() and path.stat().st_size > 0:
        return "a"
    return "w"


def _atomic_save(payload, path) -> None:
    """``torch.save`` to a temporary file, then rename it over ``path``.

    A session that dies during the write leaves the previous checkpoint
    whole, never a truncated one that ``--resume`` cannot load.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """What optimizer step ``step`` (counting from 1) multiplies the scheduled rate by."""
    if warmup_steps < 1:
        return 1.0
    return min(1.0, step / warmup_steps)


def _rng_state(loader) -> dict:
    """Every RNG a resume must continue, so a resumed run draws what an unbroken one would."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader.generator.get_state(),
    }
    return state


def _restore_rng(state, loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def _snapshot(out_dir: Path, epoch: int, keep: int) -> None:
    """Copy ``last.pt`` to ``ckpt_eNNNNN.pt`` (atomically) and keep only the ``keep`` newest."""
    target = out_dir / f"ckpt_e{epoch:05d}.pt"
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(out_dir / "last.pt", tmp)
    os.replace(tmp, target)
    for old in sorted(out_dir.glob("ckpt_e*.pt"))[:-keep]:
        old.unlink()


def _init_from(cfg, model, optimizer, scheduler, device) -> None:
    """Weights (and with ``init_optimizer`` the AdamW moments) of another run; everything else is new."""
    ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{cfg.init_from} has no 'model'")
    model.load_state_dict(ckpt["model"])
    if cfg.init_optimizer:
        if "optimizer" not in ckpt:
            raise KeyError(f"{cfg.init_from} has no optimizer; use a last.pt or ckpt_e*.pt, not best.pt")
        optimizer.load_state_dict(ckpt["optimizer"])
        # The loaded groups carry the parent's rates; this run's flags win.
        for group in optimizer.param_groups:
            group.update(lr=cfg.lr, initial_lr=cfg.lr, weight_decay=cfg.weight_decay)
        if scheduler is not None:
            scheduler.base_lrs = [cfg.lr] * len(optimizer.param_groups)
    print(
        f"init from {cfg.init_from}  (parent epoch {ckpt.get('epoch')}, "
        f"optimizer {'taken' if cfg.init_optimizer else 'fresh'})"
    )


def _save_last(
    path, model, optimizer, scheduler, scaler, epoch, step, best, best_epoch, stall, cfg, weights, use_amp, rng=None
):
    """Resume checkpoint. ``best.pt`` stays the best weights only.

    ``step`` is the number of optimizer steps taken so far, for the warmup.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_fg_miou": None if best_epoch is None else best,
        "best_epoch": best_epoch,
        "stall": stall,
        "class_weight_scheme": cfg.weight_scheme,
        "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
        "cfg": asdict(cfg),
    }
    if rng is not None:
        payload["rng"] = rng
    if use_amp:
        payload["scaler"] = scaler.state_dict()
    _atomic_save(payload, path)
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
    saved_schedule = ckpt.get("scheduler") if scheduler is not None else None
    if saved_schedule is not None:
        # The restored scheduler keeps the floor and the length it was saved
        # with. Check both against this run's flags before anything is loaded.
        asked_floor, saved_floor = float(scheduler.eta_min), float(saved_schedule.get("eta_min", 0.0))
        if not math.isclose(asked_floor, saved_floor, rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError(
                f"{path} was trained with a cosine floor of {saved_floor:g} (learning rate); "
                f"--lr times --cosine-floor is now {asked_floor:g}. A resume keeps the saved schedule, "
                "so pass the same --lr and --cosine-floor, or start a new run."
            )
        asked_epochs, saved_epochs = int(scheduler.T_max), int(saved_schedule.get("T_max", scheduler.T_max))
        if asked_epochs > saved_epochs:
            print(
                f"WARNING: {path} holds a cosine schedule {saved_epochs} epochs long and --epochs is {asked_epochs}. "
                f"The schedule keeps its length: after epoch {saved_epochs} the learning rate climbs back up. "
                "To train longer with cosine, start a new run with the larger --epochs."
            )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is None:
        if ckpt.get("scheduler") is not None:
            print("cosine is off for this run; not restoring the scheduler")
    elif saved_schedule is not None:
        scheduler.load_state_dict(saved_schedule)
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
    if ckpt.get("step") is not None:
        step = int(ckpt["step"])
    else:
        # A last.pt written before the harness counted optimizer steps.
        # AdamW keeps its own count per parameter; it is the same number.
        counts = [slot["step"] for slot in optimizer.state_dict()["state"].values() if "step" in slot]
        step = max((int(value.item() if torch.is_tensor(value) else value) for value in counts), default=0)
    return {
        "start_epoch": finished + 1,
        "step": step,
        "best": best,
        "best_epoch": best_epoch,
        "stall": int(ckpt.get("stall", 0)),
        "rng": ckpt.get("rng"),
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
    call_start = time.perf_counter()
    # BatchNorm (the CNN) needs 2 or more images per training batch. The
    # check reads the model too, so a BatchNorm model is caught whatever
    # cfg.model says, before any data is read or any folder is made.
    needs_two = cfg.model != "transformer" or _has_batchnorm(model)
    if needs_two and cfg.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 so BatchNorm does not train on a batch of 1, got {cfg.batch_size}"
        )
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    standardizer = _resolve_standardizer(cfg)

    # augment=False: no flips, and a training crop is a center crop.
    train_ds = _split(
        cfg, "train", cfg.max_train_images, train=cfg.augment, standardizer=standardizer
    )
    val_ds = _split(
        cfg, "val", cfg.max_val_images, train=False, standardizer=standardizer
    )
    if needs_two and len(train_ds) < 2:
        raise RuntimeError(
            "Need at least 2 training images. BatchNorm cannot train on a batch of 1."
        )
    drop_last = needs_two and _train_drop_last(len(train_ds), cfg.batch_size)
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
        # cosine_floor 0.0 gives eta_min 0.0, which is the scheduler's own default.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * cfg.cosine_floor
        )
    elif cfg.cosine_floor > 0:
        print("cosine_floor applies to the cosine schedule only; cosine is off, so it is ignored")
    use_amp = bool(cfg.amp) and device.type == "cuda"
    if cfg.amp and not use_amp:
        print(f"amp applies to cuda only; device is {device.type}, training in full precision")
    # enabled=False is a no-op, so cpu and mps take the same step as a plain backward.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -math.inf
    best_epoch = None
    stall = 0
    start_epoch = 1
    global_step = 0  # optimizer steps taken over the whole run, resumes included
    if cfg.resume:
        restored = _load_resume(cfg.resume, model, optimizer, scheduler, scaler, device, use_amp)
        start_epoch = restored["start_epoch"]
        global_step = restored["step"]
        best = restored["best"]
        best_epoch = restored["best_epoch"]
        stall = restored["stall"]
        if restored["rng"] is not None:
            _restore_rng(restored["rng"], train_loader)
    elif cfg.init_from:
        _init_from(cfg, model, optimizer, scheduler, device)

    n_params = count_parameters(model)
    limited = cfg.max_train_steps is not None or cfg.max_train_images is not None
    schedule = "cosine" if scheduler is not None else "constant"
    if scheduler is not None and cfg.cosine_floor > 0:
        schedule += f" (floor {cfg.lr * cfg.cosine_floor:.6g})"
    clip_shown = "off" if cfg.clip_grad_norm is None else f"{cfg.clip_grad_norm:g}"
    print(f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    if isinstance(model, BandTransformer):
        # Read from the model, not cfg, so the log names what is training:
        # the fused model and its control differ only in this line and the count.
        # A model object pickled before ``tokenizer=`` existed has no
        # ``tokenizer_kind``. It can only be the band model.
        print(
            f"transformer  tokenizer {getattr(model, 'tokenizer_kind', 'band')}  "
            f"blocks {model.depth_spectral}+{model.depth_spatial}  dim {model.dim}  "
            f"heads {model.heads}  mlp_ratio {float(model.mlp_ratio)}  patch {model.patch}"
        )
    print(
        f"device {device}  epochs {start_epoch}-{cfg.epochs}  batch {cfg.batch_size}  "
        f"lr {cfg.lr}  weight_decay {cfg.weight_decay}  schedule {schedule}  amp {'on' if use_amp else 'off'}"
    )
    if cfg.warmup_steps > 0 or cfg.clip_grad_norm is not None:
        print(f"warmup steps {cfg.warmup_steps} (done {min(global_step, cfg.warmup_steps)})  clip grad norm {clip_shown}")
    print(
        f"train images {len(train_ds)}  val images {len(val_ds)}  "
        f"downsample {cfg.downsample}  crop {cfg.crop_size}  "
        f"augment {'on' if cfg.augment else 'off'}  out {out_dir}"
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
    if cfg.resume:
        # last.pt can be older than the log (--save-every, or a kill between the two writes).
        from sw.gates import _trim_logs

        _trim_logs(out_dir, start_epoch - 1)
    jsonl_mode = _log_mode(jsonl_path, cfg.resume is not None)
    csv_mode = _log_mode(csv_path, cfg.resume is not None)

    stopped = None
    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jsonl_file, csv_path.open(
        csv_mode, newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, cfg.epochs + 1):
            stopped = None
            model.train()
            loss_sum = 0.0
            seen = 0
            steps = 0
            # Training-set confusion counts from the predictions each step
            # already makes: train mode, before that step's update. They lag
            # the weights by up to one epoch, and with dropout or drop path
            # on they read low. No extra forward pass; counted on the device.
            train_counts = torch.zeros(NUM_CLASSES**2, dtype=torch.long, device=device)
            # The rate the schedule set for this epoch. Warmup scales it per
            # step and never writes it back: CosineAnnealingLR computes the
            # next epoch's rate from the one it finds in the optimizer.
            scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
            # Before clipping, over the steps with a finite norm; read once per epoch.
            grad_norm_max = torch.zeros((), device=device)
            grad_nonfinite = torch.zeros((), dtype=torch.long, device=device)
            epoch_start = time.perf_counter()
            for cubes, masks in train_loader:
                if needs_two and cubes.shape[0] < 2:
                    raise RuntimeError(
                        f"training batch has {cubes.shape[0]} image(s); "
                        "BatchNorm needs at least 2. The train loader should have dropped this tail."
                    )
                cubes = cubes.to(device)
                masks = masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(cubes)
                    loss = criterion(logits, masks)
                _count_pairs(train_counts, logits.detach(), masks)
                scaler.scale(loss).backward()
                if cfg.warmup_steps > 0:
                    factor = _warmup_factor(global_step + 1, cfg.warmup_steps)
                    for group, scheduled in zip(optimizer.param_groups, scheduled_lrs):
                        group["lr"] = scheduled * factor
                if cfg.clip_grad_norm is not None:
                    # Under --amp the gradients are still scaled; clip the real ones.
                    # unscale_ is a no-op when the scaler is off (cpu, mps, no --amp).
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                    # A NaN or inf norm is counted, not kept: under --amp GradScaler
                    # skips such a step, and one of them should not hide the rest.
                    grad_norm = torch.as_tensor(grad_norm, device=device).detach().float()
                    finite = torch.isfinite(grad_norm)
                    grad_norm_max = torch.where(finite, torch.maximum(grad_norm_max, grad_norm), grad_norm_max)
                    grad_nonfinite += (~finite).long()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                loss_sum += float(loss.item()) * cubes.shape[0]
                seen += cubes.shape[0]
                steps += 1
                if cfg.max_train_steps is not None and steps >= cfg.max_train_steps:
                    break
            if steps == 0:
                raise RuntimeError("no training steps ran; check batch_size and the split size")
            train_loss = loss_sum / seen
            train_cm = train_counts.cpu().numpy().reshape(NUM_CLASSES, NUM_CLASSES)  # the one copy per epoch
            train_seconds = time.perf_counter() - epoch_start
            train_score, train_iou = fg_miou_from_cm(train_cm)
            # Stop after this epoch when another would not fit in --max-minutes.
            out_of_time = (
                cfg.max_minutes is not None
                and time.perf_counter() - call_start + 1.3 * train_seconds > cfg.max_minutes * 60
            )
            evaluated = epoch % cfg.eval_every == 0 or epoch == cfg.epochs or out_of_time
            if evaluated:
                score, per_class = _evaluate(model, val_loader, device)
            else:
                score, per_class = math.nan, {name: None for name in CLASS_NAMES}
            epochs_ran = epoch
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_fg_miou": None if math.isnan(score) else score,
                "steps": steps,
                "per_class_iou": per_class,
                "train_fg_miou": None if math.isnan(train_score) else train_score,
                "train_per_class_iou": _iou_payload(train_iou),
                "train_seconds": round(train_seconds, 3),  # the training pass
                "epoch_seconds": round(time.perf_counter() - epoch_start, 3),  # training + validation
            }
            if cfg.clip_grad_norm is not None:
                nonfinite = int(grad_nonfinite.item())
                # null when no step of the epoch had a finite norm, like val_fg_miou.
                record["grad_norm_max"] = None if nonfinite >= steps else float(grad_norm_max.item())
                record["grad_nonfinite_steps"] = nonfinite
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
            train_shown = "nan" if math.isnan(train_score) else f"{train_score:.4f}"
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:02d}  steps {steps:4d}  lr {current_lr:.6g}  "
                f"train_loss {train_loss:.4f}  train_fg_miou {train_shown}  val_fg_miou {shown}  "
                f"{record['epoch_seconds']:.1f}s"
            )

            improved = evaluated and math.isfinite(score) and score > best
            if improved:
                best = score
                best_epoch = epoch
                stall = 0
                _atomic_save(
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
            elif evaluated:
                stall += 1
            if cfg.warmup_steps > 0:
                # Hand the schedule back its own rate (see scheduled_lrs above).
                for group, scheduled in zip(optimizer.param_groups, scheduled_lrs):
                    group["lr"] = scheduled
            if scheduler is not None:
                scheduler.step()
            if evaluated and not improved and cfg.patience is not None and stall >= cfg.patience:
                stopped = "early_stop"
            elif out_of_time and epoch < cfg.epochs:
                stopped = "time_budget"
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs or stopped:
                _save_last(
                    out_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best,
                    best_epoch,
                    stall,
                    cfg,
                    weights,
                    use_amp,
                    rng=_rng_state(train_loader),
                )
                if cfg.keep_last > 0:
                    _snapshot(out_dir, epoch, cfg.keep_last)
            if stopped == "early_stop":
                print(f"early stop at epoch {epoch} (patience {cfg.patience})")
                break
            if stopped:
                print(f"time budget: stopping after epoch {epoch}; resume to carry on (--max-minutes {cfg.max_minutes:g})")
                break

    result = _result(best, best_epoch, epochs_ran, n_params, out_dir, jsonl_path)
    if stopped == "time_budget":
        result["stopped"] = "time_budget"
    return result


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
        help="cnn is MiniNetSeg. transformer is the band-token transformer (BandTransformer).",
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
        help=(
            "Bands per tokenizer group (transformer). 224 / 16 = 14 groups. "
            "No effect with --tokenizer collapsed."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        choices=["band", "collapsed"],
        default="band",
        help=(
            "Transformer only. band is BandTokenizer (band-group tokens). collapsed mixes all bands "
            "with a 1x1 conv before the patch embedding (one group): the band-collapsed control. "
            "It needs --depth-spectral 0; with --depth-spatial 8 it matches the default model's "
            "parameters (491,191 against 488,487). --group and --pe-init-std have no effect with it. "
            "Give the control its own --out-dir: the default, runs/transformer, is the fused model's too."
        ),
    )
    parser.add_argument("--data-root", default=None, help="Dataset folder. Default: the SW_ROOT environment variable.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default: 4, or 2 with --smoke. The transformer may use 1; the CNN needs 2 or more.",
    )
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
        "--no-aug",
        action="store_true",
        help="No training flips, and a center crop. For the overfit-one-batch check.",
    )
    parser.add_argument(
        "--no-cosine",
        action="store_true",
        help="Constant learning rate. The default is CosineAnnealingLR over --epochs.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear warmup over the first W optimizer steps (not epochs): step k uses "
            "the scheduled rate times min(1, k / W). Default 0, no warmup."
        ),
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Clip the global gradient norm of all parameters to this value before each step. Default: no clipping.",
    )
    parser.add_argument(
        "--cosine-floor",
        type=float,
        default=0.0,
        help="With cosine on, the learning rate never falls below --lr times this. Default 0, anneal to zero.",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Write last.pt every N epochs (and on the last or a stop). Default 1.")
    parser.add_argument("--keep-last", type=int, default=0, help="Also keep the K newest ckpt_eNNNNN.pt copies of last.pt. Default 0, none.")
    parser.add_argument("--eval-every", type=int, default=1, help="Validate every N epochs (and on the last or a stop). Default 1.")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Stop cleanly after a saved epoch once another would not fit in this many minutes from now. Resume later.",
    )
    parser.add_argument("--init-from", default=None, help="Start a NEW run from the weights in this checkpoint (a fork).")
    parser.add_argument("--init-optimizer", action="store_true", help="With --init-from, also take the AdamW moments (not best.pt).")
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
        "--standardize",
        action="store_true",
        help=(
            "Per-band mean/std from the TRAIN split only, cached and applied to "
            "train/val/test. Deviates from the paper's [0,1] scaling; see docs/INPUT-BLIND.md."
        ),
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Cache path for --standardize. Default: {out-dir}/band_stats.pt (not committed).",
    )
    parser.add_argument(
        "--pe-init-std",
        type=float,
        default=1.0,
        help=(
            "IndexBandPE init scale (transformer). Default 1.0 matches today's N(0,1). "
            "Try 0.02 with --standardize. No effect with --tokenizer collapsed (it has no IndexBandPE)."
        ),
    )
    parser.add_argument(
        "--head-prior-bias",
        action="store_true",
        help="Init SegHead classifier bias to log train class priors (transformer only).",
    )
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
        batch_size=args.batch_size if args.batch_size is not None else (2 if smoke else 4),
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
        augment=not args.no_aug,
        standardize=bool(args.standardize),
        stats_path=args.stats_path,
        pe_init_std=args.pe_init_std,
        head_prior_bias=bool(args.head_prior_bias),
        tokenizer=args.tokenizer,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        cosine_floor=args.cosine_floor,
        save_every=args.save_every,
        keep_last=args.keep_last,
        eval_every=args.eval_every,
        max_minutes=args.max_minutes,
        init_from=args.init_from,
        init_optimizer=bool(args.init_optimizer),
    )


def build_model(cfg: TrainConfig, in_channels: int = 224) -> nn.Module:
    """``cnn`` builds ``MiniNetSeg``. ``transformer`` builds ``BandTransformer``.

    The transformer size fields live on ``cfg``. ``in_channels`` is the CLI
    ``--in-channels`` value (real cubes are 224) and is the tokenizer's
    band count. CNN runs ignore the transformer size fields.

    When ``cfg.head_prior_bias`` is set, the SegHead classifier bias is
    initialised to ``log`` of the hyperspectral train class priors.

    ``cfg.tokenizer`` picks the transformer's tokenizer. A combination
    ``_check_cfg`` would refuse (``collapsed`` with spectral blocks, or
    with the CNN) is refused here too, before anything is built.
    """
    _check_tokenizer(cfg)
    if cfg.model == "cnn":
        return MiniNetSeg(in_channels=in_channels)
    if cfg.model == "transformer":
        model = BandTransformer(
            depth_spectral=cfg.depth_spectral,
            depth_spatial=cfg.depth_spatial,
            dim=cfg.dim,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            num_classes=7,
            n_bands=in_channels,
            group=cfg.group,
            patch=cfg.patch,
            pe_init_std=cfg.pe_init_std,
            tokenizer=cfg.tokenizer,
        )
        if cfg.head_prior_bias:
            init_seg_head_prior_bias(model.head, HYPER_LT_TRAIN_PIXEL_COUNTS)
        return model
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
