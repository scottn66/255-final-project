"""Score a ``--head-skip`` checkpoint with each head path switched off.

With ``--head-skip`` the head's ``fuse`` conv reads two things at every
pixel: the decoder map that comes from the tokens (32 channels), and the
skip, a 1x1 conv of that pixel's own spectrum (16 channels). The skip
does not pass through the transformer. So a better score with the skip
could come from the skip alone. This probe measures that on a trained
checkpoint, on the validation split, with no retraining::

    PYTHONPATH=src python -m sw.skip_probe --checkpoint runs/tf_skip/best.pt

It prints val FG mIoU and per-class IoU three ways:

* ``normal``: the model as trained. This row must reproduce the score the
  run logged for that checkpoint, or the probe stops.
* ``skip zeroed``: the skip's channels are set to zero at the input of
  ``fuse``. Only the transformer path is left.
* ``tokens zeroed``: the decoder map's channels are set to zero at the
  input of ``fuse``. Only the skip path is left. No token, no attention
  and no neighbouring pixel reaches the logits.

**This is an ablation at test time, not a retrained model.** ``fuse`` and
the classifier were trained with both paths present. With one path zeroed
they see inputs they never saw in training. A row with one path off can
read lower or higher than the normal row, and it is not the score of a
model trained without that path. A high ``tokens zeroed`` row does show
that the skip alone carries that class. A low row shows less: the
trained model may still lean on both paths together, so a low
``tokens zeroed`` row does not show that the transformer is needed. Do
not compare the ``skip zeroed`` row with a run trained without the skip.

Why zeros and not the mean: the per-image mean of the decoder map still
carries what the tokens saw in that image (which classes are present, for
example), so a "skip only" row filled with it would not be skip only.
Zero carries nothing. ``--fill mean`` gives the other reading: each
channel of the switched-off path is replaced by its mean over the image.

The probe uses the run's own settings, read from the checkpoint:
``downsample``, ``crop_size``, ``max_val_images`` and ``standardize``.
With ``standardize`` it loads the run's ``band_stats.pt`` (beside the
checkpoint, or the run's ``--stats-path``) and stops if the file is
missing. It never computes new statistics. The split, the loader and the
metric are the ones ``sw.train`` uses for validation. It reads the
validation split only.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
from pathlib import Path

import torch

from sw.data import CLASS_NAMES
from sw.models.cnn import count_parameters
from sw.standardize import BandStandardizer, load_band_stats
from sw.train import (
    TrainConfig,
    _evaluate,
    _has_head_skip,
    _loader,
    _split,
    build_model,
    resolve_device,
)

# mode -> the label printed for its row.
MODES = {
    "normal": "normal",
    "skip_zeroed": "skip zeroed (transformer path only)",
    "tokens_zeroed": "tokens zeroed (skip path only)",
}
FILLS = ("zero", "mean")
# The normal row must be this close to the logged val FG mIoU and per-class IoU.
TOLERANCE = 1e-3

ABLATION_NOTE = (
    "Switching off a path of a jointly trained model is an ablation at test time, not a retrained model. "
    "fuse and the classifier were trained with both paths, so a row with one path off can read lower or higher "
    "than the normal row, and it is not the score of a model trained without that path. A high value for a class "
    "in the 'skip path only' row means the skip alone carries that class. A low 'skip path only' row does not show "
    "that the transformer is needed. Do not compare the 'transformer path only' row with a run trained without the skip. "
    "One checkpoint, validation only."
)


@contextlib.contextmanager
def ablate_head_path(model, mode: str, fill: str = "zero"):
    """Inside the ``with`` block, ``model`` runs with one head path switched off.

    ``mode`` is ``"normal"`` (nothing changes), ``"skip_zeroed"`` or
    ``"tokens_zeroed"``. The input of ``model.head.fuse`` is
    ``(B, hidden + skip_channels, H, W)``: the decoder map first, then the
    skip. A forward pre-hook replaces one of the two parts, with zeros
    (``fill="zero"``) or with each channel's mean over the image
    (``fill="mean"``). The weights are not touched, and the hook is removed
    when the block ends.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {list(FILLS)}, got {fill!r}")
    if not _has_head_skip(model):
        raise ValueError("this model has no head skip; there is no second path to switch off")
    if mode == "normal":
        yield model
        return
    hidden = model.head.hidden
    part = slice(hidden, None) if mode == "skip_zeroed" else slice(0, hidden)

    def replace(module, args):
        (joined,) = args
        joined = joined.clone()
        if fill == "zero":
            joined[:, part] = 0.0
        else:
            joined[:, part] = joined[:, part].mean(dim=(2, 3), keepdim=True)
        return (joined,)

    handle = model.head.fuse.register_forward_pre_hook(replace)
    try:
        yield model
    finally:
        handle.remove()


def _logged_score(checkpoint: Path, ckpt: dict):
    """``(val FG mIoU, per-class IoU)`` the run logged for this checkpoint's epoch, else ``(None, None)``.

    ``best.pt`` stores both. ``last.pt`` does not, so its epoch is looked
    up in the ``metrics.jsonl`` beside it.
    """
    if ckpt.get("val_fg_miou") is not None:
        return float(ckpt["val_fg_miou"]), ckpt.get("per_class_iou")
    from sw.gates import read_metrics

    try:
        rows = read_metrics(checkpoint.parent / "metrics.jsonl")
    except (FileNotFoundError, ValueError):
        return None, None
    for row in rows:
        if int(row["epoch"]) == int(ckpt.get("epoch", -1)) and row.get("val_fg_miou") is not None:
            return float(row["val_fg_miou"]), row.get("per_class_iou")
    return None, None


def _load_run(checkpoint: Path):
    """The checkpoint dict, its ``TrainConfig`` and the model with its weights (on the CPU)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "cfg" not in ckpt:
        raise KeyError(f"{checkpoint} is missing 'model' or 'cfg'; pass a best.pt or last.pt written by sw.train")
    state = ckpt["model"]
    if "head.skip.0.weight" not in state:
        raise ValueError(
            f"{checkpoint} was trained without --head-skip. Its head has one path, so there is nothing to probe."
        )
    names = {field.name for field in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{key: value for key, value in ckpt["cfg"].items() if key in names})
    cfg.resume = None
    n_bands = int(state["head.skip.0.weight"].shape[1])
    model = build_model(cfg, in_channels=n_bands)
    model.load_state_dict(state)  # strict: the checkpoint must be this exact model
    return ckpt, cfg, model


def _standardizer(cfg: TrainConfig, checkpoint: Path, stats_path, n_bands: int):
    """The run's own band standardizer, or ``(None, None)`` when the run did not standardize."""
    if not cfg.standardize:
        if stats_path is not None:
            raise ValueError("--stats-path was given, but this run was trained without --standardize")
        return None, None
    path = Path(stats_path or cfg.stats_path or checkpoint.parent / "band_stats.pt")
    if not path.is_file():
        raise FileNotFoundError(
            f"this run was trained with --standardize and its band statistics are not at {path}. "
            "The probe never computes new ones. Put the run's band_stats.pt back, or pass --stats-path."
        )
    mean, std = load_band_stats(path)
    if mean.numel() != n_bands:
        raise ValueError(f"{path} has {mean.numel()} bands; the model reads {n_bands}")
    return BandStandardizer(mean, std), path


def probe(
    checkpoint,
    data_root=None,
    stats_path=None,
    device: str = "auto",
    num_workers: int | None = None,
    batch_size: int | None = None,
    fill: str = "zero",
    tolerance: float = TOLERANCE,
) -> dict:
    """Score ``checkpoint`` on the validation split in the three modes of ``MODES``.

    Returns a dict with ``rows`` (mode -> ``{"val_fg_miou", "per_class_iou"}``),
    ``logged_val_fg_miou`` and ``difference`` (normal minus logged; both
    ``None`` when the run logged no score for this checkpoint), and what
    was read: ``checkpoint``, ``epoch``, ``param_count``, ``val_images``,
    ``data_root``, ``stats_path``, ``fill``, ``cfg``.

    ``data_root`` replaces the run's data root (a checkpoint from Colab
    names a Colab path); the default is the run's own, then ``SW_ROOT``.
    ``device``, ``num_workers`` and ``batch_size`` do not change the
    score: the metric sums one confusion matrix over the whole split.

    Raises ``RuntimeError`` when the normal row is more than ``tolerance``
    from the logged val FG mIoU or from a logged per-class IoU. Then the
    probe is not reading the validation set the way the run did, and the
    other rows would mean nothing.
    """
    checkpoint = Path(checkpoint)
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {list(FILLS)}, got {fill!r}")
    ckpt, cfg, model = _load_run(checkpoint)
    if data_root is not None:
        cfg.data_root = str(data_root)
    if num_workers is not None:
        cfg.num_workers = int(num_workers)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    n_bands = model.head.n_bands
    standardizer, used_stats = _standardizer(cfg, checkpoint, stats_path, n_bands)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    val_ds = _split(cfg, "val", cfg.max_val_images, train=False, standardizer=standardizer)
    loader = _loader(val_ds, cfg, shuffle=False, pin_memory=torch_device.type == "cuda", drop_last=False)

    rows = {}
    for mode in MODES:
        with ablate_head_path(model, mode, fill=fill):
            score, per_class = _evaluate(model, loader, torch_device)
        rows[mode] = {"val_fg_miou": None if math.isnan(score) else score, "per_class_iou": per_class}

    logged, logged_per_class = _logged_score(checkpoint, ckpt)
    difference = None
    normal = rows["normal"]
    if logged is not None:
        difference = (normal["val_fg_miou"] if normal["val_fg_miou"] is not None else float("nan")) - logged
        gaps = [abs(difference)]
        for name, value in (logged_per_class or {}).items():
            now = normal["per_class_iou"].get(name)
            if (value is None) != (now is None):
                gaps.append(float("inf"))
            elif value is not None:
                gaps.append(abs(now - value))
        if not max(gaps) <= tolerance:  # also true for nan
            raise RuntimeError(
                f"the normal row does not reproduce the logged score of {checkpoint}: "
                f"val FG mIoU {normal['val_fg_miou']} now, {logged} logged (largest gap {max(gaps):.2e}, "
                f"tolerance {tolerance:g}). The probe is not reading the validation set the way the run did: "
                "check --data-root and --stats-path. No ablation row is reported."
            )
    return {
        "checkpoint": str(checkpoint),
        "epoch": ckpt.get("epoch"),
        "param_count": count_parameters(model),
        "tokenizer": model.tokenizer_kind,
        "blocks": f"{model.depth_spectral}+{model.depth_spatial}",
        "skip": f"{n_bands}->{model.head.skip_channels}",
        "val_images": len(val_ds),
        "data_root": cfg.data_root,
        "stats_path": None if used_stats is None else str(used_stats),
        "device": str(torch_device),
        "fill": fill,
        "logged_val_fg_miou": logged,
        "difference": difference,
        "rows": rows,
        "cfg": dataclasses.asdict(cfg),
    }


def _cell(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_probe(result: dict) -> str:
    """The printed report for one :func:`probe` result."""
    cfg = result["cfg"]
    stats = f"standardize on ({result['stats_path']})" if result["stats_path"] else "standardize off"
    lines = [
        f"skip probe  {result['checkpoint']}  (epoch {result['epoch']})",
        f"model  BandTransformer  tokenizer {result['tokenizer']}  blocks {result['blocks']}  "
        f"parameters {result['param_count']:,}  head skip {result['skip']}",
        f"data   {result['data_root'] or 'SW_ROOT'}  val images {result['val_images']}  "
        f"downsample {cfg['downsample']}  crop {cfg['crop_size']}  {stats}  device {result['device']}",
    ]
    if result["logged_val_fg_miou"] is None:
        lines.append("logged val FG mIoU: none found for this checkpoint, so the normal row is not checked against the log")
    else:
        lines.append(
            f"logged val FG mIoU {result['logged_val_fg_miou']:.4f}; the normal row reproduces it "
            f"(difference {result['difference']:+.1e})"
        )
    if cfg.get("max_val_images") is not None:
        lines.append(f"Limited run: only the first {cfg['max_val_images']} validation images, as in the run. Not a score to report.")
    labels = dict(MODES)
    if result["fill"] == "mean":
        labels = {mode: label.replace("zeroed", "set to mean") for mode, label in MODES.items()}
    width = max(len(label) for label in labels.values())
    lines.append("")
    lines.append(f"{'mode':{width}s} {'FG mIoU':>7s} " + " ".join(f"{name[:10]:>10s}" for name in CLASS_NAMES))
    for mode, label in labels.items():
        row = result["rows"][mode]
        lines.append(
            f"{label:{width}s} {_cell(row['val_fg_miou']):>7s} "
            + " ".join(f"{_cell(row['per_class_iou'].get(name)):>10s}" for name in CLASS_NAMES)
        )
    lines.append("")
    if result["fill"] == "zero":
        lines.append("A zeroed path is set to 0 at the input of head.fuse. Weights are not changed.")
    else:
        lines.append(
            "--fill mean: a switched-off path is set to its per-image mean at the input of head.fuse, per channel. "
            "The mean of the decoder map still carries what the tokens saw in the image."
        )
    lines.append(ABLATION_NOTE)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sw.skip_probe",
        description="Score a --head-skip checkpoint on the validation split: normal, skip zeroed, tokens zeroed.",
    )
    parser.add_argument("--checkpoint", required=True, help="best.pt (or last.pt) of a run trained with --head-skip.")
    parser.add_argument(
        "--data-root", default=None, help="Dataset folder. Default: the run's own, then the SW_ROOT environment variable."
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        help="The run's band statistics. Default: the run's --stats-path, else band_stats.pt beside the checkpoint. Never computed here.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=None, help="Default: the run's.")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: the run's. Does not change the score.")
    parser.add_argument(
        "--fill",
        default="zero",
        choices=list(FILLS),
        help="What replaces a switched-off path: zero (the default), or each channel's mean over the image.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help=f"How close the normal row must be to the logged score. Default {TOLERANCE:g}.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    result = probe(
        args.checkpoint,
        data_root=args.data_root,
        stats_path=args.stats_path,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        fill=args.fill,
        tolerance=args.tolerance,
    )
    print(format_probe(result))
    return result


if __name__ == "__main__":
    main()
