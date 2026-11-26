"""Score a saved checkpoint on one split of SpectralWaste.

``sw.train`` scores the validation split only. The protocol is: pick the
checkpoint with the best validation FG mIoU, then score the test split
once. This module does that second step::

    PYTHONPATH=src python -m sw.evaluate --checkpoint <run>/best.pt --split test

The model is rebuilt from the ``cfg`` stored in the checkpoint, so no
training flags are needed. The band count is not in ``cfg``; it is read
from the first cube of the split. Weights are loaded strictly.

Preprocessing is the run's own: the same ``downsample``, and with
``standardize`` the same statistics file. That is ``cfg.stats_path`` when
the run set one, else ``band_stats.pt`` next to the checkpoint. A missing
statistics file is an error. It is never recomputed here, and a file the
run did not name is never used in its place. There is no flip and no crop.

The score is ``sw.metrics``: one confusion matrix over every pixel of the
split, then the mean IoU of classes 1..6. A class with no pixels in the
target and none in the prediction has no IoU and is left out of that mean.

Validation check. ``best.pt`` records its validation FG mIoU. ``--split
val`` compares its result with that number and prints MATCH or MISMATCH.
``--split test`` scores val first and stops on a mismatch, because a
mismatch means the model or the preprocessing is not the run's. The check
is skipped, with a note, for ``last.pt`` (it records no score of its own)
and for runs with ``crop_size`` or ``max_val_images`` (their log is not a
whole-split score).

The numbers are printed and written next to the checkpoint:
``eval_<split>.json`` for ``best.pt``, ``eval_<split>_<name>.json`` for a
checkpoint with another name. The test file is written once. A second
test run for the same checkpoint stops unless ``--force`` is given. A test
file left by another checkpoint (a retrain into the same folder) also
stops the run, and the message says the file is stale. A test folder that
does not hold the official number of images is refused unless
``--allow-partial-split`` is given.

``--max-images`` is a wiring check. It writes no file and its number is
not a score. On test it must be at most 8 and smaller than the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sw.data import CLASS_NAMES, SPLIT_COUNTS, SpectralWasteDataset
from sw.metrics import ConfusionMeter, fg_miou_from_cm
from sw.models.cnn import count_parameters
from sw.standardize import BandStandardizer, load_band_stats
from sw.train import TrainConfig, TransformedSplit, build_model, resolve_device

SPLITS = ("train", "val", "test")

ABSENT_RULE = (
    "A class with no pixels in the target and none in the prediction has no IoU (n/a) "
    "and is left out of the FG mean (sw.metrics)."
)

TEST_ONCE = (
    "The protocol is: choose the checkpoint on validation, then score test once per "
    "final configuration. A second look at test makes it a tuning set."
)

# Largest --max-images on test. A capped run is a wiring check, not a look at the split.
TEST_CAP = 8

# The val check passes when the two FG mIoU values differ by at most this.
# On the same machine they are equal. The slack is for another device.
VAL_MATCH_TOL = 1e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cfg_from_checkpoint(raw: dict) -> tuple[TrainConfig, list[str]]:
    """``TrainConfig`` from a checkpoint's ``cfg`` dict, plus the keys it did not know.

    A field the checkpoint lacks takes today's default. That is how a
    checkpoint saved before ``tokenizer`` existed loads as the band model.
    Keys this code does not know are left out and returned, so the caller
    can print them.
    """
    known = {field.name for field in fields(TrainConfig)}
    unknown = sorted(key for key in raw if key not in known)
    return TrainConfig(**{key: value for key, value in raw.items() if key in known}), unknown


def load_checkpoint(path) -> tuple[dict, TrainConfig, list[str]]:
    """Read a ``best.pt`` or ``last.pt`` and its config."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "cfg" not in ckpt:
        raise ValueError(f"{path} is missing 'model' or 'cfg'; it is not a sw.train checkpoint")
    cfg, unknown = cfg_from_checkpoint(dict(ckpt["cfg"]))
    return ckpt, cfg, unknown


def find_stats(cfg: TrainConfig, checkpoint: Path, override=None) -> Path | None:
    """The band statistics file the run trained with, or ``None`` without ``standardize``.

    ``override`` (``--stats-path``) wins. Else a run that set
    ``cfg.stats_path`` read that file and no other, so only that path is
    tried. Else the run wrote ``band_stats.pt`` in its own folder, so the
    file next to the checkpoint is used. A missing file raises. The
    statistics are never recomputed here.
    """
    if not cfg.standardize:
        return None
    if override is not None:
        candidate = Path(override)
        source = "--stats-path"
    elif cfg.stats_path is not None:
        candidate = Path(cfg.stats_path)
        source = "the stats_path in the checkpoint's cfg"
    else:
        candidate = Path(checkpoint).parent / "band_stats.pt"
        source = "band_stats.pt next to the checkpoint"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "this run trained with --standardize, and its band statistics file is missing "
        f"(looked at {source}: {candidate}). Pass the run's own file with --stats-path. "
        "It is not recomputed here, and no other file is used in its place: the score "
        "must use the same statistics the run trained with."
    )


def _payload(iou) -> dict:
    out = {}
    for name, value in zip(CLASS_NAMES, iou.tolist()):
        value = float(value)
        out[name] = None if math.isnan(value) else value
    return out


def result_path(checkpoint: Path, split: str) -> Path:
    """Where the numbers go: one file per checkpoint name and split."""
    checkpoint = Path(checkpoint)
    if checkpoint.stem == "best":
        return checkpoint.parent / f"eval_{split}.json"
    return checkpoint.parent / f"eval_{split}_{checkpoint.stem}.json"


def _confusion(model, base, n_images, cfg, standardizer, torch_device, batch_size, num_workers):
    """One confusion matrix over the first ``n_images`` whole images of ``base``."""
    dataset = TransformedSplit(
        base, range(n_images), None, cfg.downsample, train=False, standardizer=standardizer
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    meter = ConfusionMeter()
    with torch.no_grad():
        for cubes, masks in loader:
            meter.update(model(cubes.to(torch_device)), masks)
    return meter.cm


def _val_check_skip(ckpt: dict, cfg: TrainConfig) -> str | None:
    """Why the val check cannot run for this checkpoint, or ``None`` if it can."""
    if ckpt.get("val_fg_miou") is None:
        return "the checkpoint records no val score of its own (it is not a best.pt)"
    if cfg.crop_size is not None:
        return f"the run scored a center crop (crop_size {cfg.crop_size}), not whole images"
    if cfg.max_val_images is not None:
        return f"the run scored only the first {cfg.max_val_images} val images"
    return None


def _compare_val(score: float, recorded: float) -> dict:
    diff = abs(score - recorded) if math.isfinite(score) else None
    matched = diff is not None and diff <= VAL_MATCH_TOL
    return {
        "status": "match" if matched else "mismatch",
        "val_fg_miou": score if math.isfinite(score) else None,
        "recorded_val_fg_miou": recorded,
        "abs_diff": diff,
        "tolerance": VAL_MATCH_TOL,
        "reason": None,
    }


def evaluate_checkpoint(
    checkpoint,
    split: str = "val",
    *,
    data_root=None,
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    max_images: int | None = None,
    stats_path=None,
    force: bool = False,
    allow_partial_split: bool = False,
    allow_val_mismatch: bool = False,
) -> dict:
    """Score ``checkpoint`` on ``split`` and return the numbers.

    Writes the result file next to the checkpoint unless ``max_images`` is
    set. See the module docstring for the rules.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if max_images is not None and max_images < 1:
        raise ValueError(f"max_images must be >= 1, got {max_images}")
    checkpoint = Path(checkpoint)
    ckpt, cfg, unknown = load_checkpoint(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    capped = max_images is not None
    final_test = split == "test" and not capped
    out_path = result_path(checkpoint, split)
    notes = []

    # Every refusal comes before any scoring, so a refused run costs nothing
    # and shows nothing of the test split.
    base = SpectralWasteDataset(data_root, split=split, random_flip=False)
    if len(base) == 0:
        raise RuntimeError(f"{split} split is empty under {data_root or 'SW_ROOT'}")
    official = len(base) == SPLIT_COUNTS[split]
    if split == "test" and capped:
        if max_images > TEST_CAP or max_images >= len(base):
            raise ValueError(
                f"--max-images {max_images} on test is refused. A capped test run is a wiring "
                f"check: at most {TEST_CAP} images, and fewer than the split ({len(base)}). "
                "More than that is a look at the test score. Use --split val for a wiring check."
            )
    if not official:
        if final_test and not allow_partial_split:
            raise ValueError(
                f"the test split under {base.root} has {len(base)} images, the official split "
                f"has {SPLIT_COUNTS[split]}. A score on it is not the test score. Fix the data "
                "folder, or pass --allow-partial-split to score it anyway."
            )
        notes.append(
            f"NOT THE OFFICIAL SPLIT: {split} has {len(base)} images here, the official "
            f"split has {SPLIT_COUNTS[split]}"
        )

    replacing = final_test and out_path.is_file()
    if replacing:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        old_sha = old.get("checkpoint_sha256")
        old_score = old.get("fg_miou")
        if old_sha == checkpoint_sha:
            if not force:
                raise FileExistsError(
                    f"{out_path} already exists: this checkpoint was already scored on test. "
                    f"{TEST_ONCE} Read the file for the number. Pass --force to score again "
                    "and overwrite it."
                )
            notes.append(f"--force: this replaces an earlier test score (FG mIoU {old_score})")
        else:
            if not force:
                raise FileExistsError(
                    f"{out_path} already exists and belongs to another checkpoint "
                    f"(sha256 {str(old_sha)[:12]}, test FG mIoU {old_score}). {checkpoint.name} "
                    f"has changed since (sha256 {checkpoint_sha[:12]}), so that file is stale "
                    "and its number is not this checkpoint's. Move the file away if the old "
                    f"score is still wanted, or pass --force to replace it. {TEST_ONCE}"
                )
            notes.append(
                "--force: this replaces the test score of another checkpoint "
                f"(sha256 {str(old_sha)[:12]}, FG mIoU {old_score})"
            )

    if batch_size is None:
        batch_size = cfg.batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = resolve_device(device)

    stats_file = find_stats(cfg, checkpoint, stats_path)
    standardizer = None
    if stats_file is not None:
        standardizer = BandStandardizer(*load_band_stats(stats_file))
        if stats_path is None and not stats_file.is_absolute():
            notes.append(
                f"cfg.stats_path is relative ({stats_file}); it was read from {stats_file.resolve()}"
            )

    n_images = len(base) if max_images is None else min(len(base), max_images)
    in_channels = int(base[0][0].shape[0])
    if standardizer is not None and standardizer.n_bands != in_channels:
        raise ValueError(
            f"{stats_file} has {standardizer.n_bands} bands, the cubes have {in_channels}"
        )

    model = build_model(cfg, in_channels=in_channels)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{checkpoint} does not fit the model its cfg describes "
            f"({model.__class__.__name__}, {in_channels} bands). Weights are loaded strictly. "
            f"{error}"
        ) from error
    model = model.to(torch_device).eval()
    scoring = (cfg, standardizer, torch_device, batch_size, num_workers)

    # The val check. For test it runs first, so a wrong setup never reaches the test split.
    val_check = {"status": "not run", "reason": "only --split val and an uncapped --split test run it"}
    skip = _val_check_skip(ckpt, cfg)
    if capped or split == "train":
        pass
    elif skip is not None:
        val_check = {"status": "skipped", "reason": skip}
    elif split == "test":
        val_base = SpectralWasteDataset(data_root, split="val", random_flip=False)
        if len(val_base) == 0:
            raise RuntimeError(
                f"val split is empty under {data_root or 'SW_ROOT'}. A test run scores val "
                "first and compares it with the score the checkpoint recorded."
            )
        val_score, _ = fg_miou_from_cm(_confusion(model, val_base, len(val_base), *scoring))
        val_check = _compare_val(val_score, float(ckpt["val_fg_miou"]))
        if val_check["status"] == "mismatch":
            message = (
                f"val check MISMATCH: val FG mIoU is {val_score:.4f} here, the checkpoint "
                f"recorded {float(ckpt['val_fg_miou']):.4f}"
            )
            if not allow_val_mismatch:
                raise RuntimeError(
                    f"{message}. The model or the preprocessing is not the run's (data folder, "
                    "statistics file, code version). Test was not scored. Find the cause with "
                    "--split val. --allow-val-mismatch scores test anyway."
                )
            notes.append(message + "; test scored anyway (--allow-val-mismatch)")

    cm = _confusion(model, base, n_images, *scoring)
    score, iou = fg_miou_from_cm(cm)
    total = int(cm.sum())
    accuracy = float(cm.trace() / total) if total else None
    if split == "val" and not capped and skip is None:
        val_check = _compare_val(score, float(ckpt["val_fg_miou"]))

    if unknown:
        notes.append("cfg keys this code does not know, left out: " + ", ".join(unknown))
    if "tokenizer" not in ckpt["cfg"]:
        notes.append("cfg has no tokenizer entry (an older checkpoint): built as the band model")
    if cfg.crop_size is not None:
        notes.append(
            f"the run used crop_size {cfg.crop_size}. This score is on whole images, so on val "
            "it differs from the training log, which scored a center crop"
        )
    if split == "val" and cfg.max_val_images is not None:
        notes.append(
            f"the run scored only the first {cfg.max_val_images} val images, so its log "
            "differs from a whole-split score"
        )

    own_val = ckpt.get("val_fg_miou")
    best_val = ckpt.get("best_val_fg_miou")
    result = {
        "split": split,
        "n_images": n_images,
        "split_size": len(base),
        "official_split": official,
        "capped": capped,
        "fg_miou": None if math.isnan(score) else score,
        "per_class_iou": _payload(iou),
        "pixel_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "absent_class_rule": ABSENT_RULE,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": ckpt.get("epoch"),
        # The val score of these weights. Only best.pt records one.
        "checkpoint_val_fg_miou": None if own_val is None else float(own_val),
        # last.pt: the best score of the run so far, at another epoch.
        "best_val_fg_miou": None if best_val is None else float(best_val),
        "best_epoch": ckpt.get("best_epoch"),
        "val_check": val_check,
        "commit": ckpt.get("commit"),
        "model": model.__class__.__name__,
        "tokenizer": cfg.tokenizer if cfg.model == "transformer" else None,
        "parameters": count_parameters(model),
        "in_channels": in_channels,
        "downsample": cfg.downsample,
        "standardize": bool(cfg.standardize),
        "stats_path": None if stats_file is None else str(stats_file),
        "stats_sha256": None if stats_file is None else _sha256(stats_file),
        "device": str(torch_device),
        "batch_size": batch_size,
        "notes": notes,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cfg": dict(ckpt["cfg"]),
        "written": None,
    }
    if not capped:
        tmp = out_path.with_name(out_path.name + ".tmp")
        result["written"] = str(out_path)
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out_path)
    _print_result(result)
    return result


def _print_result(result: dict) -> None:
    def shown(value):
        return "n/a" if value is None else f"{value:.4f}"

    kind = result["model"]
    if result["tokenizer"] is not None:
        kind += f" (tokenizer {result['tokenizer']})"
    print(f"checkpoint {result['checkpoint']}")
    print(f"sha256 {result['checkpoint_sha256']}")
    if result["checkpoint_val_fg_miou"] is not None:
        recorded = f"recorded val FG mIoU {shown(result['checkpoint_val_fg_miou'])}"
    elif result["best_val_fg_miou"] is not None:
        recorded = (
            "no val score recorded for these weights  best val so far "
            f"{shown(result['best_val_fg_miou'])} (epoch {result['best_epoch']})"
        )
    else:
        recorded = "no val score recorded"
    print(
        f"epoch {result['checkpoint_epoch']}  {recorded}  "
        f"commit {result['commit'] or 'not recorded'}"
    )
    print(f"{kind}  parameters {result['parameters']:,}  bands {result['in_channels']}")
    stats = result["stats_path"] if result["standardize"] else "off"
    print(
        f"device {result['device']}  batch {result['batch_size']}  "
        f"downsample {result['downsample']}  standardize {stats}"
    )
    for note in result["notes"]:
        print(f"note: {note}")
    print(f"split {result['split']}  images {result['n_images']} of {result['split_size']}")
    for name in CLASS_NAMES:
        print(f"  iou {name:<11s} {shown(result['per_class_iou'][name])}")
    print(ABSENT_RULE)
    print(f"pixel accuracy {shown(result['pixel_accuracy'])}")
    print(f"{result['split']} FG mIoU {shown(result['fg_miou'])}")
    check = result["val_check"]
    if check["status"] == "match":
        print(
            f"val check: MATCH  val FG mIoU {shown(check['val_fg_miou'])}, "
            f"recorded {shown(check['recorded_val_fg_miou'])}"
        )
    elif check["status"] == "mismatch":
        print(
            f"val check: MISMATCH  val FG mIoU {shown(check['val_fg_miou'])}, recorded "
            f"{shown(check['recorded_val_fg_miou'])}. The model or the preprocessing is not "
            "the run's. Do not report a number from this setup."
        )
    elif check["status"] == "skipped":
        print(f"val check: skipped, {check['reason']}")
    if result["capped"]:
        print(
            "Capped run (--max-images): this checks the wiring. It is not a score to report, "
            "and no file was written."
        )
    else:
        print(f"wrote {result['written']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score a sw.train checkpoint on one split. The model and the preprocessing come "
            "from the checkpoint's cfg."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="best.pt of a run (last.pt also loads).")
    parser.add_argument(
        "--split",
        choices=list(SPLITS),
        default="val",
        help="val reproduces the training log. test is scored once per final configuration.",
    )
    parser.add_argument("--data-root", default=None, help="Dataset folder. Default: the SW_ROOT environment variable.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Default: cuda if available, else cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Default: the run's batch size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Score only the first N images. A wiring check: not a score, and no file is "
            "written. On test: at most 8, and fewer than the split."
        ),
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        help=(
            "Band statistics of a --standardize run, if the file moved. Default: the path in "
            "the checkpoint's cfg if the run set one, else band_stats.pt next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Score test again and overwrite eval_test.json. The protocol is test once.",
    )
    parser.add_argument(
        "--allow-partial-split",
        action="store_true",
        help="Score a test folder that does not hold the official 171 images. Not the test score.",
    )
    parser.add_argument(
        "--allow-val-mismatch",
        action="store_true",
        help="Score test even if val does not reproduce the checkpoint's recorded val score.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    return evaluate_checkpoint(
        args.checkpoint,
        args.split,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        stats_path=args.stats_path,
        force=args.force,
        allow_partial_split=args.allow_partial_split,
        allow_val_mismatch=args.allow_val_mismatch,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
