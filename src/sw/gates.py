"""Gates that must pass before any long transformer run, and resumable runs.

``notebooks/06_transformer_gates.ipynb`` runs the gates on Colab in order:
smoke, memory and step time, the overfit check, then the half-resolution
screening runs. This module holds everything that decides something, so
the decisions are tested code (``tests/test_gates.py``) and not notebook
cells:

* ``build_subset_root`` links the four overfit cubes into a small
  ``SW_ROOT``, because ``--max-train-images 4`` takes the *first* four
  training images, and those hold only background, film and bag.
* ``judge_overfit`` reads the overfit run's ``metrics.jsonl`` and returns
  pass or fail, with the reason.
* ``python -m sw.gates run --out-dir DIR -- <sw.train flags>`` runs one
  training job so that re-running it is safe:
  * it skips a finished run;
  * it resumes a partial one from ``last.pt``, first dropping log rows
    written after that checkpoint;
  * it refuses a second live copy, and refuses a run whose flags changed
    under the same folder;
  * at the end it writes ``DONE.json`` (result, commit, flags, wall time).

  A Colab session can die. Re-running the notebook cell continues the
  same runs.

Why the overfit rule is "the first epoch at which all four large classes
are above 0.9 *in that same epoch*, within 600 epochs": it was measured on
real cubes on Nov 16 (MPS, ``--downsample 2``, batch 4, one optimizer step
per epoch, lr 1e-3).

* The transformer first had every large class above 0.9 at epochs 337,
  447 and 433 (seeds 0, 1, 2). MiniNetSeg (one seed) did at epoch 43.
* Training FG mIoU stayed under 0.1 for the first 94 to 131 epochs: the
  model predicted almost only background. Learning rates 3e-4 and 3e-3
  were both slower than 1e-3.
* Seed 2 spiked at about epoch 550 (loss 0.10 to 1.13), then recovered.
  That happened at a constant learning rate with no gradient clipping.
  Reading the *first* passing epoch, not the last one, keeps a late
  spike from hiding a pass. The spike itself is a reason to land the
  optimiser recipe (warmup, clipping) before long runs.

The earlier rule (above 0.9 at epoch 200 or 400, on the first four images)
came from synthetic data. On real data it could check only two of the four
classes, and it passed or failed depending on the seed. This is a wiring
check on four training images. It is never a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sw.data import CLASS_NAMES

# Four official training cubes that together contain all six object classes.
# Chosen from the 514 train masks so that every large class has thousands of
# pixels (film 22,885, basket 28,310, cardboard 9,345, bag 41,538 at full
# resolution). Only 29 train images contain cardboard. Sorted, which is the
# order SpectralWasteDataset reads them.
OVERFIT_STEMS = (
    "20221011_02_084912",
    "20221028_02_092121",
    "20230117_08_112611",
    "20230119_02_095950",
)
LARGE_CLASSES = ("film", "basket", "cardboard", "bag")
# Reported, not judged. Without the full-resolution skip the head blurs
# boundaries by about a pixel, and these lines are 1 to 2 pixels wide.
THIN_CLASSES = ("video_tape", "filament")
PASS_IOU = 0.9
OVERFIT_EPOCHS = 600
# One resume, if the run fails at OVERFIT_EPOCHS while still rising.
OVERFIT_EXTEND_TO = 900
# "Rising" compares the last RISING_WINDOW epochs with the window before.
RISING_WINDOW = 50
# An untrained first epoch must read below this. Otherwise the check could
# not fail, and a pass would mean nothing.
EPOCH1_MAX_FG = 0.5
# The loss counts as flat if it never falls below this fraction of epoch 1.
FLAT_LOSS_RATIO = 0.95

DONE_FILE = "DONE.json"
RUNNING_FILE = "RUNNING.json"
ARGS_FILE = "ARGS.json"
# Flags that may change between sessions without changing what the run
# is: how many epochs in total, where the cubes are, the device, the loaders.
RESUMABLE_FLAGS = ("--epochs", "--data-root", "--device", "--num-workers")
_PROC = Path("/proc")

_SPLIT_FILES = (("hyper", ".tiff"), ("labels_hyper_lt", ".png"))

# Input-blind A/B (docs/INPUT-BLIND.md): first epoch whose train FG mIoU
# reaches this. Metrics are one row per epoch; steps_to_escape sums the
# per-epoch ``steps`` field through that epoch.
ESCAPE_FG = 0.05


def read_metrics(path) -> list[dict]:
    """The rows of a ``metrics.jsonl`` written by ``sw.train``, one per epoch, in epoch order.

    Blank lines are skipped. A last line that is not JSON yet is skipped
    too: a run may be writing it right now. If an epoch appears twice, for
    example after a crash and a resume, the later row wins.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    by_epoch = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise ValueError(f"{path}: line {index + 1} is not JSON") from None
        by_epoch[int(row["epoch"])] = row
    if not by_epoch:
        raise ValueError(f"{path} has no rows")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _iou(row: dict, name: str):
    return (row.get("train_per_class_iou") or {}).get(name)


def first_pass_epoch(rows, classes=LARGE_CLASSES, threshold=PASS_IOU):
    """The first epoch whose training IoU is above ``threshold`` for every class, else None.

    A class with a null IoU (absent from both the prediction and the
    target) does not count as above the threshold.
    """
    for row in rows:
        values = [_iou(row, name) for name in classes]
        if all(value is not None and value > threshold for value in values):
            return int(row["epoch"])
    return None


def summarize_escape(rows, threshold: float = ESCAPE_FG) -> dict:
    """When train FG mIoU first escapes the input-blind prior, from ``metrics.jsonl`` rows.

    ``sw.train`` logs one row per epoch, not per optimizer step. This
    helper finds the first epoch whose ``train_fg_miou`` is at least
    ``threshold`` (default ``ESCAPE_FG`` = 0.05) and reports:

    * ``escape_epoch`` / ``steps_to_escape`` — that epoch, and the sum of
      ``steps`` through it (optimizer steps so far). Both ``None`` if the
      run never cleared the threshold.
    * ``final_train_loss`` / ``final_train_fg_miou`` / ``best_val_fg_miou``
      — last-row loss and train FG, and the best val FG across the log.
    * ``epochs`` / ``total_steps`` — how far the log went.

    Used by ``notebooks/07_input_blind_ab.ipynb`` and
    ``python -m sw.gates summarize``.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no metrics rows")
    if "train_fg_miou" not in rows[0]:
        raise KeyError(
            "metrics rows have no train_fg_miou; need the harness with training-set metrics"
        )
    escape_epoch = None
    steps_to_escape = None
    cumulative = 0
    for row in rows:
        cumulative += int(row.get("steps") or 0)
        fg = row.get("train_fg_miou")
        if escape_epoch is None and fg is not None and float(fg) >= threshold:
            escape_epoch = int(row["epoch"])
            steps_to_escape = cumulative
    best_val = None
    for row in rows:
        v = row.get("val_fg_miou")
        if v is None:
            continue
        best_val = float(v) if best_val is None else max(best_val, float(v))
    last = rows[-1]
    last_fg = last.get("train_fg_miou")
    return {
        "threshold": threshold,
        "escape_epoch": escape_epoch,
        "steps_to_escape": steps_to_escape,
        "final_train_loss": float(last["train_loss"]),
        "final_train_fg_miou": None if last_fg is None else float(last_fg),
        "best_val_fg_miou": best_val,
        "epochs": int(last["epoch"]),
        "total_steps": cumulative,
    }


def format_escape_line(tag: str, summary: dict) -> str:
    """One monitor / table line for ``summarize_escape``."""
    esc = summary["escape_epoch"]
    if esc is None:
        esc_text = f"no escape (train FG never >= {summary['threshold']})"
    else:
        esc_text = f"escape epoch {esc} (~{summary['steps_to_escape']} steps)"
    final_fg = summary["final_train_fg_miou"]
    fg_text = "nan" if final_fg is None else f"{final_fg:.3f}"
    best_val = summary["best_val_fg_miou"]
    val_text = "nan" if best_val is None else f"{best_val:.3f}"
    return (
        f"{tag:28s} {esc_text}  "
        f"final loss {summary['final_train_loss']:.4f}  "
        f"train FG {fg_text}  best val FG {val_text}  "
        f"({summary['epochs']} ep / {summary['total_steps']} steps)"
    )


def judge_overfit(
    rows,
    classes=LARGE_CLASSES,
    threshold=PASS_IOU,
    epoch1_max=EPOCH1_MAX_FG,
    thin=THIN_CLASSES,
    pixels=None,
) -> dict:
    """Judge the overfit run from its ``metrics.jsonl`` rows.

    Returns a dict with ``verdict`` (``"pass"``, ``"fail"`` or
    ``"invalid"``), ``reason``, ``first_pass_epoch``, ``epochs`` (the last
    epoch logged), ``epoch1_fg``, ``closest_epoch`` and ``closest_min``
    (the epoch whose lowest large-class IoU was highest, and that IoU),
    ``rising`` (whether the weakest large class's best IoU over the last
    ``RISING_WINDOW`` epochs beats its best over the window before; False
    for logs shorter than two windows) and the per-class
    training IoU, ``final`` (last epoch) and ``best`` (max over epochs),
    for the judged and the thin classes.

    ``invalid`` means the check cannot be read at all: a judged class never
    appears in the images, or the untrained first epoch already reads high.

    ``pixels`` is the training masks' pixel count per class (from
    ``class_pixels``). Pass it. A class that is absent from the masks but
    predicted somewhere reads IoU 0.0, not null, so the log alone cannot
    tell "absent" from "never learned". Without ``pixels``, only a class
    whose IoU is null at every epoch counts as absent.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no metrics rows")
    if "train_per_class_iou" not in rows[0]:
        raise KeyError(
            "metrics rows have no train_per_class_iou; the overfit check needs the "
            "harness with training-set metrics (wp6-harness-gates or later)"
        )
    if int(rows[0]["epoch"]) != 1:
        raise ValueError(f"the log starts at epoch {rows[0]['epoch']}; judge a run from epoch 1")
    shown = tuple(classes) + tuple(name for name in thin if name not in classes)
    final = {name: _iou(rows[-1], name) for name in shown}
    best = {}
    for name in shown:
        seen = [_iou(row, name) for row in rows if _iou(row, name) is not None]
        best[name] = max(seen) if seen else None
    epoch1_fg = rows[0].get("train_fg_miou")
    passed = first_pass_epoch(rows, classes, threshold)

    def worst_best(chunk):
        # The class that fails the check is the worst one, so "rising" follows
        # the lowest per-class best IoU over the chunk, not the highest.
        return min(max((_iou(row, name) or 0.0) for row in chunk) for name in classes)

    window = RISING_WINDOW
    rising = len(rows) >= 2 * window and worst_best(rows[-window:]) > worst_best(rows[-2 * window : -window])
    # Passing needs every large class above the threshold in the same epoch,
    # so the near miss is the epoch whose lowest large-class IoU is highest.
    joint = [(min((_iou(row, name) or 0.0) for name in classes), int(row["epoch"])) for row in rows]
    closest_min, closest_epoch = max(joint, key=lambda pair: (pair[0], -pair[1]))
    result = {
        "first_pass_epoch": passed,
        "closest_epoch": closest_epoch,
        "closest_min": closest_min,
        "epochs": int(rows[-1]["epoch"]),
        "epoch1_fg": epoch1_fg,
        "rising": rising,
        "final": final,
        "best": best,
        "threshold": threshold,
        "classes": list(classes),
    }
    if pixels is not None:
        absent = [name for name in classes if int(pixels.get(name, 0)) == 0]
    else:
        absent = [name for name in classes if best[name] is None]
    if absent:
        return {
            **result,
            "verdict": "invalid",
            "reason": f"never present in the training images: {', '.join(absent)}; "
            "the check cannot judge them (use the cubes in OVERFIT_STEMS)",
        }
    if epoch1_fg is not None and epoch1_fg >= epoch1_max:
        return {
            **result,
            "verdict": "invalid",
            "reason": f"epoch 1 already reads train FG mIoU {epoch1_fg:.3f} (>= {epoch1_max}); "
            "an untrained model should read low, so this check cannot fail",
        }
    if passed is not None:
        return {
            **result,
            "verdict": "pass",
            "reason": f"every large class above {threshold} first at epoch {passed}",
        }
    first_loss = float(rows[0]["train_loss"])
    lowest = min(float(row["train_loss"]) for row in rows)
    if lowest >= FLAT_LOSS_RATIO * first_loss:
        reason = f"the loss is flat ({first_loss:.3f} at epoch 1, never below {lowest:.3f})"
    else:
        closest = next(row for row in rows if int(row["epoch"]) == closest_epoch)
        under = ", ".join(
            f"{name} {_iou(closest, name) or 0.0:.3f}"
            for name in classes
            if (_iou(closest, name) or 0.0) <= threshold
        )
        reason = (
            f"by epoch {result['epochs']} no epoch had every large class above {threshold} at once; "
            f"closest was epoch {closest_epoch}: {under}"
        )
        if all((best[name] or 0.0) > threshold for name in classes):
            reason += " (each class passed alone at some epoch, never all together)"
        if rising:
            reason += "; still rising, so resume to more epochs before calling it a bug"
    return {**result, "verdict": "fail", "reason": reason}


def _stems(root: Path, split: str) -> list[str]:
    return sorted(path.stem for path in (root / "hyper" / split).glob("*.tiff"))


def build_subset_root(sw_root, dest, train_stems=OVERFIT_STEMS, n_val=4) -> Path:
    """Link ``train_stems`` and the first ``n_val`` val cubes (and masks) into ``dest``.

    ``dest`` gets the ``SW_ROOT`` layout (``hyper/<split>/<stem>.tiff``,
    ``labels_hyper_lt/<split>/<stem>.png``) as symlinks to the files under
    ``sw_root``. Nothing is copied. The harness reads ``dest`` like any other
    ``SW_ROOT``. Files already in ``dest`` that are not in the subset are
    removed, so the split holds exactly the requested cubes. Returns ``dest``.
    """
    sw_root = Path(sw_root).expanduser().resolve()
    dest = Path(dest).expanduser()
    if dest.resolve() == sw_root:
        raise ValueError("dest must not be the full SW_ROOT")
    wanted = {"train": sorted(train_stems), "val": _stems(sw_root, "val")[:n_val]}
    if len(set(wanted["train"])) != len(wanted["train"]) or not wanted["train"]:
        raise ValueError(f"train_stems must be distinct and non-empty: {train_stems}")
    if len(wanted["val"]) < n_val:
        raise FileNotFoundError(f"{sw_root}/hyper/val has fewer than {n_val} cubes")
    for split, stems in wanted.items():
        for sub, ext in _SPLIT_FILES:
            folder = dest / sub / split
            folder.mkdir(parents=True, exist_ok=True)
            keep = {f"{stem}{ext}" for stem in stems}
            for path in folder.iterdir():
                if path.name not in keep:
                    path.unlink()
            for stem in stems:
                source = sw_root / sub / split / f"{stem}{ext}"
                if not source.is_file():
                    raise FileNotFoundError(f"missing {source}")
                link = folder / f"{stem}{ext}"
                if link.is_symlink() or link.exists():
                    if link.resolve() == source.resolve():
                        continue
                    link.unlink()
                link.symlink_to(source.resolve())
    return dest


def class_pixels(root, split="train") -> dict[str, int]:
    """Pixel count of each class over every mask in ``root/labels_hyper_lt/<split>``."""
    from PIL import Image

    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    masks = sorted((Path(root) / "labels_hyper_lt" / split).glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"no masks under {root}/labels_hyper_lt/{split}")
    for path in masks:
        mask = np.asarray(Image.open(path))
        counts += np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
    return {name: int(value) for name, value in zip(CLASS_NAMES, counts)}


def median_epoch_seconds(rows, skip_first=True):
    """Median ``epoch_seconds``, leaving out epoch 1 (it pays the start-up)."""
    values = [float(row["epoch_seconds"]) for row in rows if "epoch_seconds" in row]
    if skip_first and len(values) > 1:
        values = values[1:]
    return float(np.median(values)) if values else None


def budget_hours(epoch_seconds, epochs, runs=1) -> float:
    """GPU hours for ``runs`` runs of ``epochs`` epochs at ``epoch_seconds`` each."""
    return float(epoch_seconds) * int(epochs) * int(runs) / 3600.0


def _stat(pid: int):
    """Linux: ``(state, start)`` from ``/proc/<pid>/stat`` (fields 3 and 22), else None.

    ``start`` is when the process started, in clock ticks since boot.
    """
    try:
        text = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    # Field 2 (the command name) is in parentheses and may contain spaces;
    # the fields after it start at field 3, so field 22 is index 19.
    fields = text[text.rindex(")") + 2 :].split()
    return fields[0], int(fields[19])


def _start_time(pid: int):
    """Linux: when ``pid`` started (``/proc/<pid>/stat`` field 22), else None."""
    stat = _stat(pid)
    return None if stat is None else stat[1]


def _holder_record(pid: int) -> dict:
    """What ``RUNNING.json`` stores: the pid and, on Linux, its start time."""
    return {"pid": pid, "start": _start_time(pid)}


def _pid_alive(pid: int, start=None) -> bool:
    """True when ``pid`` is alive and, if ``start`` is known, is the same process.

    A recorded start time tells a live holder from a later process that
    reused its pid (Linux, so Colab). A zombie (exited, not yet reaped by
    its parent) is not alive. Without ``/proc`` (macOS), a live pid counts
    as the holder.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    stat = _stat(pid)
    if stat is not None:
        state, now = stat
        if state == "Z":
            return False
        if start is not None:
            return now == start
    return True


def _holder(out_dir: Path):
    """``(pid, start)`` from ``RUNNING.json``, or None if absent or unreadable."""
    try:
        info = json.loads((out_dir / RUNNING_FILE).read_text(encoding="utf-8"))
        return int(info["pid"]), info.get("start")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def run_state(out_dir) -> str:
    """``"done"``, ``"running"``, ``"partial"`` (has ``last.pt``) or ``"new"``."""
    out_dir = Path(out_dir)
    if (out_dir / DONE_FILE).is_file():
        return "done"
    holder = _holder(out_dir)
    if holder is not None and _pid_alive(*holder):
        return "running"
    if (out_dir / "last.pt").is_file():
        return "partial"
    return "new"


def _git_commit(cwd) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def run_identity(train_args) -> list[str]:
    """The flags that define a run: ``train_args`` without ``RESUMABLE_FLAGS`` and their values."""
    kept, skip = [], False
    for token in train_args:
        if skip:
            skip = False
            continue
        if token in RESUMABLE_FLAGS:
            skip = True
            continue
        if any(token.startswith(flag + "=") for flag in RESUMABLE_FLAGS):
            continue
        kept.append(token)
    return kept


def _check_identity(out_dir: Path, train_args) -> None:
    stored = out_dir / ARGS_FILE
    if not stored.is_file():
        return
    before = json.loads(stored.read_text(encoding="utf-8"))
    if run_identity(before) != run_identity(train_args):
        raise ValueError(
            f"{out_dir} was started with different flags.\n  before: {run_identity(before)}\n"
            f"  now:    {run_identity(train_args)}\nUse a new folder (a new tag), or delete this one."
        )


def _trim_logs(out_dir: Path, last_epoch: int) -> None:
    """Drop metrics rows past ``last_epoch``, the epoch ``last.pt`` holds.

    ``sw.train`` writes an epoch's metrics before its ``last.pt``. A run
    killed between the two leaves a row that the resume would write again.
    """
    jsonl = out_dir / "metrics.jsonl"
    if jsonl.is_file():
        kept, dropped = [], 0
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                epoch = int(json.loads(line)["epoch"])
            except (ValueError, KeyError, TypeError):
                dropped += 1
                continue
            if epoch <= last_epoch:
                kept.append(line)
            else:
                dropped += 1
        if dropped:
            jsonl.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            print(f"{jsonl}: dropped {dropped} row(s) after epoch {last_epoch} (written after the checkpoint)")
    table = out_dir / "metrics.csv"
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if rows:
            head, body = rows[0], rows[1:]
            keep = [row for row in body if row and row[0].isdigit() and int(row[0]) <= last_epoch]
            if len(keep) != len(body):
                with table.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerows([head, *keep])


def _checkpoint_epoch(path: Path) -> int:
    import torch

    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
    except Exception as err:  # a truncated or foreign file
        raise RuntimeError(
            f"{path} cannot be read ({type(err).__name__}: {err}). Delete it to restart this run "
            "from epoch 1, or put back a good copy."
        ) from err


def run(out_dir, train_args, repo=None) -> dict | None:
    """Run ``sw.train`` once into ``out_dir`` so that calling it again is safe.

    * ``DONE.json`` exists: return it; nothing runs. (To train longer,
      delete ``DONE.json`` and call again with the larger ``--epochs``.)
    * another live process holds ``RUNNING.json``: raise.
    * the flags differ from the ones the folder was started with (ignoring
      ``RESUMABLE_FLAGS``): raise.
    * ``last.pt`` exists: drop log rows past its epoch, then resume from it
      (``--resume``); the metrics append.
    * otherwise: a new run.

    ``train_args`` are ``sw.train`` flags without ``--out-dir`` or
    ``--resume``; this function adds both. On success it writes
    ``DONE.json`` (the result, the flags, the git commit, the wall time)
    before it releases ``RUNNING.json``. On failure there is no
    ``DONE.json``, so the next call resumes.

    ``ARGS.json`` (the flags the folder was started with) is written when
    the folder has none. If the folder had a ``last.pt`` and no
    ``ARGS.json``, and the resume fails before it finishes an epoch, the
    new ``ARGS.json`` is removed again, so flags that never trained here
    are not recorded as the folder's.
    """
    from sw.train import main as train_main

    out_dir = Path(out_dir)
    train_args = list(train_args)
    for flag in ("--out-dir", "--resume"):
        if flag in train_args:
            raise ValueError(f"pass {flag} through run(), not in train_args")
    state = run_state(out_dir)
    holder = _holder(out_dir)
    if state == "running" and holder is not None and holder[0] == os.getpid():
        # launch() wrote this process's own record before it started.
        state = "partial" if (out_dir / "last.pt").is_file() else "new"
    if state == "running":
        raise RuntimeError(f"{out_dir} is already being trained by another process")
    _check_identity(out_dir, train_args)
    if state == "done":
        done = json.loads((out_dir / DONE_FILE).read_text(encoding="utf-8"))
        print(f"{out_dir}: already done (epochs_ran {done['result'].get('epochs_ran')}); skipping")
        return done
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote_args = not (out_dir / ARGS_FILE).is_file()
    if wrote_args:
        (out_dir / ARGS_FILE).write_text(json.dumps(train_args), encoding="utf-8")
    argv = [*train_args, "--out-dir", str(out_dir)]
    last_before = None
    if state == "partial":
        last_before = (out_dir / "last.pt").stat().st_mtime_ns
        _trim_logs(out_dir, _checkpoint_epoch(out_dir / "last.pt"))
        argv += ["--resume", str(out_dir / "last.pt")]
    running = out_dir / RUNNING_FILE
    running.write_text(json.dumps({**_holder_record(os.getpid()), "argv": argv}), encoding="utf-8")
    start = time.time()
    try:
        result = train_main(argv)
        done = {
            "result": result,
            "train_args": train_args,
            "resumed": state == "partial",
            "commit": _git_commit(repo or Path(__file__).resolve().parent),
            "wall_seconds": round(time.time() - start, 1),
        }
        (out_dir / DONE_FILE).write_text(json.dumps(done, indent=2), encoding="utf-8")
    except BaseException:
        # A resume that failed before it finished an epoch (for example the
        # checkpoint was refused) added nothing to the folder. If this call
        # wrote ARGS.json, take it back: these flags are not the folder's.
        last = out_dir / "last.pt"
        if wrote_args and last_before is not None and last.is_file() and last.stat().st_mtime_ns == last_before:
            (out_dir / ARGS_FILE).unlink(missing_ok=True)
        raise
    finally:
        running.unlink(missing_ok=True)
    return done


def launch(out_dir, train_args, env=None, log_name="train.log") -> subprocess.Popen | None:
    """Start ``python -m sw.gates run`` in the background, in its own session.

    Interrupting a notebook cell does not stop it. The output goes to
    ``out_dir/<log_name>``, appended. The child's pid goes into
    ``RUNNING.json`` at once, so a second ``launch`` right after sees it
    running. Returns None, starting nothing, when the run is done or
    already running.
    """
    out_dir = Path(out_dir)
    state = run_state(out_dir)
    if state in ("done", "running"):
        print(f"{out_dir.name}: {state}; not starting another copy")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "sw.gates", "run", "--out-dir", str(out_dir), "--", *train_args]
    with open(out_dir / log_name, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    (out_dir / RUNNING_FILE).write_text(json.dumps(_holder_record(proc.pid)), encoding="utf-8")
    print(f"{out_dir.name}: started pid {proc.pid} ({state}); log {out_dir / log_name}")
    return proc


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sw.gates", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run sw.train once into --out-dir (skip if done, resume if partial)")
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("train_args", nargs=argparse.REMAINDER, help="sw.train flags after --")
    sum_parser = sub.add_parser(
        "summarize",
        help="print when train FG mIoU first escapes the input-blind prior (metrics.jsonl)",
    )
    sum_parser.add_argument(
        "paths",
        nargs="+",
        help="metrics.jsonl files, or run directories that contain them",
    )
    sum_parser.add_argument(
        "--threshold",
        type=float,
        default=ESCAPE_FG,
        help=f"train FG mIoU escape threshold (default {ESCAPE_FG})",
    )
    args = parser.parse_args(argv)
    if args.command == "summarize":
        for raw in args.paths:
            path = Path(raw)
            metrics = path / "metrics.jsonl" if path.is_dir() else path
            summary = summarize_escape(read_metrics(metrics), threshold=args.threshold)
            print(format_escape_line(path.name, summary))
        return 0
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    run(args.out_dir, train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
