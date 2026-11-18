"""Checks for sw.gates: the overfit verdict, the subset root and resumable runs.

The verdict tests build metrics rows by hand, so every expected value is
known in advance. The run tests train MiniNetSeg for one or two epochs on
a tiny synthetic split. No real SpectralWaste files.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest
import tifffile
from PIL import Image

from sw import gates
from sw.data import CLASS_NAMES, SpectralWasteDataset

LARGE = gates.LARGE_CLASSES


def _row(epoch, loss=1.0, fg=0.0, **iou):
    """One metrics.jsonl row. Classes not given are 0.0 (present, never right)."""
    per_class = {name: 0.0 for name in CLASS_NAMES}
    per_class.update(iou)
    return {"epoch": epoch, "train_loss": loss, "train_fg_miou": fg, "train_per_class_iou": per_class}


def _all_large(value):
    return {name: value for name in LARGE}


# --- first_pass_epoch and judge_overfit ---------------------------------------------------


def test_first_pass_epoch_is_strictly_above_and_needs_every_class():
    rows = [
        _row(1),
        _row(2, **{**_all_large(0.95), "film": 0.9}),  # film exactly at the threshold is not above it
        _row(3, **{**_all_large(0.95), "basket": None}),  # null never counts as passing
        _row(4, **_all_large(0.901)),
        _row(5, **_all_large(0.99)),
    ]
    assert gates.first_pass_epoch(rows) == 4
    assert gates.first_pass_epoch(rows[:3]) is None
    assert gates.first_pass_epoch(rows, threshold=0.95) == 5


def test_pass_is_read_at_the_first_passing_epoch_so_a_later_spike_cannot_hide_it():
    rows = [_row(1, loss=2.0, fg=0.01)]
    rows += [_row(e, loss=1.0) for e in range(2, 300)]
    rows += [_row(300, loss=0.1, fg=0.8, **_all_large(0.92))]
    rows += [_row(e, loss=1.1, fg=0.2, **_all_large(0.2)) for e in range(301, 401)]  # spike, as seed 2 did
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "pass"
    assert verdict["first_pass_epoch"] == 300
    assert verdict["epochs"] == 400
    assert verdict["final"]["film"] == 0.2
    assert verdict["best"]["film"] == 0.92


def test_fail_names_the_classes_that_never_passed_and_flags_rising():
    rows = [_row(1, loss=2.0, fg=0.01)]
    for epoch in range(2, 201):
        film = 0.5 + 0.35 * epoch / 200  # 0.85 at the end: never above 0.9, still rising
        rows.append(_row(epoch, loss=2.0 / epoch, **{**_all_large(0.95), "film": film}))
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "fail"
    assert verdict["first_pass_epoch"] is None
    assert verdict["rising"] is True
    assert "film 0.850" in verdict["reason"]
    assert "basket" not in verdict["reason"]
    assert "still rising" in verdict["reason"]


def test_fail_when_the_loss_is_flat():
    rows = [_row(e, loss=1.9 - 0.0001 * e) for e in range(1, 121)]
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "fail"
    assert verdict["rising"] is False
    assert "flat" in verdict["reason"]


def test_not_rising_when_the_last_fifty_epochs_are_no_better():
    rows = [_row(1, loss=2.0, fg=0.01)] + [_row(e, loss=0.5, **_all_large(0.8)) for e in range(2, 151)]
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "fail"
    assert verdict["rising"] is False
    assert "still rising" not in verdict["reason"]
    assert "flat" not in verdict["reason"]  # 2.0 down to 0.5 is not flat


def test_all_four_must_pass_in_the_same_epoch():
    # Each class tops 0.95 in turn; no single epoch has all four above 0.9.
    rows = [_row(1, loss=2.0, fg=0.01)]
    for epoch in range(2, 10):
        turn = LARGE[epoch % 4]
        rows.append(_row(epoch, loss=0.2, **{**_all_large(0.85), turn: 0.95}))
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "fail"
    assert all(verdict["best"][name] == 0.95 for name in LARGE)
    assert verdict["closest_min"] == 0.85 and verdict["closest_epoch"] == 2
    assert "never all together" in verdict["reason"]
    assert "closest was epoch 2" in verdict["reason"]


def test_rising_compares_two_windows_of_fifty_from_one_hundred_rows():
    window = gates.RISING_WINDOW
    assert window == 50
    # Film improves once, at the start of the last window, then stays flat.
    rows = [_row(1, loss=2.0, fg=0.01, **{**_all_large(0.95), "film": 0.5})]
    rows += [_row(e, loss=0.5, **{**_all_large(0.95), "film": 0.5}) for e in range(2, 51)]
    rows += [_row(e, loss=0.5, **{**_all_large(0.95), "film": 0.6}) for e in range(51, 101)]
    assert len(rows) == 100
    assert gates.judge_overfit(rows)["rising"] is True  # exactly two windows is enough
    assert gates.judge_overfit(rows[:99])["rising"] is False  # fewer than two windows: never rising
    # Measured over the last 25 against the 25 before, this run would look flat.


def test_flat_loss_boundary():
    def run(final_loss):
        rows = [_row(1, loss=2.0, fg=0.01)] + [_row(e, loss=final_loss) for e in range(2, 20)]
        return gates.judge_overfit(rows)["reason"]

    assert "flat" in run(2.0 * gates.FLAT_LOSS_RATIO)  # never below 95% of epoch 1
    assert "flat" not in run(2.0 * gates.FLAT_LOSS_RATIO - 0.01)
    assert "flat" not in run(0.5)


def test_read_metrics_skips_blanks_a_half_written_last_line_and_keeps_the_last_row_per_epoch(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1, "v": "a"}\n\n{"epoch": 2, "v": "b"}\n{"epoch": 2, "v": "c"}\n{"epoch": 3, "v"')
    assert [(r["epoch"], r["v"]) for r in gates.read_metrics(path)] == [(1, "a"), (2, "c")]
    path.write_text('{"epoch": 2, "v": "b"}\n{"epoch": 1, "v": "a"}\n')
    assert [r["epoch"] for r in gates.read_metrics(path)] == [1, 2]
    path.write_text('{"epoch": 1}\nnot json\n{"epoch": 2}\n')
    with pytest.raises(ValueError, match="line 2"):
        gates.read_metrics(path)
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="no rows"):
        gates.read_metrics(path)


def test_invalid_when_a_judged_class_is_absent_from_the_images():
    # The first four official train images: only background, film and bag.
    absent = {"basket": None, "cardboard": None, "video_tape": None, "filament": None}
    rows = [_row(1, loss=2.0, fg=0.01, **absent)]
    rows += [_row(e, loss=0.1, fg=0.95, film=0.95, bag=0.95, **absent) for e in range(2, 50)]
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "invalid"
    assert "basket" in verdict["reason"] and "cardboard" in verdict["reason"]
    assert "film" not in verdict["reason"].split(":")[1]


def test_absence_comes_from_the_mask_pixels_when_they_are_given():
    # MiniNetSeg on the first four official images predicted some basket pixels
    # that are in no mask, so basket reads IoU 0.0 (not null) at every epoch.
    rows = [_row(1, loss=2.0, fg=0.01)] + [_row(e, loss=0.05, fg=0.6, film=0.99, bag=0.99) for e in range(2, 30)]
    assert gates.judge_overfit(rows)["verdict"] == "fail"  # the log alone cannot tell
    pixels = {name: 100 for name in CLASS_NAMES} | {"basket": 0, "cardboard": 0}
    verdict = gates.judge_overfit(rows, pixels=pixels)
    assert verdict["verdict"] == "invalid"
    assert "basket, cardboard" in verdict["reason"]
    # Present in the masks but never predicted (null IoU throughout) is a failure, not an absence.
    never = [_row(1, loss=2.0, fg=0.01, basket=None)] + [_row(e, loss=0.05, fg=0.7, **{**_all_large(0.99), "basket": None}) for e in range(2, 30)]
    verdict = gates.judge_overfit(never, pixels={name: 100 for name in CLASS_NAMES})
    assert verdict["verdict"] == "fail"
    assert "basket 0.000" in verdict["reason"]


def test_invalid_when_the_untrained_first_epoch_already_reads_high():
    rows = [_row(1, loss=2.0, fg=0.6, **_all_large(0.95))] + [_row(2, loss=0.1, fg=0.99, **_all_large(0.99))]
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "invalid"
    assert "epoch 1" in verdict["reason"]
    just_under = [_row(1, loss=2.0, fg=0.49)] + [_row(2, loss=0.1, fg=0.99, **_all_large(0.99))]
    assert gates.judge_overfit(just_under)["verdict"] == "pass"
    at_limit = [_row(1, loss=2.0, fg=0.5)] + just_under[1:]
    assert gates.judge_overfit(at_limit)["verdict"] == "invalid"


def test_thin_classes_are_reported_but_never_judged():
    rows = [_row(1, loss=2.0, fg=0.01), _row(2, loss=0.1, fg=0.6, filament=0.0, video_tape=0.1, **_all_large(0.95))]
    verdict = gates.judge_overfit(rows)
    assert verdict["verdict"] == "pass"
    assert verdict["final"]["filament"] == 0.0
    assert verdict["final"]["video_tape"] == 0.1


def test_judge_refuses_logs_it_cannot_read():
    old = [{"epoch": 1, "train_loss": 1.0, "val_fg_miou": 0.0}]  # harness before training-set metrics
    with pytest.raises(KeyError, match="train_per_class_iou"):
        gates.judge_overfit(old)
    with pytest.raises(ValueError, match="epoch 2"):
        gates.judge_overfit([_row(2), _row(3)])  # a resumed fragment
    with pytest.raises(ValueError):
        gates.judge_overfit([])


def test_overfit_constants():
    assert list(gates.OVERFIT_STEMS) == sorted(set(gates.OVERFIT_STEMS)) and len(gates.OVERFIT_STEMS) == 4
    assert set(LARGE) | set(gates.THIN_CLASSES) == set(CLASS_NAMES) - {"background"}
    assert not set(LARGE) & set(gates.THIN_CLASSES)


# --- subset root and class pixels -----------------------------------------------------------


def _write_pair(root, split, stem, classes, size=16):
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(hyper / f"{stem}.tiff", np.zeros((224, size, size), dtype=np.uint16), photometric="minisblack")
    mask = np.zeros((size, size), dtype=np.uint8)
    for row, cls in enumerate(classes):
        mask[row] = cls  # one full row of `size` pixels per listed class
    Image.fromarray(mask, mode="L").save(labels / f"{stem}.png")


def _full_root(tmp_path):
    root = tmp_path / "full"
    for index, stem in enumerate(["a", "b", "c", "d", "e"]):
        _write_pair(root, "train", stem, classes=[index + 1] if index < 6 else [])
    for stem in ["v1", "v2", "v3"]:
        _write_pair(root, "val", stem, classes=[6])
    return root


def test_subset_root_links_exactly_the_requested_cubes(tmp_path):
    root = _full_root(tmp_path)
    dest = gates.build_subset_root(root, tmp_path / "sub", train_stems=("d", "b"), n_val=2)
    for split, stems in (("train", ["b", "d"]), ("val", ["v1", "v2"])):
        for sub, ext in (("hyper", ".tiff"), ("labels_hyper_lt", ".png")):
            files = sorted(p.name for p in (dest / sub / split).iterdir())
            assert files == [f"{s}{ext}" for s in stems]
            for stem in stems:
                link = dest / sub / split / f"{stem}{ext}"
                assert link.is_symlink()
                assert link.resolve() == (root / sub / split / f"{stem}{ext}").resolve()
    ds = SpectralWasteDataset(dest, split="train", random_flip=False)
    assert [cube.stem for cube, _ in ds.pairs] == ["b", "d"]
    # Mask of b has one row of class 2, d one row of class 4: 16 pixels each.
    pixels = gates.class_pixels(dest, "train")
    assert pixels["basket"] == 16 and pixels["video_tape"] == 16
    assert pixels["film"] == 0 and pixels["background"] == 2 * 16 * 16 - 32
    assert gates.class_pixels(dest, "val")["bag"] == 2 * 16  # the last class counts too


def test_subset_root_rebuild_drops_stale_cubes_and_refuses_bad_input(tmp_path):
    root = _full_root(tmp_path)
    dest = tmp_path / "sub"
    gates.build_subset_root(root, dest, train_stems=("a", "b"), n_val=1)
    gates.build_subset_root(root, dest, train_stems=("c",), n_val=1)
    assert sorted(p.name for p in (dest / "hyper" / "train").iterdir()) == ["c.tiff"]
    with pytest.raises(FileNotFoundError):
        gates.build_subset_root(root, dest, train_stems=("zz",), n_val=1)
    with pytest.raises(FileNotFoundError):
        gates.build_subset_root(root, dest, train_stems=("a",), n_val=9)
    with pytest.raises(ValueError):
        gates.build_subset_root(root, root, train_stems=("a",), n_val=1)
    with pytest.raises(ValueError):
        gates.build_subset_root(root, dest, train_stems=("a", "a"), n_val=1)


# --- budget -----------------------------------------------------------------------------------


def test_budget_and_median_epoch_seconds():
    rows = [{"epoch_seconds": 50.0}, {"epoch_seconds": 10.0}, {"epoch_seconds": 12.0}, {"epoch_seconds": 30.0}]
    assert gates.median_epoch_seconds(rows) == 12.0  # epoch 1 (start-up) left out
    assert gates.median_epoch_seconds(rows, skip_first=False) == 21.0
    assert gates.median_epoch_seconds([{"epoch": 1}]) is None
    assert gates.budget_hours(36.0, 200, runs=3) == pytest.approx(6.0)


# --- resumable runs ---------------------------------------------------------------------------


def _train_root(tmp_path):
    root = tmp_path / "data"
    for split, n in (("train", 4), ("val", 2), ("test", 1)):
        for index in range(n):
            _write_pair(root, split, f"{split}_{index}", classes=list(range(1, 7)), size=32)
    return root


def _args(root, epochs):
    return [
        "--model", "cnn", "--data-root", str(root), "--epochs", str(epochs), "--batch-size", "2",
        "--max-train-images", "4", "--max-val-images", "2", "--num-workers", "0", "--device", "cpu",
        "--no-cosine",
    ]


def test_run_is_safe_to_repeat_and_resumes_a_partial_run(tmp_path, capsys):
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    assert gates.run_state(out) == "new"
    done = gates.run(out, _args(root, 1))
    assert gates.run_state(out) == "done"
    assert done["result"]["epochs_ran"] == 1 and done["resumed"] is False
    assert json.loads((out / gates.DONE_FILE).read_text())["train_args"] == _args(root, 1)
    assert not (out / gates.RUNNING_FILE).exists()

    # Repeating a finished run trains nothing.
    gates.run(out, _args(root, 1))
    assert [r["epoch"] for r in gates.read_metrics(out / "metrics.jsonl")] == [1]

    # Without DONE.json the run is partial: it resumes from last.pt and appends.
    (out / gates.DONE_FILE).unlink()
    assert gates.run_state(out) == "partial"
    first_row = gates.read_metrics(out / "metrics.jsonl")[0]
    capsys.readouterr()
    done = gates.run(out, _args(root, 3))
    printed = capsys.readouterr().out
    assert "resumed" in printed and "finished epoch 1" in printed
    assert done["resumed"] is True and done["result"]["epochs_ran"] == 3
    rows = gates.read_metrics(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[0] == first_row  # epoch 1 was not trained again


def test_a_failed_run_is_not_marked_done_and_releases_its_lock(tmp_path):
    out = tmp_path / "run"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        gates.run(out, _args(tmp_path / "no-data", 1))
    assert not (out / gates.DONE_FILE).exists()
    assert not (out / gates.RUNNING_FILE).exists()
    assert gates.run_state(out) == "new"


def _hold(out, pid=None, **extra):
    """Write RUNNING.json the way launch() and run() do: pid, and the start time on Linux."""
    out.mkdir(parents=True, exist_ok=True)
    record = gates._holder_record(os.getpid() if pid is None else pid)
    (out / gates.RUNNING_FILE).write_text(json.dumps({**record, **extra}))


def test_run_refuses_a_second_live_copy_and_ignores_a_dead_one(tmp_path):
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    _hold(out, pid=os.getppid())  # some other live process holds it
    assert gates.run_state(out) == "running"
    with pytest.raises(RuntimeError, match="already being trained"):
        gates.run(out, _args(root, 1))
    dead = 2**22 + 12345  # above the default pid range on Linux and macOS
    (out / gates.RUNNING_FILE).write_text(json.dumps({"pid": dead, "start": None}))
    assert gates.run_state(out) == "new"
    for junk in ("[]", "{", '{"pid": "x"}', "null"):
        (out / gates.RUNNING_FILE).write_text(junk)
        assert gates.run_state(out) == "new"
    with pytest.raises(ValueError, match="--out-dir"):
        gates.run(out, [*_args(root, 1), "--out-dir", str(out)])


def test_run_proceeds_when_the_record_is_its_own(tmp_path):
    # launch() writes the child's pid before the child starts; the child must not refuse itself.
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    _hold(out)
    assert gates.run_state(out) == "running"
    assert gates.run(out, _args(root, 1))["result"]["epochs_ran"] == 1
    assert not (out / gates.RUNNING_FILE).exists()


def _fake_proc(tmp_path, monkeypatch, pid, state, start):
    proc = tmp_path / "proc"
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    # A command name with spaces and a ")" inside, as /proc allows.
    # Fields 3 to 21 (19 of them), then 22 = the start time, then more.
    fields = [state, "1", "1", "1", "0", "-1", "0", *["0"] * 12, str(start), "0", "0"]
    assert len(fields[: fields.index(str(start))]) == 19
    (proc / str(pid) / "stat").write_text(f"{pid} (python -m sw) gates) " + " ".join(fields) + "\n")
    monkeypatch.setattr(gates, "_PROC", proc)


def test_linux_liveness_uses_the_start_time_and_skips_zombies(tmp_path, monkeypatch):
    pid = os.getpid()
    _fake_proc(tmp_path, monkeypatch, pid, "S", 4242)
    assert gates._stat(pid) == ("S", 4242)
    assert gates._holder_record(pid) == {"pid": pid, "start": 4242}
    assert gates._pid_alive(pid, 4242) is True
    assert gates._pid_alive(pid, 4243) is False  # the pid now belongs to a later process
    assert gates._pid_alive(pid, None) is True
    _fake_proc(tmp_path, monkeypatch, pid, "Z", 4242)
    assert gates._pid_alive(pid, 4242) is False  # exited, not yet reaped
    out = tmp_path / "run"
    out.mkdir()
    (out / gates.RUNNING_FILE).write_text(json.dumps({"pid": pid, "start": 4242}))
    assert gates.run_state(out) == "new"


def test_flags_that_change_the_run_are_refused_and_resumable_ones_are_not(tmp_path, capsys):
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    gates.run(out, _args(root, 1))
    assert json.loads((out / gates.ARGS_FILE).read_text()) == _args(root, 1)
    changed = [*_args(root, 1)]
    changed[changed.index("--batch-size") + 1] = "4"
    with pytest.raises(ValueError, match="different flags"):
        gates.run(out, changed)  # done, but asked for something else
    (out / gates.DONE_FILE).unlink()
    with pytest.raises(ValueError, match="different flags"):
        gates.run(out, changed)  # partial: a resume under new flags
    elsewhere = [*_args(tmp_path / "other-root", 2)]  # data root and epochs may change
    elsewhere[elsewhere.index("--data-root") + 1] = str(root)
    elsewhere[elsewhere.index("--device") + 1] = "cpu"
    assert gates.run(out, elsewhere)["result"]["epochs_ran"] == 2
    assert gates.run_identity(["--epochs=3", "--lr", "1e-3", "--device", "cuda"]) == ["--lr", "1e-3"]


def test_a_run_started_before_the_recipe_flags_still_resumes(tmp_path):
    """ARGS.json and last.pt from before --warmup-steps, --clip-grad-norm and
    --cosine-floor existed: the same flags resume it (the new ones default to
    off and are not in the identity), and turning one on is a different run."""
    import torch

    root = _train_root(tmp_path)
    out = tmp_path / "run"
    gates.run(out, _args(root, 1))
    (out / gates.DONE_FILE).unlink()
    old = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    del old["step"]  # the checkpoint as the harness wrote it before the recipe
    for name in ("warmup_steps", "clip_grad_norm", "cosine_floor"):
        del old["cfg"][name]
    torch.save(old, out / "last.pt")
    assert not any("warmup" in token or "clip" in token or "floor" in token for token in json.loads((out / gates.ARGS_FILE).read_text()))

    with pytest.raises(ValueError, match="different flags"):
        gates.run(out, [*_args(root, 2), "--clip-grad-norm", "1.0"])
    done = gates.run(out, _args(root, 2))
    assert done["resumed"] is True and done["result"]["epochs_ran"] == 2
    assert [r["epoch"] for r in gates.read_metrics(out / "metrics.jsonl")] == [1, 2]
    assert "grad_norm_max" not in gates.read_metrics(out / "metrics.jsonl")[-1]


def test_resume_drops_log_rows_written_after_the_checkpoint(tmp_path):
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    gates.run(out, _args(root, 1))
    (out / gates.DONE_FILE).unlink()
    # A session killed after epoch 2's metrics row but before its last.pt.
    with (out / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"epoch": 2, "train_loss": 9.0, "stale": True}) + "\n")
    with (out / "metrics.csv").open("a") as handle:
        handle.write("2,9.0,0.0,1" + ",0" * 7 + "\n")
    gates.run(out, _args(root, 2))
    lines = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert [row["epoch"] for row in lines] == [1, 2]
    assert not any(row.get("stale") for row in lines)
    assert [row.split(",")[0] for row in (out / "metrics.csv").read_text().splitlines()[1:]] == ["1", "2"]


def test_an_unreadable_checkpoint_stops_the_resume_with_advice(tmp_path):
    root = _train_root(tmp_path)
    out = tmp_path / "run"
    gates.run(out, _args(root, 1))
    (out / gates.DONE_FILE).unlink()
    (out / "last.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="Delete it to restart"):
        gates.run(out, _args(root, 2))
    assert not (out / gates.RUNNING_FILE).exists()


def test_done_records_the_commit_and_wall_time(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    done = gates.run(tmp_path / "run", _args(_train_root(tmp_path), 1), repo=repo)
    assert done["commit"] == head
    assert done["wall_seconds"] > 0


def _env():
    src = str(gates.Path(gates.__file__).parents[1])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}


def test_cli_and_background_launch(tmp_path):
    root = _train_root(tmp_path)
    assert gates._cli(["run", "--out-dir", str(tmp_path / "cli"), "--", *_args(root, 1)]) == 0
    assert gates.run_state(tmp_path / "cli") == "done"

    out = tmp_path / "bg"
    out.mkdir()
    (out / "train.log").write_text("previous session\n")
    proc = gates.launch(out, _args(root, 2), env=_env())
    assert proc is not None
    assert os.getsid(proc.pid) != os.getsid(0)  # its own session: a cell interrupt cannot reach it
    assert gates.launch(out, _args(root, 2), env=_env()) is None  # back to back: not a second copy
    assert proc.wait(timeout=120) == 0
    assert gates.run_state(out) == "done"
    log = (out / "train.log").read_text()
    assert log.startswith("previous session\n")  # appended, not replaced
    assert "done  best val FG mIoU" in log
    assert gates.launch(out, _args(root, 2), env=_env()) is None

    live = tmp_path / "live"
    _hold(live, pid=os.getppid())
    assert gates.launch(live, _args(root, 1), env=_env()) is None  # never a second copy
    assert not (live / "train.log").exists()
