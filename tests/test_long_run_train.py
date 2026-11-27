"""The long-run additions to sw.train (save-every, keep-last, eval-every, max-minutes, RNG resume, init-from)
and the lock around a real run. Tiny synthetic cubes, CPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from sw import gates, longrun
from sw.train import _build_parser, config_from_args, main
from test_train import _adam_step, _synthetic_root


def _common(root, out):
    return [
        "--data-root", str(root), "--out-dir", str(out), "--device", "cpu", "--batch-size", "2",
        "--downsample", "4", "--num-workers", "0", "--max-train-images", "4", "--max-val-images", "2",
        "--max-train-steps", "1", "--seed", "0",
    ]


def _gates_args(root, out):
    """sw.train flags without --out-dir, which sw.gates.run adds itself."""
    flags = _common(root, out)
    del flags[flags.index("--out-dir") : flags.index("--out-dir") + 2]
    return flags


def _rows(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "data"
    _synthetic_root(path)
    return path


def test_defaults_keep_todays_behaviour(root, tmp_path):
    cfg = config_from_args(_build_parser().parse_args([]))
    assert (cfg.save_every, cfg.keep_last, cfg.eval_every, cfg.max_minutes, cfg.init_from, cfg.init_optimizer) == (1, 0, 1, None, None, False)
    out = tmp_path / "run"
    result = main([*_common(root, out), "--epochs", "2"])
    assert "stopped" not in result
    assert not list(out.glob("ckpt_e*.pt")) and torch.load(out / "last.pt", weights_only=False)["epoch"] == 2
    assert all(row["val_fg_miou"] is not None for row in _rows(out))


def test_save_every_and_keep_last_keep_the_newest_snapshots_and_always_save_the_end(root, tmp_path):
    out = tmp_path / "run"
    main([*_common(root, out), "--epochs", "5", "--save-every", "2", "--keep-last", "2"])
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 5  # the final epoch is saved off-cycle
    assert sorted(p.name for p in out.glob("ckpt_e*.pt")) == ["ckpt_e00004.pt", "ckpt_e00005.pt"]
    assert not list(out.glob("*.tmp"))
    snap = torch.load(out / "ckpt_e00004.pt", weights_only=False)
    assert snap["epoch"] == 4 and "optimizer" in snap and "rng" in snap


def test_resume_drops_the_log_rows_the_checkpoint_does_not_cover(root, tmp_path):
    out = tmp_path / "run"
    main([*_common(root, out), "--epochs", "3", "--max-minutes", "1e-9"])  # last.pt holds epoch 1
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 1
    with (out / "metrics.jsonl").open("a") as handle:  # a row written after the last save, as with --save-every
        handle.write(json.dumps({"epoch": 2, "train_loss": 9.9, "val_fg_miou": 0.0}) + "\n")
    main([*_common(root, out), "--epochs", "3", "--resume", str(out / "last.pt")])
    rows = _rows(out)
    assert [r["epoch"] for r in rows] == [1, 2, 3] and rows[1]["train_loss"] != 9.9


def test_eval_every_skips_validation_and_patience_counts_validations(root, tmp_path):
    out = tmp_path / "run"
    main([*_common(root, out), "--epochs", "4", "--eval-every", "3"])
    rows = _rows(out)
    assert [r["val_fg_miou"] is not None for r in rows] == [False, False, True, True]  # epoch 3, and the last
    assert all(v is None for v in rows[0]["per_class_iou"].values())
    csv_rows = (out / "metrics.csv").read_text().splitlines()
    assert csv_rows[1].split(",")[2] == "" and csv_rows[3].split(",")[2] != ""
    assert torch.load(out / "best.pt", weights_only=False)["epoch"] in (3, 4)


def test_time_budget_stops_after_a_saved_epoch_and_gates_run_resumes_it(root, tmp_path):
    out = tmp_path / "run"
    args = _gates_args(root, out) + ["--epochs", "3"]
    first = gates.run(out, [*args, "--max-minutes", "1e-9"])
    assert first["paused"] and first["result"]["stopped"] == "time_budget" and first["result"]["epochs_ran"] == 1
    assert not (out / gates.DONE_FILE).exists()  # not finished: no DONE.json
    assert gates.run_state(out) == "partial"
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 1
    assert _rows(out)[0]["val_fg_miou"] is not None  # a stop always validates
    second = gates.run(out, [*args, "--max-minutes", "600"])  # --max-minutes may differ between sessions
    assert second["resumed"] and second["result"]["epochs_ran"] == 3 and (out / gates.DONE_FILE).is_file()
    assert [r["epoch"] for r in _rows(out)] == [1, 2, 3]


def test_a_resume_continues_the_random_stream_of_an_unbroken_run(root, tmp_path):
    straight, broken = tmp_path / "straight", tmp_path / "broken"
    flags = ["--epochs", "4", "--max-train-steps", "2"]
    main([*_common(root, straight), *flags])
    main([*_common(root, broken), *flags, "--max-minutes", "1e-9"])  # stops after epoch 1
    main([*_common(root, broken), *flags, "--resume", str(broken / "last.pt")])
    assert [r["train_loss"] for r in _rows(broken)] == [r["train_loss"] for r in _rows(straight)]


def test_init_from_starts_a_new_run_from_the_weights_and_optionally_the_moments(root, tmp_path, capsys):
    parent = tmp_path / "parent"
    main([*_common(root, parent), "--epochs", "2"])
    plain, moments = tmp_path / "plain", tmp_path / "moments"
    main([*_common(root, plain), "--epochs", "1", "--lr", "1e-2", "--init-from", str(parent / "last.pt")])
    main([*_common(root, moments), "--epochs", "1", "--lr", "1e-2", "--init-from", str(parent / "last.pt"), "--init-optimizer"])
    printed = capsys.readouterr().out
    assert "init from" in printed and "optimizer taken" in printed and "optimizer fresh" in printed
    a, b = (torch.load(d / "last.pt", weights_only=False) for d in (plain, moments))
    assert a["epoch"] == b["epoch"] == 1  # a new run counts its own epochs
    assert _adam_step(a) == 1 and _adam_step(b) == 3  # the moments came along: 2 parent steps + 1
    for ckpt in (a, b):
        group = ckpt["optimizer"]["param_groups"][0]
        assert group["initial_lr"] == pytest.approx(1e-2) and group["weight_decay"] == pytest.approx(1e-4)
    parent_weights = torch.load(parent / "last.pt", weights_only=False)["model"]
    assert not all(torch.equal(parent_weights[k], a["model"][k]) for k in parent_weights)  # it trained on from them


def test_resume_wins_over_init_from(root, tmp_path, capsys):
    parent, run = tmp_path / "parent", tmp_path / "run"
    main([*_common(root, parent), "--epochs", "1"])
    main([*_common(root, run), "--epochs", "2", "--max-minutes", "1e-9"])
    capsys.readouterr()
    main([*_common(root, run), "--epochs", "2", "--resume", str(run / "last.pt"), "--init-from", str(parent / "last.pt")])
    assert "init from" not in capsys.readouterr().out and torch.load(run / "last.pt", weights_only=False)["epoch"] == 2


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--save-every", "0"], "save_every"),
        (["--eval-every", "0"], "eval_every"),
        (["--keep-last", "-1"], "keep_last"),
        (["--max-minutes", "0"], "max_minutes"),
        (["--init-optimizer"], "needs --init-from"),
        (["--init-from", "nowhere.pt"], "does not exist"),
    ],
)
def test_new_flags_are_checked(root, tmp_path, flags, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        main([*_common(root, tmp_path / "run"), *flags])


def test_band_stats_save_is_atomic(tmp_path):
    from sw.standardize import save_band_stats

    save_band_stats(tmp_path / "band_stats.pt", torch.zeros(3), torch.ones(3))
    assert (tmp_path / "band_stats.pt").is_file() and not list(tmp_path.glob("*.tmp"))


# --- a real run under the lock ---


def _launch(root, out, extra=()):
    args = _gates_args(root, out) + ["--epochs", "2", *extra]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    return longrun.launch(out, "ann", args, Path(__file__).resolve().parents[1], root, env=env, settle=0.0)


def _wait(proc, seconds=120):
    deadline = time.time() + seconds
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    assert proc.poll() is not None, "run did not finish"


def test_launch_runs_under_a_lock_writes_metadata_and_leaves_the_lock_done(root, tmp_path):
    out = tmp_path / "runs" / "ann-first"
    proc, session = _launch(root, out)
    assert longrun.run_status(out)["state"] == "running"  # claimed before the child started
    with pytest.raises(longrun.LockHeld):
        longrun.claim(out, "bob")
    _wait(proc)
    assert proc.returncode == 0
    status = longrun.run_status(out)
    assert status["state"] == "done" and status["lock"]["session"] == session
    meta = longrun.read_json(out / longrun.META_FILE)
    assert meta["owner"] == "ann" and meta["seed"] == 0 and meta["git_sha"] and meta["data"]["train"] == 4
    assert longrun.launch(out, "ann", ["--epochs", "2"], Path("."), root) == (None, None)  # done: nothing to start
    row = longrun.scan(tmp_path / "runs")[0]
    assert (row["tag"], row["state"], row["epoch"], row["owner"]) == ("ann-first", "done", 2, "ann")


def test_a_crashed_run_is_marked_failed_with_the_reason_and_can_be_claimed_again(root, tmp_path):
    out = tmp_path / "runs" / "ann-bad"
    proc, _ = _launch(root, out, extra=["--init-from", str(tmp_path / "missing.pt")])
    _wait(proc)
    assert proc.returncode != 0
    status = longrun.run_status(out)
    assert status["state"] == "failed" and "does not exist" in status["lock"]["error"]
    assert "Traceback" in (out / "train.log").read_text()
    assert longrun.claim(out, "bob")  # a failed lock is free to take


def test_fork_two_children_and_score_them_with_the_evaluator_a_soup_uses(root, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    base = _gates_args(root, runs / "parent")
    main([*_common(root, runs / "parent"), "--epochs", "2", "--standardize"])
    data = longrun.data_fingerprint(root)
    longrun.write_metadata(runs / "parent", "ann", "s0", [*base, "--epochs", "2", "--standardize"], repo, data)
    children = []
    for tag, lr in (("fork-a", "1e-2"), ("fork-b", "3e-3")):
        args = [*base, "--epochs", "1", "--standardize", "--lr", lr, "--seed", "1"]
        forked = longrun.fork(runs / "parent", runs / tag, args)
        assert forked["parent"] == {"tag": "parent", "epoch": 2, "optimizer": False}
        main([*_common(root, runs / tag), "--epochs", "1", "--standardize", "--lr", lr, "--seed", "1", *forked["args"]])
        longrun.write_metadata(runs / tag, "bob", "s1", args, repo, data, parent=forked["parent"])
        children.append(runs / tag)
    evaluate = longrun.make_evaluator(children[0], root)
    best = torch.load(children[0] / "best.pt", weights_only=False)
    score, per_class = evaluate(best["model"])
    assert score == pytest.approx(best["val_fg_miou"]) and set(per_class) == set(best["per_class_iou"])
    result = longrun.soup_runs(children, longrun.make_evaluator(children[0], root), runs / "soup", "ann")
    assert set(result["ingredients"]) == {"fork-a", "fork-b"}
    assert result["saved"] == (result["soup_val_fg_miou"] > max(result["ingredients"].values()))
    assert (runs / "soup" / "best.pt").is_file() == result["saved"]
    rows = {r["tag"]: r for r in longrun.scan(runs)}
    assert rows["fork-a"]["parent"] == {"tag": "parent", "epoch": 2, "optimizer": False}
    assert rows["fork-a"]["offset"] == 2 and rows["fork-b"]["owner"] == "bob"
