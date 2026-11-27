"""Lock and heartbeat, registry, fork and soup on synthetic folders. No real data, no GPU."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch

from dataclasses import asdict

from sw import gates, longrun
from sw.data import CLASS_NAMES
from sw.train import TrainConfig

TRAIN_ARGS = ["--model", "cnn", "--epochs", "4", "--seed", "3", "--lr", "3e-4"]
DATA = {"train": 4, "val": 2, "sha": "abc123"}


def _cfg(**overrides):
    return asdict(TrainConfig(model="cnn", **overrides))


def _fake_run(root, tag, scores, *, owner="ann", parent=None, status="running", beat_at=None, data=DATA, done=False):
    """A run folder as sw.train and sw.longrun leave it: metrics, metadata, lock."""
    out = Path(root) / tag
    out.mkdir(parents=True)
    rows = [
        {"epoch": i + 1, "train_loss": 1.0 / (i + 1), "val_fg_miou": s, "epoch_seconds": 2.0,
         "per_class_iou": {c: (s if c == "film" else 0.0) for c in CLASS_NAMES}}
        for i, s in enumerate(scores)
    ]
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    longrun.write_json(out / longrun.META_FILE, {
        "run_tag": tag, "owner": owner, "seed": 1, "lr": 3e-4, "epochs_target": 10, "git_sha": "deadbeef0",
        "data": data, "parent": parent,
    })
    if status:
        longrun.write_json(out / longrun.LOCK_FILE, {
            "owner": owner, "session": "s-" + tag, "session_start": 0.0,
            "heartbeat": time.time() if beat_at is None else beat_at, "status": status,
        })
    if done:
        longrun.write_json(out / gates.DONE_FILE, {"result": {}})
    return out


# --- lock and heartbeat ---


def test_a_live_heartbeat_refuses_a_second_claim_and_a_stale_one_allows_takeover(tmp_path):
    out = tmp_path / "run"
    first = longrun.claim(out, "ann", now=1000.0)
    assert longrun.run_status(out, now=1100.0)["state"] == "running"
    with pytest.raises(longrun.LockHeld, match="ann"):
        longrun.claim(out, "bob", now=1100.0)
    assert longrun.read_json(out / longrun.LOCK_FILE)["session"] == first  # the refused claim wrote nothing
    assert longrun.run_status(out, now=1000.0 + longrun.STALE_SECONDS + 1)["state"] == "stale"
    second = longrun.claim(out, "bob", now=1000.0 + longrun.STALE_SECONDS + 1)
    lock = longrun.read_json(out / longrun.LOCK_FILE)
    assert (lock["owner"], lock["session"]) == ("bob", second)
    # The old session sees it lost the lock: its beat does nothing.
    assert longrun.beat(out, first) is False
    assert longrun.beat(out, second) is True


def test_force_takes_over_a_live_lock_and_the_same_session_may_reclaim(tmp_path):
    out = tmp_path / "run"
    session = longrun.claim(out, "ann", now=1000.0)
    assert longrun.claim(out, "ann", session=session, now=1001.0) == session
    assert longrun.claim(out, "ann", force=True, now=1002.0) != session


def test_two_claims_at_the_same_moment_are_caught_by_the_read_back(tmp_path, monkeypatch):
    out = tmp_path / "run"
    real_sleep = time.sleep

    def rival_writes_meanwhile(seconds):
        longrun.write_json(out / longrun.LOCK_FILE, {"owner": "bob", "session": "bobs", "heartbeat": time.time(), "status": "running"})
        real_sleep(0)

    monkeypatch.setattr(longrun.time, "sleep", rival_writes_meanwhile)
    with pytest.raises(longrun.LockHeld, match="bob"):
        longrun.claim(out, "ann", settle=0.5)


def test_status_for_paused_failed_done_and_new(tmp_path):
    assert longrun.run_status(tmp_path / "none")["state"] == "new"
    for status in ("paused", "failed"):
        out = tmp_path / status
        session = longrun.claim(out, "ann")
        longrun.beat(out, session, status=status, error="boom")
        assert longrun.run_status(out)["state"] == status
        longrun.claim(out, "bob")  # a released or failed lock is free at once
    out = tmp_path / "done"
    longrun.claim(out, "ann")
    longrun.write_json(out / gates.DONE_FILE, {})
    assert longrun.run_status(out)["state"] == "done"


def test_free_tag_skips_only_live_folders(tmp_path):
    longrun.claim(tmp_path / "x", "ann")
    assert longrun.free_tag(tmp_path, "x") == "x-2"
    assert longrun.free_tag(tmp_path, "y") == "y"
    longrun.claim(tmp_path / "x-2", "bob", now=1.0)  # stale long ago
    assert longrun.free_tag(tmp_path, "x") == "x-2"


def test_heartbeat_thread_beats_and_exits_the_process_when_the_lock_is_lost(tmp_path, monkeypatch):
    out = tmp_path / "run"
    session = longrun.claim(out, "ann", now=1.0)
    heart = longrun.Heartbeat(out, session, interval=0.01)
    heart.start()
    deadline = time.time() + 5
    while longrun.read_json(out / longrun.LOCK_FILE)["heartbeat"] < time.time() - 5 and time.time() < deadline:
        time.sleep(0.01)
    assert longrun.run_status(out)["state"] == "running"
    exits = []
    monkeypatch.setattr(longrun.os, "_exit", lambda code: exits.append(code))
    longrun.claim(out, "bob", force=True)
    deadline = time.time() + 5
    while not exits and time.time() < deadline:
        time.sleep(0.01)
    heart.stop_event.set()
    assert exits and exits[0] == 3


def test_json_writes_are_atomic(tmp_path, monkeypatch):
    path = tmp_path / "LOCK.json"
    longrun.write_json(path, {"a": 1})

    def dies(self, text, encoding=None):
        with open(self, "w") as handle:
            handle.write("{half")
        raise OSError("session died")

    monkeypatch.setattr(Path, "write_text", dies)
    with pytest.raises(OSError):
        longrun.write_json(path, {"a": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"a": 1}


# --- registry ---


def test_scan_and_leaderboard_cover_every_state(tmp_path):
    now = time.time()
    _fake_run(tmp_path, "ann-live", [0.1, 0.3, 0.2], owner="ann")
    _fake_run(tmp_path, "bob-dead", [0.2], owner="bob", beat_at=now - 3 * longrun.STALE_SECONDS)
    _fake_run(tmp_path, "cy-failed", [], owner="cy", status="failed")
    _fake_run(tmp_path, "di-done", [0.4, 0.5], owner="di", status="done", done=True)
    _fake_run(tmp_path, "ed-fork", [0.45], owner="ed", parent={"tag": "di-done", "epoch": 2, "optimizer": False})
    (tmp_path / "scratch").mkdir()  # not a run
    rows = {r["tag"]: r for r in longrun.scan(tmp_path, now=now)}
    assert set(rows) == {"ann-live", "bob-dead", "cy-failed", "di-done", "ed-fork"}
    assert {t: r["state"] for t, r in rows.items()} == {
        "ann-live": "running", "bob-dead": "stale", "cy-failed": "failed", "di-done": "done", "ed-fork": "running",
    }
    assert rows["ann-live"]["epoch"] == 3 and rows["ann-live"]["best_val"] == pytest.approx(0.3)
    assert rows["ann-live"]["best_epoch"] == 2 and rows["ann-live"]["per_class"]["film"] == pytest.approx(0.3)
    assert rows["cy-failed"]["best_val"] is None and rows["cy-failed"]["epoch"] == 0
    assert rows["ed-fork"]["offset"] == 2 and rows["di-done"]["offset"] == 0
    ordered = [r["tag"] for r in longrun.scan(tmp_path, now=now)]
    assert ordered[:3] == ["di-done", "ed-fork", "ann-live"]  # best val first, no-score last
    text = longrun.leaderboard(longrun.scan(tmp_path, now=now))
    for needle in ("running", "stale", "failed", "done", "di-done@2", "per-class val IoU"):
        assert needle in text
    assert "WARNING" not in text


def test_leaderboard_warns_when_data_fingerprints_differ(tmp_path):
    _fake_run(tmp_path, "a", [0.1])
    _fake_run(tmp_path, "b", [0.2], data={"train": 3, "val": 2, "sha": "other"})
    assert "different data fingerprints" in longrun.leaderboard(longrun.scan(tmp_path))
    assert longrun.leaderboard([]) == "no runs yet"


def test_total_offset_follows_a_chain_of_forks(tmp_path):
    _fake_run(tmp_path, "a", [0.1], status=None)
    _fake_run(tmp_path, "b", [0.1], status=None, parent={"tag": "a", "epoch": 5})
    _fake_run(tmp_path, "c", [0.1], status=None, parent={"tag": "b", "epoch": 7})
    assert [longrun.total_offset(tmp_path, t) for t in "abc"] == [0, 5, 12]


def test_data_fingerprint_changes_with_the_file_names(tmp_path):
    for split, stems in (("train", ["t1", "t2"]), ("val", ["v1"])):
        (tmp_path / "hyper" / split).mkdir(parents=True)
        for stem in stems:
            (tmp_path / "hyper" / split / f"{stem}.tiff").write_bytes(b"")
    first = longrun.data_fingerprint(tmp_path)
    assert (first["train"], first["val"]) == (2, 1)
    (tmp_path / "hyper" / "train" / "t3.tiff").write_bytes(b"")
    assert longrun.data_fingerprint(tmp_path)["sha"] != first["sha"]


def test_metadata_records_the_run_and_refuses_other_flags_or_data(tmp_path):
    out = tmp_path / "run"
    meta = longrun.write_metadata(out, "ann", "s1", TRAIN_ARGS, Path("."), DATA)
    assert (meta["owner"], meta["seed"], meta["lr"], meta["epochs_target"]) == ("ann", 3, 3e-4, 4)
    assert meta["data"] == DATA and meta["flags"] == gates.run_identity(TRAIN_ARGS) and meta["parent"] is None
    again = longrun.write_metadata(out, "bob", "s2", [*TRAIN_ARGS, "--max-minutes", "5"], Path("."), DATA)
    assert again["owner"] == "ann" and [s["owner"] for s in again["sessions"]] == ["ann", "bob"]  # max-minutes may change
    with pytest.raises(ValueError, match="other flags"):
        longrun.write_metadata(out, "ann", "s3", [*TRAIN_ARGS, "--no-aug"], Path("."), DATA)
    with pytest.raises(ValueError, match="other data"):
        longrun.write_metadata(out, "ann", "s3", TRAIN_ARGS, Path("."), {**DATA, "sha": "zzz"})


# --- fork ---


def _checkpoint(path, value, epoch, *, cfg=None, optimizer=True):
    payload = {"model": {"w": torch.full((2, 2), float(value)), "n": torch.tensor(7)}, "epoch": epoch, "cfg": cfg or _cfg()}
    if optimizer:
        payload["optimizer"] = {"state": {}, "param_groups": []}
    torch.save(payload, path)


def test_fork_copies_the_checkpoint_and_records_the_parent(tmp_path):
    parent = tmp_path / "ann-base"
    parent.mkdir()
    _checkpoint(parent / "last.pt", 1.0, 40)
    _checkpoint(parent / "ckpt_e00020.pt", 2.0, 20)
    (parent / "band_stats.pt").write_bytes(b"stats")
    longrun.write_json(parent / longrun.META_FILE, {"data": DATA})
    child = tmp_path / "bob-lr1e-4"
    out = longrun.fork(parent, child, ["--model", "cnn", "--lr", "1e-4"], epoch=20)
    assert out["parent"] == {"tag": "ann-base", "epoch": 20, "optimizer": False}
    assert out["args"] == ["--init-from", str(child / "init.pt")]
    init = torch.load(child / "init.pt", weights_only=False)
    assert init["epoch"] == 20 and init["model"]["w"][0, 0] == 2.0 and "optimizer" not in init
    assert (child / "band_stats.pt").read_bytes() == b"stats"
    meta = longrun.read_json(child / longrun.META_FILE)
    assert meta["parent"] == out["parent"] and meta["data"] == DATA
    assert longrun.fork(parent, child, ["--model", "cnn", "--lr", "1e-4"], epoch=20)["parent"] == out["parent"]  # Run all again
    assert not list(child.glob("*.tmp"))


def test_fork_from_last_with_the_optimizer_keeps_it(tmp_path):
    parent = tmp_path / "p"
    parent.mkdir()
    _checkpoint(parent / "last.pt", 1.0, 9)
    out = longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], optimizer=True)
    assert out["parent"]["epoch"] == 9 and out["args"][-1] == "--init-optimizer"
    assert "optimizer" in torch.load(tmp_path / "c" / "init.pt", weights_only=False)


def test_fork_refuses_a_missing_snapshot_another_architecture_and_another_parent(tmp_path):
    parent = tmp_path / "p"
    parent.mkdir()
    _checkpoint(parent / "last.pt", 1.0, 9, optimizer=False)
    _checkpoint(parent / "ckpt_e00005.pt", 1.0, 5)
    with pytest.raises(FileNotFoundError, match="ckpt_e00005.pt"):
        longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], epoch=6)
    with pytest.raises(ValueError, match="architecture"):
        longrun.fork(parent, tmp_path / "c", ["--model", "transformer"])
    with pytest.raises(ValueError, match="no optimizer"):
        longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], optimizer=True)
    first = longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], epoch=5)
    again = longrun.fork(parent, tmp_path / "c", ["--model", "cnn"])  # a re-run keeps the fork it started from
    assert again == first and first["parent"]["epoch"] == 5
    _checkpoint(parent / "ckpt_e00009.pt", 1.0, 9)
    with pytest.raises(ValueError, match="already exists with parent"):
        longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], epoch=9)
    with pytest.raises(ValueError, match="without the optimizer"):
        longrun.fork(parent, tmp_path / "c", ["--model", "cnn"], optimizer=True)


# --- soup ---


def test_soup_state_averages_floats_and_keeps_integer_buffers(tmp_path):
    for name, value in (("a", 1.0), ("b", 3.0)):
        _checkpoint(tmp_path / f"{name}.pt", value, 1)
    state = longrun.soup_state([tmp_path / "a.pt", tmp_path / "b.pt"])
    assert torch.equal(state["w"], torch.full((2, 2), 2.0)) and state["n"] == 7 and state["w"].dtype == torch.float32
    torch.save({"model": {"other": torch.zeros(1)}}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="same parameters"):
        longrun.soup_state([tmp_path / "a.pt", tmp_path / "c.pt"])


def _forks(tmp_path, values):
    runs = []
    for i, value in enumerate(values):
        out = tmp_path / f"fork{i}"
        out.mkdir()
        _checkpoint(out / "best.pt", value, 10 + i)
        longrun.write_json(out / longrun.META_FILE, {"parent": {"tag": "base", "epoch": 5}, "data": DATA})
        runs.append(out)
    return runs


def test_soup_is_saved_only_when_it_beats_every_ingredient(tmp_path):
    runs = _forks(tmp_path, [1.0, 3.0])
    per_class = {c: 0.0 for c in CLASS_NAMES}

    def better(state):  # the soup (w = 2) scores highest
        return {1.0: 0.30, 3.0: 0.32, 2.0: 0.40}[float(state["w"][0, 0])], per_class

    result = longrun.soup_runs(runs, better, tmp_path / "soup", "ann")
    assert result["saved"] and result["soup_val_fg_miou"] == 0.40 and result["ingredients"] == {"fork0": 0.30, "fork1": 0.32}
    meta = longrun.read_json(tmp_path / "soup" / longrun.META_FILE)
    assert meta["soup_of"] == ["fork0", "fork1"] and meta["experimental"] is True and meta["parent"]["tag"] == "base"
    assert longrun.scan(tmp_path)[0]["tag"] == "soup" and longrun.scan(tmp_path)[0]["state"] == "done"

    def worse(state):
        return {1.0: 0.30, 3.0: 0.32, 2.0: 0.31}[float(state["w"][0, 0])], per_class

    result = longrun.soup_runs(runs, worse, tmp_path / "soup2", "ann")
    assert not result["saved"] and not (tmp_path / "soup2").exists()


def test_soup_needs_runs_from_one_parent(tmp_path):
    runs = _forks(tmp_path, [1.0, 3.0])
    longrun.write_json(runs[1] / longrun.META_FILE, {"parent": {"tag": "elsewhere", "epoch": 5}})
    with pytest.raises(ValueError, match="same parent"):
        longrun.soup_runs(runs, lambda s: (0.0, {}), tmp_path / "soup", "ann")
    with pytest.raises(ValueError, match="same parent"):
        longrun.soup_runs(runs[:1], lambda s: (0.0, {}), tmp_path / "soup", "ann")
