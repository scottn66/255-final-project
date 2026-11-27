"""Long runs shared by 2 to 4 people on one Drive folder (``notebooks/09_long_run.ipynb``).

Layout, one folder per run tag under one shared root::

    <root>/<run_tag>/LOCK.json       who is training here, and a heartbeat
                     metadata.json   git sha, flags, seed, owner, data fingerprint, parent
                     last.pt best.pt ckpt_eNNNNN.pt metrics.jsonl metrics.csv train.log ...

* **Lock.** ``claim`` writes ``LOCK.json`` (owner, session, heartbeat). A lock
  whose heartbeat is newer than ``STALE_SECONDS`` refuses a second claimant;
  an older one can be taken over. The training process beats every
  ``HEARTBEAT_SECONDS`` (``python -m sw.longrun run``), so closing the notebook
  tab does not make a live run look dead; a dead VM stops beating and goes
  stale. Drive has no atomic create-if-absent across machines: ``claim`` writes,
  waits ``settle`` seconds, and reads back to see it still holds the lock. That
  catches two people starting at once, not every race; use distinct run tags.
* **Registry.** ``scan`` reads every run folder; ``leaderboard`` prints it.
* **Fork.** ``fork`` starts a new tag from another run's checkpoint at epoch N
  (weights, optionally AdamW moments) for ``sw.train --init-from``.
* **Soup** (experimental). ``soup_runs`` averages the weights of runs that
  share a parent checkpoint and keeps the result only if it beats them.

Every write to the run folder goes through a temporary file and a rename.
Training itself is not data parallel across machines: one run, one VM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

from sw import gates

LOCK_FILE = "LOCK.json"
META_FILE = "metadata.json"
STALE_SECONDS = 600.0
HEARTBEAT_SECONDS = 60.0
ARCH_KEYS = ("model", "tokenizer", "depth_spectral", "depth_spatial", "dim", "heads", "mlp_ratio", "patch", "group")


class LockHeld(RuntimeError):
    """Another live session holds the run folder."""


def _iso(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gates.atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def read_json(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _copy_atomic(src, dst) -> None:
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def data_fingerprint(root) -> dict:
    """Counts and a short hash of the train and val file names, so two runs can tell they saw the same data."""
    root = Path(root)
    stems = {split: gates._stems(root, split) for split in ("train", "val")}
    digest = hashlib.sha256(json.dumps(stems).encode()).hexdigest()[:12]
    return {"train": len(stems["train"]), "val": len(stems["val"]), "sha": digest}


# --- lock and heartbeat ---


def run_status(out_dir, now=None, stale_seconds=STALE_SECONDS) -> dict:
    """``state`` is done, running, stale, paused, failed or new, with the lock and its heartbeat age."""
    out_dir = Path(out_dir)
    now = time.time() if now is None else now
    lock = read_json(out_dir / LOCK_FILE)
    age = None if lock is None else now - float(lock.get("heartbeat", 0.0))
    if (out_dir / gates.DONE_FILE).is_file():
        state = "done"
    elif lock is None:
        state = "stale" if (out_dir / "last.pt").is_file() else "new"
    elif lock.get("status") == "running":
        state = "running" if age <= stale_seconds else "stale"
    else:
        state = {"paused": "paused", "failed": "failed", "done": "done"}.get(lock.get("status"), "stale")
    return {"state": state, "lock": lock, "age": age}


def claim(out_dir, owner, *, session=None, stale_seconds=STALE_SECONDS, force=False, settle=0.0, now=None) -> str:
    """Take the run folder for ``owner`` and return the session id. Raises ``LockHeld`` if a live session has it.

    ``force`` takes over a live lock too (use it for your own session that just lost its VM).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    info = run_status(out_dir, now=now, stale_seconds=stale_seconds)
    lock = info["lock"]
    session = session or uuid.uuid4().hex[:12]
    if info["state"] == "running" and lock.get("session") != session and not force:
        wait = max(0.0, stale_seconds - info["age"])
        raise LockHeld(
            f"{out_dir.name} is being trained by {lock.get('owner')} (heartbeat {info['age']:.0f} s ago, "
            f"since {_iso(lock.get('session_start', 0))}). Pick another RUN_TAG, or wait about {wait:.0f} s "
            "for the heartbeat to go stale if that session is gone. force=True takes over a live lock."
        )
    if info["state"] == "stale" and lock is not None:
        print(f"{out_dir.name}: taking over a stale lock of {lock.get('owner')} (heartbeat {info['age']:.0f} s old)")
    new = {
        "owner": owner,
        "session": session,
        "session_start": now,
        "heartbeat": now,
        "heartbeat_iso": _iso(now),
        "status": "running",
        "host": socket.gethostname(),
    }
    write_json(out_dir / LOCK_FILE, new)
    if settle > 0:
        time.sleep(settle)
        held = read_json(out_dir / LOCK_FILE) or {}
        if held.get("session") != session:
            raise LockHeld(f"{out_dir.name}: {held.get('owner')} claimed it at the same moment; pick another RUN_TAG")
    return session


def beat(out_dir, session, status="running", error=None, now=None) -> bool:
    """Refresh the heartbeat (or set a final ``status``). False when another session holds the lock."""
    out_dir = Path(out_dir)
    lock = read_json(out_dir / LOCK_FILE)
    if lock is None or lock.get("session") != session:
        return False
    now = time.time() if now is None else now
    lock.update(heartbeat=now, heartbeat_iso=_iso(now), status=status)
    if error:
        lock["error"] = str(error)[:2000]
    write_json(out_dir / LOCK_FILE, lock)
    return True


class Heartbeat(threading.Thread):
    """Beats every ``interval`` seconds. If another session took the lock, this process exits at once."""

    def __init__(self, out_dir, session, interval=HEARTBEAT_SECONDS):
        super().__init__(daemon=True)
        self.out_dir, self.session, self.interval = Path(out_dir), session, interval
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                if not beat(self.out_dir, self.session):
                    print(f"{self.out_dir.name}: the lock was taken by another session; exiting", flush=True)
                    os._exit(3)
            except OSError as err:  # a Drive hiccup; the next beat retries
                print(f"heartbeat write failed: {err}", flush=True)


def free_tag(root, tag, stale_seconds=STALE_SECONDS) -> str:
    """``tag``, or ``tag-2``, ``tag-3`` ... the first whose folder has no live lock."""
    candidate, n = tag, 1
    while run_status(Path(root) / candidate, stale_seconds=stale_seconds)["state"] == "running":
        n += 1
        candidate = f"{tag}-{n}"
    return candidate


# --- metadata ---


def write_metadata(out_dir, owner, session, train_args, repo, data, parent=None) -> dict:
    """Create or refresh ``metadata.json``. The flags and the data must match what the folder started with."""
    out_dir = Path(out_dir)
    from sw.train import _build_parser, config_from_args
    from dataclasses import asdict

    flags = gates.run_identity(train_args)
    cfg = asdict(config_from_args(_build_parser().parse_args(train_args)))
    meta = read_json(out_dir / META_FILE) or {}
    if meta:
        if meta.get("flags") not in (None, flags):
            raise ValueError(f"{out_dir} was started with other flags.\n  before: {meta.get('flags')}\n  now:    {flags}")
        if meta.get("data") not in (None, data):
            raise ValueError(
                f"{out_dir} saw other data (fingerprint {meta.get('data')}, now {data}); its band statistics would "
                "not match. Use a new RUN_TAG."
            )
    meta.setdefault("run_tag", out_dir.name)
    meta.setdefault("owner", owner)
    meta.setdefault("created", _iso(time.time()))
    meta.setdefault("flags", flags)
    meta.setdefault("seed", cfg["seed"])
    meta.setdefault("lr", cfg["lr"])
    meta.setdefault("epochs_target", cfg["epochs"])
    meta["data"] = meta.get("data") or data
    meta.setdefault("parent", parent)
    meta["git_sha"] = gates._git_commit(repo)
    meta["epochs_target"] = cfg["epochs"]
    meta.setdefault("sessions", []).append({"owner": owner, "session": session, "start": _iso(time.time())})
    write_json(out_dir / META_FILE, meta)
    return meta


# --- launch (parent) and run (child) ---


def launch(out_dir, owner, train_args, repo, data_root, env=None, force=False, settle=2.0):
    """Claim the folder, write the metadata, and start ``python -m sw.longrun run`` in the background.

    Returns ``(proc, session)``; ``(None, None)`` when the run is already done. Raises ``LockHeld``.
    """
    out_dir = Path(out_dir)
    parent = (read_json(out_dir / META_FILE) or {}).get("parent")
    if run_status(out_dir)["state"] == "done":
        print(f"{out_dir.name}: done; nothing to start")
        return None, None
    info = run_status(out_dir)
    lock = info["lock"] or {}
    if info["state"] == "running" and lock.get("owner") == owner and lock.get("host") == socket.gethostname():
        if gates.run_state(out_dir) == "running":  # Run all again while this VM's process is still training
            print(f"{out_dir.name}: already running here (session {lock.get('session')})")
            return None, lock.get("session")
    session = claim(out_dir, owner, force=force, settle=settle)
    (out_dir / gates.RUNNING_FILE).unlink(missing_ok=True)  # a dead VM's pid means nothing here
    write_metadata(out_dir, owner, session, train_args, repo, data_fingerprint(data_root), parent=parent)
    cmd = [
        sys.executable, "-m", "sw.longrun", "run", "--out-dir", str(out_dir), "--owner", owner,
        "--session", session, "--", *train_args,
    ]
    proc = gates.launch(out_dir, train_args, env=env, cmd=cmd)
    return proc, session


def run(out_dir, owner, session, train_args) -> int:
    """The child: beat while ``gates.run`` trains; leave the lock ``done``, ``paused`` or ``failed``."""
    out_dir = Path(out_dir)
    if not beat(out_dir, session):
        raise LockHeld(f"{out_dir}: this session does not hold the lock")
    heart = Heartbeat(out_dir, session)
    heart.start()
    try:
        gates.run(out_dir, train_args)
    except KeyboardInterrupt:
        heart.stop_event.set()
        beat(out_dir, session, status="paused")
        raise
    except BaseException as err:
        heart.stop_event.set()
        beat(out_dir, session, status="failed", error=f"{type(err).__name__}: {err}")
        raise
    heart.stop_event.set()
    beat(out_dir, session, status="done" if (out_dir / gates.DONE_FILE).is_file() else "paused")
    return 0


# --- registry ---


def _best_row(rows):
    scored = [r for r in rows if r.get("val_fg_miou") is not None]
    return max(scored, key=lambda r: r["val_fg_miou"]) if scored else None


def total_offset(root, tag, _seen=()) -> int:
    """Epochs the run's ancestors had trained, so a fork plots after its parent's epoch."""
    parent = (read_json(Path(root) / tag / META_FILE) or {}).get("parent")
    if not parent:
        return 0
    own = int(parent.get("epoch") or 0)
    if parent["tag"] in _seen or not (Path(root) / parent["tag"]).is_dir():
        return own
    return own + total_offset(root, parent["tag"], (*_seen, tag))


def scan(root, now=None, stale_seconds=STALE_SECONDS) -> list[dict]:
    """One row per run folder under ``root``, best validation score first."""
    rows = []
    for out_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []:
        meta = read_json(out_dir / META_FILE)
        if meta is None and not (out_dir / "metrics.jsonl").is_file() and not (out_dir / "best.pt").is_file():
            continue
        meta = meta or {}
        info = run_status(out_dir, now=now, stale_seconds=stale_seconds)
        try:
            metrics = gates.read_metrics(out_dir / "metrics.jsonl")
        except (OSError, ValueError):
            metrics = []
        best = _best_row(metrics)
        row = {
            "tag": out_dir.name,
            "owner": meta.get("owner") or (info["lock"] or {}).get("owner") or "?",
            "state": info["state"],
            "epoch": int(metrics[-1]["epoch"]) if metrics else 0,
            "epochs_target": meta.get("epochs_target"),
            "best_val": None if best is None else best["val_fg_miou"],
            "best_epoch": None if best is None else best["epoch"],
            "per_class": {} if best is None else (best.get("per_class_iou") or {}),
            "train_loss": metrics[-1]["train_loss"] if metrics else None,
            "lr": meta.get("lr"),
            "seed": meta.get("seed"),
            "sha": (meta.get("git_sha") or "")[:7],
            "data": (meta.get("data") or {}).get("sha"),
            "parent": meta.get("parent"),
            "soup_of": meta.get("soup_of"),
            "offset": total_offset(root, out_dir.name),
            "heartbeat_age": info["age"],
            "error": (info["lock"] or {}).get("error"),
        }
        if best is None and (out_dir / "best.pt").is_file():  # a soup has no epochs, only best.pt
            ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
            row.update(best_val=ckpt.get("val_fg_miou"), best_epoch=ckpt.get("epoch"), per_class=ckpt.get("per_class_iou") or {})
        rows.append(row)
    return sorted(rows, key=lambda r: math.inf if r["best_val"] is None else -r["best_val"])


def leaderboard(rows) -> str:
    """The text table: one run per line, then per-class IoU at each run's best epoch."""
    if not rows:
        return "no runs yet"
    classes = list(next((r["per_class"] for r in rows if r["per_class"]), {}))
    shown = lambda v: "   n/a" if v is None else f"{v:6.3f}"  # noqa: E731
    lines = [f"{'tag':28s} {'owner':10s} {'state':8s} {'epoch':>11s} {'best val':>8s} {'at':>5s}  {'lr':>7s} {'seed':>4s} {'sha':7s} {'data':12s} parent"]
    for r in rows:
        target = "" if not r["epochs_target"] else f"/{r['epochs_target']}"
        parent = r["parent"] and f"{r['parent']['tag']}@{r['parent']['epoch']}" or (f"soup of {len(r['soup_of'])}" if r["soup_of"] else "")
        lr = "" if r["lr"] is None else f"{r['lr']:.0e}"
        lines.append(
            f"{r['tag'][:28]:28s} {str(r['owner'])[:10]:10s} {r['state']:8s} {str(r['epoch']) + target:>11s} "
            f"{shown(r['best_val']):>8s} {str(r['best_epoch'] or ''):>5s}  {lr:>7s} {str(r['seed'] if r['seed'] is not None else ''):>4s} "
            f"{r['sha']:7s} {str(r['data'] or ''):12s} {parent}"
        )
    if len({r["data"] for r in rows if r["data"]}) > 1:
        lines.append("WARNING: runs saw different data fingerprints; their scores are not comparable")
    if classes:
        lines += ["", f"{'per-class val IoU at best':28s} " + " ".join(f"{c[:8]:>8s}" for c in classes)]
        lines += [f"{r['tag'][:28]:28s} " + " ".join(f"{shown(r['per_class'].get(c)):>8s}" for c in classes) for r in rows if r["per_class"]]
    return "\n".join(lines)


# --- fork and soup ---


def _arch(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in ARCH_KEYS}


def checkpoint_path(run_dir, epoch=None) -> Path:
    """``last.pt`` (epoch None) or the ``ckpt_eNNNNN.pt`` snapshot of that epoch."""
    run_dir = Path(run_dir)
    path = run_dir / ("last.pt" if epoch is None else f"ckpt_e{int(epoch):05d}.pt")
    if not path.is_file():
        have = sorted(p.name for p in run_dir.glob("ckpt_e*.pt"))
        raise FileNotFoundError(f"{path} does not exist; this run has {have or 'no snapshots'} (train with --keep-last)")
    return path


def fork(parent_dir, child_dir, train_args, epoch=None, optimizer=False) -> dict:
    """Prepare ``child_dir`` to start from ``parent_dir``'s checkpoint. Returns ``{"parent", "args"}``.

    Copies the checkpoint to ``child_dir/init.pt`` (weights only unless ``optimizer``), plus the parent's
    ``band_stats.pt``. Pass ``parent`` to ``launch`` through the child's ``metadata.json`` (written here)
    and ``args`` (``--init-from ...``) on to ``sw.train``. Safe to call again for the same parent and epoch.
    """
    from dataclasses import asdict
    from sw.train import _build_parser, config_from_args

    parent_dir, child_dir = Path(parent_dir), Path(child_dir)
    meta = read_json(child_dir / META_FILE) or {}
    done = meta.get("parent")
    if done and done["tag"] == parent_dir.name and (child_dir / "init.pt").is_file() and epoch in (None, done["epoch"]):
        if done["optimizer"] != bool(optimizer):
            raise ValueError(f"{child_dir.name} was forked {'with' if done['optimizer'] else 'without'} the optimizer state; use a new RUN_TAG")
        flags = ["--init-from", str(child_dir / "init.pt"), *(["--init-optimizer"] if done["optimizer"] else [])]
        return {"parent": done, "args": flags}  # a re-run: keep the fork it started from, even if the parent trained on
    src = checkpoint_path(parent_dir, epoch)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = asdict(config_from_args(_build_parser().parse_args(train_args)))
    if _arch(ckpt.get("cfg", {})) != _arch(cfg):
        raise ValueError(f"{src} has another architecture:\n  parent: {_arch(ckpt.get('cfg', {}))}\n  child:  {_arch(cfg)}")
    if optimizer and "optimizer" not in ckpt:
        raise ValueError(f"{src} has no optimizer state")
    info = {"tag": parent_dir.name, "epoch": int(ckpt["epoch"]), "optimizer": bool(optimizer)}
    child_dir.mkdir(parents=True, exist_ok=True)
    if meta and meta.get("parent") != info:
        raise ValueError(f"{child_dir.name} already exists with parent {meta.get('parent')}, not {info}; use a new RUN_TAG")
    init = child_dir / "init.pt"
    if not init.is_file():
        if optimizer:
            _copy_atomic(src, init)
        else:
            tmp = init.with_name(init.name + ".tmp")
            torch.save({"model": ckpt["model"], "epoch": ckpt["epoch"], "cfg": ckpt.get("cfg")}, tmp)
            os.replace(tmp, init)
    if (parent_dir / "band_stats.pt").is_file() and not (child_dir / "band_stats.pt").is_file():
        _copy_atomic(parent_dir / "band_stats.pt", child_dir / "band_stats.pt")
    write_json(child_dir / META_FILE, {**meta, "run_tag": child_dir.name, "parent": info, "data": (read_json(parent_dir / META_FILE) or {}).get("data")})
    return {"parent": info, "args": ["--init-from", str(init), *(["--init-optimizer"] if optimizer else [])]}


def soup_state(paths) -> dict:
    """Uniform average of the ``model`` weights in ``paths``. Integer buffers come from the first file."""
    states = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in paths]
    if not states:
        raise ValueError("soup needs at least one checkpoint")
    if any(set(s) != set(states[0]) for s in states):
        raise ValueError("the checkpoints do not have the same parameters")
    out = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            out[key] = sum(s[key].double() for s in states).div(len(states)).to(first.dtype)
        else:
            out[key] = first.clone()
    return out


def make_evaluator(run_dir, data_root, device="cpu"):
    """``evaluate(state_dict) -> (val_fg_miou, per_class_iou)`` for runs trained like ``run_dir`` (same flags).

    Builds the model and the val loader once, with the band statistics saved in ``run_dir``.
    """
    from dataclasses import replace

    from sw import train as T

    flags = (read_json(Path(run_dir) / META_FILE) or {})["flags"]
    args = T._build_parser().parse_args(flags)
    cfg = replace(T.config_from_args(args), out_dir=str(run_dir), data_root=str(data_root), device=device, num_workers=0)
    dev = T.resolve_device(device)
    model = T.build_model(cfg, in_channels=args.in_channels).to(dev)
    loader = T._loader(T._split(cfg, "val", cfg.max_val_images, False, T._resolve_standardizer(cfg)), cfg, False, False, False)

    def evaluate(state):
        model.load_state_dict(state)
        return T._evaluate(model, loader, dev)

    return evaluate


def soup_runs(run_dirs, evaluate, dest_dir, owner, ckpt_name="best.pt") -> dict:
    """EXPERIMENTAL. Average the ``ckpt_name`` weights of runs that fork from one parent checkpoint.

    ``evaluate(state_dict) -> (val_fg_miou, per_class_iou)``. The soup is saved as a new run in ``dest_dir``
    (``best.pt``, ``metadata.json``, ``DONE.json``) only if its score is higher than every ingredient's.
    """
    run_dirs = [Path(p) for p in run_dirs]
    parents = {json.dumps((read_json(d / META_FILE) or {}).get("parent"), sort_keys=True) for d in run_dirs}
    if len(run_dirs) < 2 or len(parents) != 1 or parents == {"null"}:
        raise ValueError("a soup needs 2 or more runs that fork from the same parent checkpoint (see metadata.json 'parent')")
    paths = [d / ckpt_name for d in run_dirs]
    scores = {d.name: evaluate(torch.load(p, map_location="cpu", weights_only=False)["model"]) for d, p in zip(run_dirs, paths)}
    state = soup_state(paths)
    score, per_class = evaluate(state)
    top = max(s for s, _ in scores.values())
    result = {"ingredients": {k: v[0] for k, v in scores.items()}, "soup_val_fg_miou": score, "saved": False}
    if not (score is not None and not math.isnan(score) and score > top):
        return result
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    payload = {"model": state, "epoch": first.get("epoch"), "val_fg_miou": score, "per_class_iou": per_class, "cfg": first.get("cfg")}
    tmp = dest_dir / "best.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, dest_dir / "best.pt")
    meta = {
        "run_tag": dest_dir.name, "owner": owner, "created": _iso(time.time()), "soup_of": [d.name for d in run_dirs],
        "parent": (read_json(run_dirs[0] / META_FILE) or {}).get("parent"), "data": (read_json(run_dirs[0] / META_FILE) or {}).get("data"),
        "git_sha": gates._git_commit(Path(__file__).resolve().parent), "experimental": True,
    }
    write_json(dest_dir / META_FILE, meta)
    write_json(dest_dir / gates.DONE_FILE, {"result": result, "soup": True})
    result["saved"] = True
    return result


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sw.longrun")
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="train into --out-dir under a claimed lock (started by launch)")
    runner.add_argument("--out-dir", required=True)
    runner.add_argument("--owner", required=True)
    runner.add_argument("--session", required=True)
    runner.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    return run(args.out_dir, args.owner, args.session, train_args)


if __name__ == "__main__":
    raise SystemExit(_cli())
