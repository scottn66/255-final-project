"""notebooks/09_long_run.ipynb: valid, no stored outputs, its flags build the models, and its cells run end to end
on a tiny synthetic root (sessions, a time-budget pause, a resume, forks, a soup, a dead run, a stuck run)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sw import gates, longrun
from sw.train import _build_parser, build_model, config_from_args
from test_train import _synthetic_root

ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT_DIR / "notebooks" / "09_long_run.ipynb"
PARAMS = {"fused": 488_487, "collapsed": 491_191, "cnn": 571_879}
ENV_NAMES = (
    "LONG_RUNNER LONG_MODEL LONG_TAG LONG_EPOCHS LONG_LR LONG_SEED LONG_DOWNSAMPLE LONG_BATCH LONG_WARMUP_EPOCHS "
    "LONG_SAVE_EVERY LONG_KEEP_LAST LONG_EVAL_EVERY LONG_BUDGET LONG_FORK_FROM LONG_FORK_EPOCH LONG_SOUP LONG_BRANCH "
    "LONG_ROOT LONG_WORK LONG_DATA LONG_ZIP LONG_DEVICE LONG_WORKERS LONG_POLL LONG_STALL_FLOOR LONG_ALLOW_PARTIAL"
).split()


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells():
    """Code cells by id."""
    return {c["id"]: "".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"}


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _settings(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    names: dict = {}
    exec(compile(_cells()["long-02"], "settings cell", "exec"), names)
    return names


def test_notebook_is_valid_and_has_no_outputs():
    nb = _notebook()
    assert nb["nbformat"] == 4 and nb["cells"]
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code") and isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    assert "widgets" not in nb["metadata"]
    badge = "".join(nb["cells"][0]["source"])
    assert "/blob/cursor/long-run-notebook-067c/notebooks/09_long_run.ipynb" in badge


def test_default_settings(clean_env):
    s = _settings(clean_env, LONG_RUNNER="ann")
    assert (s["EPOCHS_TARGET"], s["MODEL"], s["RUN_TAG"]) == (1000, "fused", "ann-fused")
    assert s["ROOT"] == Path("/content/drive/MyDrive/SW_runs/long_run") and s["DATA_DIR"] == Path("/content/data")
    assert s["AUTO_NEW_TAG"] is False and s["FORCE_TAKEOVER"] is False and s["SOUP_RUNS"] == []
    assert s["SAVE_EVERY"] == 5 and s["KEEP_LAST"] == 3 and s["TIME_BUDGET_MINUTES"] == 600


@pytest.mark.parametrize("model", sorted(PARAMS))
def test_flags_parse_and_build_the_expected_model(clean_env, model):
    s = _settings(clean_env, LONG_RUNNER="ann", LONG_MODEL=model)
    flags = s["TRAIN_ARGS"]
    assert flags.count("--epochs") == 1 and flags.count("--lr") == 1
    for flag in ("--stats-path", "--patience", "--out-dir", "--resume", "--data-root", "--init-from", "--no-cosine"):
        assert flag not in flags  # no early stop, no constant lr; the rest is added at launch
    args = _build_parser().parse_args(flags)
    cfg = config_from_args(args)
    assert (cfg.epochs, cfg.downsample, cfg.batch_size, cfg.standardize, cfg.cosine) == (1000, 2, 4, True, True)
    assert cfg.lr == pytest.approx(1e-3 if model == "cnn" else 3e-4)  # passed explicitly
    assert (cfg.warmup_steps, cfg.clip_grad_norm, cfg.cosine_floor) == (5 * (514 // 4), 1.0, 0.01)
    assert (cfg.save_every, cfg.keep_last, cfg.eval_every, cfg.max_minutes) == (5, 3, 5, 600.0)
    n_params = sum(p.numel() for p in build_model(cfg, in_channels=args.in_channels).parameters() if p.requires_grad)
    assert n_params == PARAMS[model]
    assert s["PARAM_LINE"][model] == f"{'MiniNetSeg' if model == 'cnn' else 'BandTransformer'} parameters: {n_params:,} ({n_params / 1e6:.3f}M)"


def test_optional_flags_and_the_transformer_default_lr_is_unchanged():
    assert config_from_args(_build_parser().parse_args(["--model", "transformer"])).lr == pytest.approx(1e-3)


def test_optional_recipe_flags_are_passed_only_when_set(clean_env):
    s = _settings(clean_env, LONG_RUNNER="ann", LONG_LR="1e-4")
    assert s["TRAIN_ARGS"][s["TRAIN_ARGS"].index("--lr") + 1] == "0.0001"
    for flag in ("--pe-init-std", "--no-aug", "--amp"):
        assert flag not in s["TRAIN_ARGS"]


@pytest.mark.parametrize(
    "env, message",
    [
        ({}, "RUNNER_NAME"),
        ({"LONG_RUNNER": "a b"}, "RUNNER_NAME"),
        ({"LONG_RUNNER": "ann", "LONG_MODEL": "vit"}, "MODEL must be"),
        ({"LONG_RUNNER": "ann", "LONG_TAG": "../x"}, "valid run tag"),
        ({"LONG_RUNNER": "ann", "LONG_TAG": "p", "LONG_FORK_FROM": "p"}, "own RUN_TAG"),
        ({"LONG_RUNNER": "ann", "LONG_EPOCHS": "0"}, "EPOCHS_TARGET"),
    ],
)
def test_settings_refuse_bad_names_and_values(clean_env, env, message):
    with pytest.raises(ValueError, match=message):
        _settings(clean_env, **env)


def test_the_cells_carry_their_guards():
    cells = _cells()
    assert "_build_parser().parse_args(TRAIN_ARGS)" in cells["long-04"]
    assert "EPOCHS_TARGET" in cells["long-10"] and "fork this run" in cells["long-10"]
    assert "AUTO_NEW_TAG" in cells["long-10"] and "longrun.STALE_SECONDS" in cells["long-10"]
    monitor = cells["long-12"]
    assert "is not running and not done" in monitor and "quiet > limit" in monitor and 'state == "failed"' in monitor
    assert "EXPERIMENTAL" in cells["long-02"] and "EXPERIMENTAL" in "".join(_notebook()["cells"][-3]["source"]).upper()
    assert "--epochs" not in "\n".join(v for k, v in cells.items() if k != "long-02")


# ---- the cells, run for real on a tiny root ----

pytestmark_plots = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@pytest.fixture
def lab(clean_env, tmp_path, monkeypatch):
    _synthetic_root(tmp_path / "data")
    env = {
        "LONG_RUNNER": "ann", "LONG_ROOT": str(tmp_path / "root"), "LONG_WORK": str(tmp_path / "work"),
        "LONG_DEVICE": "cpu", "LONG_WORKERS": "0", "LONG_EPOCHS": "4", "LONG_POLL": "1", "LONG_ALLOW_PARTIAL": "1",
        "SW_ROOT": str(tmp_path / "data"), "LONG_DOWNSAMPLE": "4", "LONG_BATCH": "2", "LONG_SAVE_EVERY": "1",
        "LONG_EVAL_EVERY": "1", "LONG_WARMUP_EPOCHS": "0", "REPO_ROOT": str(ROOT_DIR), "MPLBACKEND": "Agg",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)

    def run_cells(stop_before=None, only=None, ns=None, **override):
        for name, value in override.items():
            monkeypatch.setenv(name, value)
        ns = {} if ns is None else ns
        for cid, source in _cells().items():
            if only and cid not in only:
                continue
            if stop_before and cid == stop_before:
                break
            exec(compile(source, cid, "exec"), ns)
        return ns

    run_cells.root = tmp_path / "root"
    return run_cells


@pytestmark_plots
def test_notebook_runs_pauses_on_the_time_budget_and_resumes(lab):
    paused = lab(LONG_BUDGET="1e-9")  # stops after one saved epoch; the monitor reports a pause, not a failure
    run = lab.root / "ann-fused"
    assert paused["state"] == "paused" and not (run / gates.DONE_FILE).exists()
    assert longrun.run_status(run)["state"] == "paused" and len(gates.read_metrics(run / "metrics.jsonl")) == 1
    done = lab(LONG_BUDGET="600")  # Run all again, the next session
    assert longrun.run_status(run)["state"] == "done"
    assert [r["epoch"] for r in gates.read_metrics(run / "metrics.jsonl")] == [1, 2, 3, 4]
    assert [r["tag"] for r in done["ROWS"]] == ["ann-fused"] and done["ROWS"][0]["epoch"] == 4
    meta = longrun.read_json(run / longrun.META_FILE)
    assert meta["owner"] == "ann" and len(meta["sessions"]) == 2 and meta["epochs_target"] == 4 and meta["parent"] is None
    with pytest.raises(RuntimeError, match="EPOCHS_TARGET 4, not 8"):  # the cosine schedule cannot be stretched
        lab(LONG_EPOCHS="8", only=("long-02", "long-04", "long-06", "long-10"))


@pytestmark_plots
def test_notebook_forks_two_variants_and_soups_them(lab):
    lab()
    root = lab.root
    for tag, seed, lr in (("bob-a", "1", "1e-3"), ("tim-b", "2", "3e-3")):
        ns = lab(LONG_RUNNER=tag[:3], LONG_TAG=tag, LONG_FORK_FROM="ann-fused", LONG_SEED=seed, LONG_LR=lr, LONG_EPOCHS="2")
        meta = longrun.read_json(root / tag / longrun.META_FILE)
        assert meta["parent"] == {"tag": "ann-fused", "epoch": 4, "optimizer": False} and meta["seed"] == int(seed)
        assert float(meta["lr"]) == float(lr) and meta["owner"] == tag[:3]
    rows = {r["tag"]: r for r in ns["ROWS"]}
    assert set(rows) == {"ann-fused", "bob-a", "tim-b"} and rows["bob-a"]["offset"] == 4 and rows["tim-b"]["state"] == "done"
    soup = lab(only=("long-02", "long-04", "long-06", "long-19"), LONG_RUNNER="ann", LONG_TAG="ann-soup", LONG_SOUP="bob-a,tim-b", LONG_EPOCHS="2")
    assert set(soup["SOUP"]["ingredients"]) == {"bob-a", "tim-b"}
    assert (root / "ann-soup" / "best.pt").is_file() == soup["SOUP"]["saved"]


def test_a_live_heartbeat_of_someone_else_stops_the_launch_cell(lab):
    ns = lab(stop_before="long-10")
    longrun.claim(lab.root / "ann-fused", "bob")
    with pytest.raises(longrun.LockHeld, match="bob"):
        lab(only=("long-10",), ns=ns)
    ns["AUTO_NEW_TAG"] = True
    lab(only=("long-10",), ns=ns)
    assert ns["RUN_TAG"] == "ann-fused-2" and (lab.root / "ann-fused-2" / longrun.META_FILE).is_file()


def test_the_monitor_raises_on_a_dead_run_instead_of_waiting(lab, monkeypatch):
    ns = lab(stop_before="long-10")
    run = lab.root / "ann-fused"
    longrun.claim(run, "ann")  # a lock that looks alive ...
    ns["PROC"] = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])  # ... with no process behind it
    ns["PROC"].wait()
    ns["RUN_DIR"] = run
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="not running and not done"):
        lab(only=("long-12",), ns=ns)


def test_the_monitor_raises_on_a_stuck_run(lab):
    ns = lab(stop_before="long-10")
    run = lab.root / "ann-fused"
    longrun.claim(run, "ann")
    ns["PROC"] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    ns["RUN_DIR"], ns["STALL_FLOOR_SECONDS"] = run, -1
    try:
        with pytest.raises(RuntimeError, match="is stuck"):
            lab(only=("long-12",), ns=ns)
    finally:
        ns["PROC"].kill()


def test_the_monitor_names_a_run_that_failed(lab):
    ns = lab(stop_before="long-10")
    run = lab.root / "ann-fused"
    session = longrun.claim(run, "ann")
    longrun.beat(run, session, status="failed", error="CUDA out of memory")
    ns["RUN_DIR"] = run
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        lab(only=("long-12",), ns=ns)
