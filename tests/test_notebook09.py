"""notebooks/09_fullres_fused_vs_control.ipynb: valid, and its flags build the models.

The second half runs the notebook's own cells against folders made by hand:
``gates.launch`` is replaced by a recorder, the clock is a fake, and nothing
is trained.
"""

from __future__ import annotations

import ast
import json
import os
import re
import types
import weakref
from pathlib import Path

import pytest

from sw import gates
from sw.data import CLASS_NAMES
from sw.train import _build_parser, build_model, config_from_args

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "09_fullres_fused_vs_control.ipynb"
PARAMS = {"tf_fused_full": 488_487, "tf_collapsed_full": 491_191, "cnn_full": 571_879}
FULL_ENV = (
    "SW_BRANCH", "FULL_RUNS_DIR", "FULL_WORK", "FULL_DATA", "FULL_ZIP", "FULL_DEVICE",
    "FULL_WORKERS", "FULL_POLL", "FULL_EPOCHS", "FULL_TAGS", "FULL_ONE_AT_A_TIME",
    "FULL_STALL_FLOOR", "FULL_EXTEND",
)


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells():
    return ["".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _all_text():
    return "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])


def _settings(monkeypatch, **env):
    """The names the Settings cell defines, with only the given environment overrides."""
    for name in FULL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    names: dict = {}
    exec(compile(_code_cells()[0], "settings cell", "exec"), names)
    return names


@pytest.fixture
def settings(monkeypatch):
    return _settings(monkeypatch)


def test_notebook_is_valid():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"]
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert isinstance(cell["outputs"], list)  # committed with the outputs of its Colab run
            ast.parse("".join(cell["source"]))
    header = "".join(nb["cells"][0]["source"])
    assert "/blob/wp7-screen-notebook/notebooks/09_fullres_fused_vs_control.ipynb" in header
    assert header.rstrip().endswith("AI-assisted notebook; Scott reviews every line.")


def test_default_settings(settings):
    assert settings["BRANCH"] == "wp6-collapsed-tokenizer"
    assert settings["RUNS"] == Path("/content/drive/MyDrive/sw_runs/fullres")
    assert settings["FULL_EPOCHS"] == 60
    assert settings["EXTEND"] is True
    # Fused, then the control. The CNN is defined but not run by default.
    assert settings["RUN_TAGS"] == ["tf_fused_full", "tf_collapsed_full"]
    assert (settings["FUSED"], settings["CONTROL"]) == ("tf_fused_full", "tf_collapsed_full")
    assert settings["STALL_FLOOR_SECONDS"] == 3600
    assert settings["ONE_AT_A_TIME"] is True
    assert set(settings["FULL"]) == set(PARAMS)


def test_environment_overrides(monkeypatch):
    names = _settings(
        monkeypatch, FULL_EPOCHS="90", FULL_EXTEND="0", FULL_TAGS="cnn_full, tf_fused_full", FULL_RUNS_DIR="/tmp/x"
    )
    assert names["FULL_EPOCHS"] == 90
    assert names["EXTEND"] is False
    assert names["RUN_TAGS"] == ["cnn_full", "tf_fused_full"]
    assert names["RUNS"] == Path("/tmp/x")
    for flags in names["FULL"].values():
        assert flags[flags.index("--epochs") + 1] == "90"
    with pytest.raises(ValueError, match="RUN_TAGS"):
        _settings(monkeypatch, FULL_TAGS="tf_fused_std")


@pytest.mark.parametrize("tag", sorted(PARAMS))
def test_flags_parse_and_build_the_expected_model(settings, tag):
    flags = settings["FULL"][tag]
    assert flags.count("--epochs") == 1
    for flag in (
        "--downsample", "--stats-path", "--patience", "--amp", "--max-train-images", "--max-val-images",
        "--max-train-steps", "--crop-size", "--out-dir", "--resume",
    ):
        assert flag not in flags
    args = _build_parser().parse_args(flags)
    cfg = config_from_args(args)
    # No --downsample flag is native size: sw.train sets downsample 1 and prints it.
    assert (cfg.epochs, cfg.batch_size, cfg.downsample, cfg.crop_size) == (60, 8, 1, None)
    assert cfg.standardize is True
    assert cfg.cosine is False
    assert cfg.stats_path is None
    assert cfg.patience is None and cfg.amp is False
    assert cfg.max_train_images is None and cfg.max_val_images is None
    assert cfg.lr == pytest.approx(1e-3 if tag == "cnn_full" else 3e-4)
    if tag == "tf_collapsed_full":
        assert (cfg.tokenizer, cfg.depth_spectral, cfg.depth_spatial) == ("collapsed", 0, 8)
    if tag == "tf_fused_full":
        assert (cfg.tokenizer, cfg.depth_spectral, cfg.depth_spatial) == ("band", 2, 6)
    model = build_model(cfg, in_channels=args.in_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == PARAMS[tag]
    # The line the smoke cell and the monitor look for is the one sw.train prints.
    line = f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)"
    assert settings["PARAM_LINE"][tag] == line
    assert f"downsample {cfg.downsample}  " == settings["RESOLUTION_TEXT"]


def test_epochs_flag_is_set_once_and_downsample_only_in_the_smoke():
    setup, data, smoke, launch, monitor, table = _code_cells()[1:]
    later = "\n".join((setup, data, smoke, launch, monitor, table))
    assert '"--epochs"' not in later
    assert "--stats-path" not in later
    # --smoke alone would shrink the images by 4; the smoke asks for native size.
    assert '"--smoke", "--downsample", "1"' in smoke
    for source in (setup, data, launch, monitor, table):
        assert "--downsample" not in source
    # gates.run_state never returns "failed"; a dead run reads "new" or "partial".
    assert '"failed"' not in later
    assert '("new", "partial")' in monitor


def test_guards_are_in_the_cells_that_need_them():
    setup, data, smoke, launch, monitor, table = _code_cells()[1:]
    # A branch whose sw.train lacks a flag stops in Setup, before the data step.
    assert "flags_error(flags)" in setup
    # Every train.log must show the native-size banner.
    assert "RESOLUTION_TEXT" in data and "IMAGES_LINE" in monitor
    assert "RESOLUTION_TEXT" in smoke
    # The memory print runs on cuda only.
    assert 'if DEVICE == "cuda":' in smoke and "max_memory_allocated" in smoke
    # A short finished tag is reopened or refused before anything is launched.
    assert launch.index("check_flags()\n") < launch.index("extend_or_refuse()\n") < launch.rindex("start_due()")
    assert "< FULL_EPOCHS" in launch and "if not EXTEND:" in launch
    assert '"last.pt"' in launch  # no checkpoint: refuse, do not restart from epoch 1
    # A live run whose log stopped growing raises; a dead one does not block the next tag.
    assert "quiet > limit" in monitor and "is stuck" in monitor
    assert "if tag in DEAD:\n            continue" in launch
    # The table warns when the two runs differ in length, and states the limits of the run.
    assert "WARNING: the two runs have different epoch counts" in table
    assert "This run does not settle the hypothesis" in table
    assert '"competitive"' in table


def test_wording():
    text = _all_text().lower()
    for word in ("beat", "outperform", "better than"):
        assert word not in text
    assert "cnn_full` is the optional third reference" in _all_text()
    assert "estimate" in "".join(_notebook()["cells"][0]["source"])


# ---------------------------------------------------------------------------
# Behaviour: the Setup, Smoke, Launch, Monitor and Table cells, run on folders made by hand.

SETUP, DATA, SMOKE, LAUNCH, MONITOR, TABLE = range(1, 7)
NOT_ALLOWED = (
    "--downsample", "--stats-path", "--patience", "--amp", "--max-train-images", "--max-val-images",
    "--max-train-steps", "--crop-size", "--resume",
)
TOKENIZER_TAIL = "  dim 64  heads 4  mlp_ratio 4.0  patch 8"


class Clock:
    """Stands in for the time module: sleep() moves the clock and does not wait."""

    def __init__(self):
        self.now, self.sleeps = 1_000_000.0, 0

    def time(self):
        return self.now

    def strftime(self, fmt):
        return "00:00:00"

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds
        if self.sleeps > 100:
            raise AssertionError("the monitor did not stop")


class World:
    """The notebook's namespace after Settings, with what Setup and Data would add."""

    def __init__(self, monkeypatch, tmp_path, **env):
        env = {"FULL_RUNS_DIR": str(tmp_path / "runs"), "FULL_DEVICE": "cpu", **env}
        self.ns = _settings(monkeypatch, **env)
        self.clock = Clock()
        self.launched: list[tuple[str, list[str]]] = []
        self.hold = False  # True: a launched tag then reads as running (this test process holds it)

        def launch(out_dir, train_args, env=None, **kw):
            self.launched.append((Path(out_dir).name, list(train_args)))
            if self.hold:
                (Path(out_dir) / gates.RUNNING_FILE).write_text(json.dumps(gates._holder_record(os.getpid())))

        monkeypatch.setattr(gates, "launch", launch)
        self.ns.update(
            json=json, re=re, time=self.clock, gates=gates, SW_ROOT=tmp_path / "data", ENV={}, COMMIT="test",
            T_START=self.clock.now, tail=lambda path, n=25: "(log tail)",
            IMAGES_LINE=f"train images 514  val images 167  {self.ns['RESOLUTION_TEXT']}",
        )
        self.runs = self.ns["RUNS"]

    def run(self, index):
        exec(compile(_code_cells()[index], f"cell {index}", "exec"), self.ns)

    def flags(self, tag):
        return [*self.ns["FULL"][tag], "--data-root", str(self.ns["SW_ROOT"])]

    def folder(self, tag, values, done=True, last=True, args=None, banner=None, running=False):
        """A run folder with one metrics row per value of val FG mIoU."""
        out = self.runs / tag
        out.mkdir(parents=True)
        rows = [
            {"epoch": i, "val_fg_miou": v, "per_class_iou": dict.fromkeys(CLASS_NAMES, v), "epoch_seconds": 60.0}
            for i, v in enumerate(values, start=1)
        ]
        (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / gates.ARGS_FILE).write_text(json.dumps(self.flags(tag) if args is None else args))
        if banner is None:
            banner = self.banner(tag)
        (out / "train.log").write_text("\n".join(banner) + "\n")
        if last:
            (out / "last.pt").write_bytes(b"not a real checkpoint")
        if done:
            (out / gates.DONE_FILE).write_text(json.dumps({"result": {"param_count": PARAMS[tag]}}))
        if running:  # held by this test process, which is alive
            (out / gates.RUNNING_FILE).write_text(json.dumps(gates._holder_record(os.getpid())))
        return out

    def banner(self, tag, images=None, params=None):
        lines = [params or self.ns["PARAM_LINE"][tag]]
        if self.ns["TOKENIZER_LINE"][tag]:
            lines.append(self.ns["TOKENIZER_LINE"][tag] + TOKENIZER_TAIL)
        lines.append(images or (self.ns["IMAGES_LINE"] + "crop None  augment on  out x"))
        return lines


@pytest.fixture
def world(monkeypatch, tmp_path):
    return lambda **env: World(monkeypatch, tmp_path, **env)


def _listing(root):
    return sorted((str(p.relative_to(root)), p.read_bytes()) for p in root.rglob("*") if p.is_file())


def test_no_cell_after_settings_adds_a_training_flag():
    for source in _code_cells()[SETUP:]:
        for flag in ("--patience", "--amp", "--max-train", "--max-val", "--stats-path", "--crop-size", "--batch-size"):
            assert flag not in source


def test_setup_stops_on_a_branch_that_lacks_a_flag(monkeypatch, tmp_path):
    repo = NOTEBOOK.parents[1]
    monkeypatch.chdir(repo)
    env = {"REPO_ROOT": str(repo), "FULL_DEVICE": "cpu", "FULL_RUNS_DIR": str(tmp_path / "r"), "FULL_WORK": str(tmp_path / "w")}
    names = _settings(monkeypatch, **env)
    exec(compile(_code_cells()[SETUP], "setup cell", "exec"), names)
    assert names["flags_error"](["--model", "cnn"]) is None
    assert "invalid choice" in names["flags_error"](["--tokenizer", "nope"])
    # The check covers every tag, the first one included, and stops before the runs folder is made.
    for tag in ("tf_fused_full", "cnn_full"):
        names = _settings(monkeypatch, **{**env, "FULL_RUNS_DIR": str(tmp_path / f"never_{tag}")})
        names["FULL"][tag] = [*names["FULL"][tag], "--no-such-flag"]
        with pytest.raises(RuntimeError, match=f"does not take the flags of {tag}: .*--no-such-flag"):
            exec(compile(_code_cells()[SETUP], "setup cell", "exec"), names)
        assert not (tmp_path / f"never_{tag}").exists()


def _smoke_world(world, stdout_for, device="cpu"):
    w = world()
    commands = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        tag = Path(cmd[cmd.index("--out-dir") + 1]).name.removeprefix("smoke_")
        return types.SimpleNamespace(returncode=0, stdout=stdout_for(w, tag), stderr="")

    w.ns.update(subprocess=types.SimpleNamespace(run=run), sys=types.SimpleNamespace(executable="python"),
                WORK=w.runs.parent / "work", REPO=NOTEBOOK.parents[1], DEVICE=device)
    return w, commands


def _good_smoke(w, tag):
    return "\n".join(w.banner(tag, images="train images 4  val images 4  downsample 1  crop None  augment on")) + "\n"


def test_smoke_runs_every_model_at_native_size_and_reads_the_banner(world, capsys):
    w, commands = _smoke_world(world, _good_smoke)
    w.run(SMOKE)
    assert len(commands) == 3
    for cmd, tag in zip(commands, w.ns["ARMS"]):
        assert cmd[:3] == ["python", "-m", "sw.train"]
        assert cmd[3:] == [
            *w.ns["ARMS"][tag], "--smoke", "--downsample", "1", "--standardize", "--device", "cpu",
            "--data-root", str(w.ns["SW_ROOT"]), "--out-dir", str(w.ns["WORK"] / f"smoke_{tag}"),
        ]
    assert "peak GPU memory: skipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    "old, new",
    [
        ("downsample 1  crop None", "downsample 4  crop None"),
        ("downsample 1  crop None", "downsample 1  crop 128"),
        ("488,487 (0.488M)", "488,488 (0.488M)"),
        ("tokenizer collapsed  blocks 0+8", "tokenizer band  blocks 2+6"),
    ],
)
def test_smoke_stops_on_a_wrong_banner(world, old, new):
    w, _ = _smoke_world(world, lambda w, tag: _good_smoke(w, tag).replace(old, new))
    with pytest.raises(RuntimeError, match="smoke .* failed: the output has no line"):
        w.run(SMOKE)


def test_memory_check_frees_a_failed_model_before_it_empties_the_cache(world, capsys):
    """After an out-of-memory error the model must be gone when empty_cache() runs."""
    import gc

    w, _ = _smoke_world(world, _good_smoke)
    w.run(SMOKE)

    class Big:
        pass

    made, alive_at_empty = [], []

    def measure(tag):
        big = Big()  # stands in for the model and the batch; the traceback keeps it alive
        made.append(weakref.ref(big))
        if tag == "tf_collapsed_full":
            raise RuntimeError("CUDA out of memory")
        return 1.5

    cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: alive_at_empty.append(made[-1]() is not None)
    )
    w.ns.update(torch=types.SimpleNamespace(cuda=cuda), gc=gc)
    w.ns["measure_memory"](["tf_fused_full", "tf_collapsed_full"], measure=measure)
    assert alive_at_empty == [False, False]  # emptied once per tag, each time with nothing held
    out = capsys.readouterr().out
    assert "tf_fused_full      1.50 GiB" in out
    assert "tf_collapsed_full  could not be measured: RuntimeError: CUDA out of memory" in out
    # Only the tags that will run are measured.
    assert "measure_memory(RUN_TAGS)" in _code_cells()[SMOKE]


def test_launch_passes_exactly_the_tag_flags_and_the_data_root(world):
    w = world()
    w.run(LAUNCH)
    assert w.launched == [("tf_fused_full", [*w.ns["FULL"]["tf_fused_full"], "--data-root", str(w.ns["SW_ROOT"])])]
    for flag in NOT_ALLOWED:
        assert flag not in w.launched[0][1]
    # The control starts only after the fused run is done, and a dead tag does not block it.
    w.ns["start_due"]()
    assert len(w.launched) == 1
    w.ns["DEAD"].add("tf_fused_full")
    w.ns["start_due"]()
    assert w.launched[1] == ("tf_collapsed_full", w.flags("tf_collapsed_full"))


def test_launch_all_at_once(world):
    w = world(FULL_ONE_AT_A_TIME="0", FULL_TAGS="tf_fused_full,tf_collapsed_full,cnn_full")
    w.run(LAUNCH)
    assert w.launched == [(tag, w.flags(tag)) for tag in w.ns["RUN_TAGS"]]


def test_extend_reopens_a_short_finished_tag(world, capsys):
    w = world(FULL_EPOCHS="5")
    out = w.folder("tf_fused_full", [0.1, 0.2, 0.3])
    record = (out / gates.DONE_FILE).read_text()
    w.run(LAUNCH)
    assert not (out / gates.DONE_FILE).exists()
    assert (out / "DONE.after_3_epochs.json").read_text() == record
    assert "EXTEND tf_fused_full: it finished at 3 epochs and FULL_EPOCHS is 5" in capsys.readouterr().out
    assert w.launched == [("tf_fused_full", w.flags("tf_fused_full"))]
    assert w.launched[0][1][w.launched[0][1].index("--epochs") + 1] == "5"


def test_extend_off_refuses_and_changes_nothing(world):
    w = world(FULL_EPOCHS="5", FULL_EXTEND="0")
    w.folder("tf_fused_full", [0.1, 0.2, 0.3])
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match=r"finished with fewer than 5 epochs: \{'tf_fused_full': 3\}"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


def test_extend_without_a_checkpoint_refuses_and_changes_nothing(world):
    w = world(FULL_EPOCHS="5")
    w.folder("tf_fused_full", [0.1, 0.2, 0.3])
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3], last=False)
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match=r"\['tf_collapsed_full'\].*no last.pt"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


@pytest.mark.parametrize("epochs", ["3", "2"])
def test_a_tag_done_at_full_epochs_or_more_is_skipped(world, epochs):
    w = world(FULL_EPOCHS=epochs)
    w.folder("tf_fused_full", [0.1, 0.2, 0.3])
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3])
    before = _listing(w.runs)
    w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


@pytest.mark.parametrize("state", ["done", "partial"])
def test_a_folder_started_with_other_flags_stops_the_launch_before_any_change(world, state):
    w = world(FULL_EPOCHS="5")
    other = [("4" if before == "--batch-size" else flag) for before, flag in zip(["", *w.flags("tf_fused_full")], w.flags("tf_fused_full"))]
    assert other != w.flags("tf_fused_full")
    w.folder("tf_fused_full", [0.1, 0.2, 0.3], args=other, done=state == "done")
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match="tf_fused_full: .* was started with different flags"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []  # DONE.json was not moved aside


def test_flags_a_resume_may_change_do_not_stop_the_launch(world):
    w = world(FULL_EPOCHS="3")
    stored = [*w.ns["ARMS"]["tf_fused_full"], "--batch-size", "8", "--no-cosine", "--standardize", "--epochs", "2",
              "--num-workers", "7", "--device", "cuda", "--data-root", "/elsewhere"]
    w.folder("tf_fused_full", [0.1, 0.2, 0.3], args=stored)
    w.run(LAUNCH)
    assert w.ns["flags_differ"]("tf_fused_full") is None
    assert w.ns["flags_differ"]("tf_collapsed_full") is None  # no folder yet


def test_log_check_and_the_monitor_on_finished_runs(world, capsys):
    w = world(FULL_EPOCHS="3")
    for tag in w.ns["RUN_TAGS"]:
        w.folder(tag, [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.run(MONITOR)
    assert w.clock.sleeps == 0 and w.launched == []
    assert "tf_fused_full      done     epoch   3/3" in capsys.readouterr().out
    assert w.ns["log_check"]("tf_fused_full") == (True, [])
    (w.runs / "tf_fused_full" / "train.log").write_text("starting\n")
    assert w.ns["log_check"]("tf_fused_full") == (False, [])  # banner not written yet: not wrong, not complete


@pytest.mark.parametrize(
    "tag, kind, line",
    [
        ("tf_fused_full", "images", "train images 514  val images 167  downsample 2  crop None"),
        ("tf_fused_full", "images", "train images 64  val images 167  downsample 1  crop None"),
        ("tf_fused_full", "images", "train images 514  val images 16  downsample 1  crop None"),
        ("tf_collapsed_full", "params", "BandTransformer parameters: 488,487 (0.488M)"),
        ("tf_collapsed_full", "tokenizer", "transformer  tokenizer band  blocks 2+6" + TOKENIZER_TAIL),
    ],
)
def test_monitor_stops_on_a_log_from_the_wrong_run(world, tag, kind, line):
    w = world(FULL_EPOCHS="3")
    for name in w.ns["RUN_TAGS"]:
        w.folder(name, [0.1, 0.2, 0.3])
    log = w.runs / tag / "train.log"
    # A resumed run prints the banner twice: a good first copy does not hide a wrong second one.
    log.write_text(log.read_text() + line + "\n")
    w.run(LAUNCH)
    with pytest.raises(RuntimeError, match=f"{tag}: .* shows the wrong run"):
        w.run(MONITOR)
    assert w.ns["log_check"](tag) == (True, [line])


def test_monitor_raises_for_a_live_run_whose_log_stopped_growing(world):
    w = world()
    w.folder("tf_fused_full", [0.1, 0.2, 0.3], done=False, running=True)
    w.run(LAUNCH)
    assert w.launched == []  # a live run is left alone
    with pytest.raises(RuntimeError, match=r"tf_fused_full is stuck: .* for 3900 s \(limit 3600 s\)"):
        w.run(MONITOR)
    assert w.clock.sleeps == 13  # 13 polls of 300 s: the first time the quiet time is over the 3600 s floor


def test_monitor_names_a_dead_run_and_still_finishes_the_others(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full", [0.1, 0.2], done=False)  # stopped at epoch 2; the recorder starts nothing
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    assert w.launched == [("tf_fused_full", w.flags("tf_fused_full"))]
    with pytest.raises(RuntimeError, match=r"\['tf_fused_full'\] died"):
        w.run(MONITOR)
    assert "tf_fused_full is not running and not done" in capsys.readouterr().out
    assert len(w.launched) == 1  # the monitor does not start a dead run again


def test_monitor_extends_a_run_that_finished_at_an_older_lower_target(world, capsys):
    w = world(FULL_EPOCHS="3")
    for tag in w.ns["RUN_TAGS"]:
        w.folder(tag, [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.ns.update(FULL_EPOCHS=5, WAIT_FOR_RUNS=False)  # as if the runs had been started at 3 and finished just now
    w.hold = True
    w.run(MONITOR)
    out = capsys.readouterr().out
    assert "EXTEND tf_fused_full" in out and "tf_fused_full      running  epoch   3/5" in out
    assert not (w.runs / "tf_fused_full" / gates.DONE_FILE).exists()
    assert (w.runs / "tf_fused_full" / "DONE.after_3_epochs.json").is_file()
    assert w.launched == [("tf_fused_full", w.flags("tf_fused_full"))]  # one at a time: the control waits its turn


def test_monitor_with_extend_off_raises_for_a_short_run(world):
    w = world(FULL_EPOCHS="3", FULL_EXTEND="0")
    for tag in w.ns["RUN_TAGS"]:
        w.folder(tag, [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.ns["FULL_EPOCHS"] = 5
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match="finished with fewer than 5 epochs: .* EXTEND is off"):
        w.run(MONITOR)
    assert _listing(w.runs) == before and w.launched == []


def test_table_compares_the_epochs_both_runs_have(world, capsys):
    w = world(FULL_EPOCHS="25")
    fused = [0.20] * 30
    fused[11] = 0.30  # epoch 12: the best of epochs 1 to 25
    fused[25:] = [0.50] * 5  # epochs 26 to 30: the control does not have them
    control = [0.25] * 25
    control[19] = 0.28  # epoch 20
    w.folder("tf_fused_full", fused)
    w.folder("tf_collapsed_full", control)
    w.run(LAUNCH)
    w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    assert w.launched == []
    lines = out.splitlines()

    def line_with(text):
        found = [line for line in lines if text in line]
        assert len(found) == 1, (text, found)
        return found[0].split()

    # The rows: tag, state, best, its epoch, epochs run, last-10 mean, s/epoch, parameters, log, flags.
    assert line_with("tf_fused_full      done") == ["tf_fused_full", "done", "0.500", "26", "30", "0.350", "60.0", "488,487", "ok", "ok"]
    assert line_with("tf_collapsed_full  done") == ["tf_collapsed_full", "done", "0.280", "20", "25", "0.253", "60.0", "491,191", "ok", "ok"]
    assert "WARNING: the two runs have different epoch counts (30 and 25)" in out
    assert "uses epochs 1 to 25 of each" in out
    comparison = w.ns["COMPARISON"]
    assert comparison["epochs"] == 25
    assert (comparison["fused"]["best_epoch"], comparison["control"]["best_epoch"]) == (12, 20)
    # Best epoch of each: 0.30 - 0.28. Last 10 of each (epochs 16 to 25): 0.20 - 0.253.
    assert line_with("best epoch of each (fused 12, control 20)")[-8:] == ["+2.0"] * 8
    assert line_with("mean of the last 10 epochs of each")[-8:] == ["-5.3"] * 8
    # Still rising: the last 10 against the 10 before, over each run's own epochs.
    assert "0.210 -> 0.350  +14.0 points  still rising" in out
    assert "0.250 -> 0.253  +0.3 points  not clearly rising" in out
    for text in (
        "One seed, validation only.", "tf_fused_full 30, tf_collapsed_full 25", "up to 200 epochs, three seeds",
        "warmup-then-cosine", "still a screen, not the protocol run", "This run does not settle the hypothesis",
    ):
        assert text in out
    assert "level with" not in out


def test_table_has_no_warning_for_equal_runs_and_handles_missing_scores(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full", [0.1, None, 0.3])
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.25])
    w.run(LAUNCH)
    w.run(MONITOR)
    w.run(TABLE)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "epochs run: tf_fused_full 3, tf_collapsed_full 3" in out
    assert "n/a (needs 20 epochs; has 3)" in out
    assert "best epoch of each (fused 3, control 3)" in out


def test_table_marks_a_row_from_other_flags(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full", [0.1, 0.2, 0.3], args=["--model", "transformer", "--lr", "1e-2", "--epochs", "3"])
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3])
    with pytest.raises(RuntimeError, match="different flags"):
        w.run(LAUNCH)
    # Run by hand past the Launch cell's stop, the table still says so.
    w.run(MONITOR)
    w.run(TABLE)
    out = capsys.readouterr().out
    assert re.search(r"tf_fused_full +done .* ok +DIFFER", out)
    assert re.search(r"tf_collapsed_full +done .* ok +ok", out)
    assert "WARNING: ['tf_fused_full'] were started with flags that differ" in out
