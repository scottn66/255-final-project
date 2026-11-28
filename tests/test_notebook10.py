"""notebooks/10_fullres_head_skip.ipynb: valid, and its flags build the models.

The notebook is notebook 09 with ``--head-skip`` on both models, plus
notebook 09's two finished runs as read-only reference tags and the skip
probe at the end of the table.

The second half runs the notebook's own cells against folders made by hand:
``gates.launch`` is replaced by a recorder, the clock is a fake, the probe
is a recorder too, and nothing is trained.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import types
import weakref
from pathlib import Path

import pytest

from sw import gates
from sw.data import CLASS_NAMES
from sw.train import _build_parser, build_model, config_from_args

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "10_fullres_head_skip.ipynb"
NOTEBOOK_09 = NOTEBOOK.with_name("09_fullres_fused_vs_control.ipynb")
PARAMS = {
    "tf_fused_full_skip": 492_599, "tf_collapsed_full_skip": 495_303,
    "tf_fused_full": 488_487, "tf_collapsed_full": 491_191,
}
RUN = ["tf_fused_full_skip", "tf_collapsed_full_skip"]
REFERENCE = ["tf_fused_full", "tf_collapsed_full"]
SKIP_LINE = "head skip  on  1x1 conv 224->16 from the model input, at its resolution"
FULL_ENV = (
    "SW_BRANCH", "FULL_RUNS_DIR", "FULL_WORK", "FULL_DATA", "FULL_ZIP", "FULL_DEVICE",
    "FULL_WORKERS", "FULL_POLL", "FULL_EPOCHS", "FULL_TAGS", "FULL_ONE_AT_A_TIME",
    "FULL_STALL_FLOOR", "FULL_EXTEND", "FULL_PROBE",
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


def test_notebook_is_valid_and_committed_without_outputs():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"]
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None  # not run yet
            ast.parse("".join(cell["source"]))
    assert "widgets" not in nb["metadata"]
    header = "".join(nb["cells"][0]["source"])
    assert "/blob/wp7-head-skip/notebooks/10_fullres_head_skip.ipynb" in header
    assert header.rstrip().endswith("AI-assisted notebook; Scott reviews every line.")


def test_default_settings(settings):
    assert settings["BRANCH"] == "wp7-head-skip"
    # Notebook 09's runs folder, so its finished no-skip folders sit beside the new ones.
    assert settings["RUNS"] == Path("/content/drive/MyDrive/sw_runs/fullres")
    assert settings["FULL_EPOCHS"] == 60
    assert settings["EXTEND"] is True
    assert settings["RUN_TAGS"] == RUN
    assert settings["REFERENCE_TAGS"] == REFERENCE
    assert (settings["FUSED"], settings["CONTROL"]) == tuple(RUN)
    assert settings["SKIP_OF"] == dict(zip(REFERENCE, RUN))
    assert settings["STALL_FLOOR_SECONDS"] == 3600
    assert settings["ONE_AT_A_TIME"] is True
    assert settings["PROBE"] is True
    # Only the skip tags can be launched. The reference tags have flags to compare with, and no more.
    assert list(settings["FULL"]) == list(settings["ARMS"]) == RUN
    assert list(settings["REFERENCE"]) == REFERENCE
    assert set(settings["EXPECTED"]) == set(PARAMS)
    assert settings["SKIP_LINE"] == SKIP_LINE


def test_run_tags_are_notebook_09_flags_plus_head_skip_and_nothing_else(monkeypatch, settings):
    for name in FULL_ENV:
        monkeypatch.delenv(name, raising=False)
    nb09 = json.loads(NOTEBOOK_09.read_text(encoding="utf-8"))
    first = next("".join(cell["source"]) for cell in nb09["cells"] if cell["cell_type"] == "code")
    old: dict = {}
    exec(compile(first, "notebook 09 settings", "exec"), old)
    assert old["RUNS"] == settings["RUNS"]
    assert old["SHARED"] == settings["SHARED"]
    for tag in REFERENCE:
        # The flags this notebook expects in a reference folder are the ones notebook 09 launched it with.
        assert settings["REFERENCE"][tag] == old["FULL"][tag]
        skip = settings["FULL"][settings["SKIP_OF"][tag]]
        assert skip.count("--head-skip") == 1
        assert [flag for flag in skip if flag != "--head-skip"] == old["FULL"][tag]
        assert "--head-skip" not in old["FULL"][tag]
        # The two differ in the run identity sw.gates stores, so neither folder can be resumed as the other.
        assert gates.run_identity(skip) != gates.run_identity(old["FULL"][tag])


def test_environment_overrides(monkeypatch):
    names = _settings(
        monkeypatch, FULL_EPOCHS="90", FULL_EXTEND="0", FULL_TAGS="tf_collapsed_full_skip", FULL_RUNS_DIR="/tmp/x",
        FULL_PROBE="0",
    )
    assert names["FULL_EPOCHS"] == 90
    assert names["EXTEND"] is False
    assert names["PROBE"] is False
    assert names["RUN_TAGS"] == ["tf_collapsed_full_skip"]
    assert names["RUNS"] == Path("/tmp/x")
    for flags in names["EXPECTED"].values():
        assert flags[flags.index("--epochs") + 1] == "90"
    # A reference tag cannot be run from here, and neither can a tag from another notebook.
    for tags in ("tf_fused_full", "tf_fused_full_skip,tf_collapsed_full", "cnn_full", "tf_fused_std"):
        with pytest.raises(ValueError, match="RUN_TAGS"):
            _settings(monkeypatch, FULL_TAGS=tags)


@pytest.mark.parametrize("tag", sorted(PARAMS))
def test_flags_parse_and_build_the_expected_model(settings, tag):
    flags = settings["EXPECTED"][tag]
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
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.head_skip is (tag in RUN)
    if tag.startswith("tf_collapsed_full"):
        assert (cfg.tokenizer, cfg.depth_spectral, cfg.depth_spatial) == ("collapsed", 0, 8)
    else:
        assert (cfg.tokenizer, cfg.depth_spectral, cfg.depth_spatial) == ("band", 2, 6)
    model = build_model(cfg, in_channels=args.in_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == PARAMS[tag]
    # The lines the smoke cell and the monitor look for are the ones sw.train prints.
    line = f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)"
    assert settings["PARAM_LINE"][tag] == line
    assert f"downsample {cfg.downsample}  " == settings["RESOLUTION_TEXT"]
    if tag in RUN:
        conv = model.head.skip[0]
        assert f"head skip  on  1x1 conv {conv.in_channels}->{conv.out_channels} from the model input, at its resolution" == SKIP_LINE
    else:
        assert model.head.skip is None


def test_the_head_skip_line_is_the_one_sw_train_prints():
    """The notebook matches this line in every log, so it must be the text in ``sw.train``."""
    import inspect

    import sw.train

    assert 'print(f"head skip  on  1x1 conv {conv.in_channels}->{conv.out_channels} from the model input, at its resolution")' in inspect.getsource(sw.train.train)


def test_epochs_flag_is_set_once_and_downsample_only_in_the_smoke():
    setup, data, smoke, launch, monitor, table = _code_cells()[1:]
    later = "\n".join((setup, data, smoke, launch, monitor, table))
    assert '"--epochs"' not in later
    assert "--stats-path" not in later
    assert "--head-skip" not in "\n".join((data, smoke, launch, monitor, table))  # set once, in Settings
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
    assert "flags_error(flags)" in setup and "needs --head-skip" in setup
    # Every train.log must show the native-size banner.
    assert "RESOLUTION_TEXT" in data and "IMAGES_LINE" in monitor
    assert "RESOLUTION_TEXT" in smoke and "SKIP_LINE" in smoke and "SKIP_LINE" in monitor
    # The memory print runs on cuda only.
    assert 'if DEVICE == "cuda":' in smoke and "max_memory_allocated" in smoke
    # A short finished tag is reopened or refused before anything is launched.
    assert launch.index("check_flags()\n") < launch.index("extend_or_refuse()\n") < launch.rindex("start_due()")
    assert "< FULL_EPOCHS" in launch and "if not EXTEND:" in launch
    assert '"last.pt"' in launch  # no checkpoint: refuse, do not restart from epoch 1
    # A live run whose log stopped growing raises; a dead one does not block the next tag.
    assert "quiet > limit" in monitor and "is stuck" in monitor
    assert "if tag in DEAD:\n            continue" in launch
    # Launch, extend and the monitor loop over RUN_TAGS only; the reference tags appear in the table alone.
    for source in (smoke, launch, monitor):
        assert "REFERENCE" not in source
    assert "never launches it" in launch
    # The table warns when two runs differ in length, and states the limits of the run.
    assert "WARNING: the two runs have different epoch counts" in table
    assert "it does not settle the hypothesis" in table
    assert '"competitive"' in table
    assert "chosen on this validation set" in table
    assert "the skip alone does it" in table
    assert "not retrained models" in table
    # The probe is guarded: a failure is printed and the cell goes on.
    assert "except Exception as err:" in table and "sw.skip_probe" in table


def test_wording():
    text = _all_text().lower()
    for word in ("beat", "outperform", "better than"):
        assert word not in text
    header = "".join(_notebook()["cells"][0]["source"])
    assert "**Cost (an estimate).**" in header
    assert "63.4 s per epoch" in header and "With the skip the time has not been measured" in header
    assert "chosen on this validation set" in header
    assert "never launched and never changed" in header
    assert "near zero on the two thin classes at their best epoch" in header and "ended near zero" not in header
    # What differs between a skip run and its reference besides the skip, and what the skip does to fused minus control.
    table_text = "".join(_notebook()["cells"][2 * TABLE + 1]["source"]) + _code_cells()[TABLE]
    for text in (header, table_text):
        assert "start from different" in text and "random weights" in text
        assert "318358e" in text and "spread is not" in text and "still rising at epoch 60" in text
        assert "not evidence of an effect" in text
        assert "as the control's tokenizer does" in text or "which is what the control's tokenizer does" in text
        assert "says little about band tokens" in text
        assert "four runs, one seed each" in text
        assert "does not show" in text and "that the transformer is needed" in text


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
        self.probed: list[list[str]] = []
        self.probe_kwargs: list[dict] = []
        self.probe_result = types.SimpleNamespace(returncode=0, stdout="normal 0.500\nskip zeroed 0.400\ntokens zeroed 0.100\n", stderr="")
        self.hold = False  # True: a launched tag then reads as running (this test process holds it)

        def launch(out_dir, train_args, env=None, **kw):
            self.launched.append((Path(out_dir).name, list(train_args)))
            if self.hold:
                (Path(out_dir) / gates.RUNNING_FILE).write_text(json.dumps(gates._holder_record(os.getpid())))

        def probe(cmd, **kwargs):
            self.probed.append(list(cmd))
            self.probe_kwargs.append(kwargs)
            if isinstance(self.probe_result, Exception):
                raise self.probe_result
            return self.probe_result

        monkeypatch.setattr(gates, "launch", launch)
        self.ns.update(
            subprocess=types.SimpleNamespace(run=probe), sys=types.SimpleNamespace(executable="python"), REPO=tmp_path,
            json=json, re=re, time=self.clock, gates=gates, SW_ROOT=tmp_path / "data", ENV={}, COMMIT="test",
            T_START=self.clock.now, tail=lambda path, n=25: "(log tail)",
            IMAGES_LINE=f"train images 514  val images 167  {self.ns['RESOLUTION_TEXT']}",
        )
        self.runs = self.ns["RUNS"]

    def run(self, index):
        exec(compile(_code_cells()[index], f"cell {index}", "exec"), self.ns)

    def flags(self, tag):
        return [*self.ns["EXPECTED"][tag], "--data-root", str(self.ns["SW_ROOT"])]

    def folder(self, tag, values, done=True, last=True, args=None, banner=None, running=False, best=False):
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
        if best:
            (out / "best.pt").write_bytes(b"not a real checkpoint")
        if done:
            (out / gates.DONE_FILE).write_text(json.dumps({"result": {"param_count": PARAMS[tag]}}))
        if running:  # held by this test process, which is alive
            (out / gates.RUNNING_FILE).write_text(json.dumps(gates._holder_record(os.getpid())))
        return out

    def banner(self, tag, images=None, params=None):
        lines = [params or self.ns["PARAM_LINE"][tag]]
        if self.ns["TOKENIZER_LINE"][tag]:
            lines.append(self.ns["TOKENIZER_LINE"][tag] + TOKENIZER_TAIL)
        if tag in RUN:
            lines.append(SKIP_LINE)
        lines.append(images or (self.ns["IMAGES_LINE"] + "crop None  augment on  out x"))
        return lines


@pytest.fixture
def world(monkeypatch, tmp_path):
    return lambda **env: World(monkeypatch, tmp_path, **env)


def _listing(root):
    return sorted((str(p.relative_to(root)), p.read_bytes()) for p in root.rglob("*") if p.is_file())


def test_no_cell_after_settings_adds_a_training_flag():
    for source in _code_cells()[SETUP:]:
        for flag in ("--patience", "--amp", "--max-train", "--max-val", "--stats-path", "--crop-size", "--batch-size", '"--head-skip"'):
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
    assert "unrecognized arguments" in names["flags_error"](["--model", "transformer", "--head-skipp"])
    for tag in RUN:
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

    w.ns.update(subprocess=types.SimpleNamespace(run=run), WORK=w.runs.parent / "work", REPO=NOTEBOOK.parents[1], DEVICE=device)
    return w, commands


def _good_smoke(w, tag):
    return "\n".join(w.banner(tag, images="train images 4  val images 4  downsample 1  crop None  augment on")) + "\n"


def test_smoke_runs_every_model_at_native_size_and_reads_the_banner(world, capsys):
    w, commands = _smoke_world(world, _good_smoke)
    w.run(SMOKE)
    assert len(commands) == 2
    assert list(w.ns["ARMS"]) == RUN
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
        ("492,599 (0.493M)", "488,487 (0.488M)"),  # the fused model without the skip
        ("495,303 (0.495M)", "491,191 (0.491M)"),
        ("tokenizer collapsed  blocks 0+8", "tokenizer band  blocks 2+6"),
        (SKIP_LINE + "\n", ""),  # the count is right but the line is missing
        ("224->16", "224->8"),
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
        if tag == "tf_collapsed_full_skip":
            raise RuntimeError("CUDA out of memory")
        return 1.5

    cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: alive_at_empty.append(made[-1]() is not None)
    )
    w.ns.update(torch=types.SimpleNamespace(cuda=cuda), gc=gc)
    w.ns["measure_memory"](RUN, measure=measure)
    assert alive_at_empty == [False, False]  # emptied once per tag, each time with nothing held
    out = capsys.readouterr().out
    assert "tf_fused_full_skip     1.50 GiB" in out
    assert "tf_collapsed_full_skip could not be measured: RuntimeError: CUDA out of memory" in out
    # Only the tags that will run are measured.
    assert "measure_memory(RUN_TAGS)" in _code_cells()[SMOKE]


def test_launch_passes_exactly_the_tag_flags_and_the_data_root(world):
    w = world()
    w.run(LAUNCH)
    assert w.launched == [("tf_fused_full_skip", [*w.ns["FULL"]["tf_fused_full_skip"], "--data-root", str(w.ns["SW_ROOT"])])]
    for flag in NOT_ALLOWED:
        assert flag not in w.launched[0][1]
    # The control starts only after the fused run is done, and a dead tag does not block it.
    w.ns["start_due"]()
    assert len(w.launched) == 1
    w.ns["DEAD"].add("tf_fused_full_skip")
    w.ns["start_due"]()
    assert w.launched[1] == ("tf_collapsed_full_skip", w.flags("tf_collapsed_full_skip"))


def test_launch_all_at_once(world):
    w = world(FULL_ONE_AT_A_TIME="0")
    w.run(LAUNCH)
    assert w.launched == [(tag, w.flags(tag)) for tag in RUN]


def test_reference_folders_are_never_launched_extended_or_changed(world, capsys):
    """Short, unfinished or missing-checkpoint reference folders are left exactly as they are."""
    w = world(FULL_EPOCHS="5")
    w.folder("tf_fused_full", [0.1, 0.2, 0.3])  # finished at 3 of 5: a run tag would be extended
    w.folder("tf_collapsed_full", [0.1, 0.2], done=False)  # stopped: a run tag would be resumed
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3, 0.4, 0.5])
    before = _listing(w.runs)
    w.run(LAUNCH)
    w.run(MONITOR)
    w.run(TABLE)
    assert w.launched == [] and _listing(w.runs) == before
    out = capsys.readouterr().out
    assert "EXTEND" not in out
    assert "not finished: ['tf_collapsed_full']" in out
    # Asked by hand, start_due still refuses a reference tag.
    w.ns["RUN_TAGS"] = ["tf_fused_full"]
    w.ns["STARTED"].clear()
    (w.runs / "tf_fused_full" / gates.DONE_FILE).unlink()
    with pytest.raises(RuntimeError, match="reference tag; this notebook never launches it"):
        w.ns["start_due"]()
    assert w.launched == []


def test_extend_reopens_a_short_finished_tag(world, capsys):
    w = world(FULL_EPOCHS="5")
    out = w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3])
    record = (out / gates.DONE_FILE).read_text()
    w.run(LAUNCH)
    assert not (out / gates.DONE_FILE).exists()
    assert (out / "DONE.after_3_epochs.json").read_text() == record
    assert "EXTEND tf_fused_full_skip: it finished at 3 epochs and FULL_EPOCHS is 5" in capsys.readouterr().out
    assert w.launched == [("tf_fused_full_skip", w.flags("tf_fused_full_skip"))]
    assert w.launched[0][1][w.launched[0][1].index("--epochs") + 1] == "5"


def test_extend_off_refuses_and_changes_nothing(world):
    w = world(FULL_EPOCHS="5", FULL_EXTEND="0")
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3])
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match=r"finished with fewer than 5 epochs: \{'tf_fused_full_skip': 3\}"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


def test_extend_without_a_checkpoint_refuses_and_changes_nothing(world):
    w = world(FULL_EPOCHS="5")
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3])
    w.folder("tf_collapsed_full_skip", [0.1, 0.2, 0.3], last=False)
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match=r"\['tf_collapsed_full_skip'\].*no last.pt"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


@pytest.mark.parametrize("epochs", ["3", "2"])
def test_a_tag_done_at_full_epochs_or_more_is_skipped(world, epochs):
    w = world(FULL_EPOCHS=epochs)
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3])
    w.folder("tf_collapsed_full_skip", [0.1, 0.2, 0.3])
    before = _listing(w.runs)
    w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []


@pytest.mark.parametrize("state", ["done", "partial"])
def test_a_folder_started_with_other_flags_stops_the_launch_before_any_change(world, state):
    w = world(FULL_EPOCHS="5")
    other = [("4" if before == "--batch-size" else flag) for before, flag in zip(["", *w.flags("tf_fused_full_skip")], w.flags("tf_fused_full_skip"))]
    assert other != w.flags("tf_fused_full_skip")
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3], args=other, done=state == "done")
    before = _listing(w.runs)
    with pytest.raises(RuntimeError, match="tf_fused_full_skip: .* was started with different flags"):
        w.run(LAUNCH)
    assert _listing(w.runs) == before and w.launched == []  # DONE.json was not moved aside


def test_flags_a_resume_may_change_do_not_stop_the_launch(world):
    w = world(FULL_EPOCHS="3")
    stored = [*w.ns["ARMS"]["tf_fused_full_skip"], "--batch-size", "8", "--no-cosine", "--standardize", "--epochs", "2",
              "--num-workers", "7", "--device", "cuda", "--data-root", "/elsewhere"]
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3], args=stored)
    w.run(LAUNCH)
    assert w.ns["flags_differ"]("tf_fused_full_skip") is None
    assert w.ns["flags_differ"]("tf_collapsed_full_skip") is None  # no folder yet


def test_log_check_and_the_monitor_on_finished_runs(world, capsys):
    w = world(FULL_EPOCHS="3")
    for tag in w.ns["RUN_TAGS"]:
        w.folder(tag, [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.run(MONITOR)
    assert w.clock.sleeps == 0 and w.launched == []
    assert "tf_fused_full_skip     done     epoch   3/3" in capsys.readouterr().out
    assert w.ns["log_check"]("tf_fused_full_skip") == (True, [])
    (w.runs / "tf_fused_full_skip" / "train.log").write_text("starting\n")
    assert w.ns["log_check"]("tf_fused_full_skip") == (False, [])  # banner not written yet: not wrong, not complete


@pytest.mark.parametrize(
    "tag, kind, line",
    [
        ("tf_fused_full_skip", "images", "train images 514  val images 167  downsample 2  crop None"),
        ("tf_fused_full_skip", "images", "train images 64  val images 167  downsample 1  crop None"),
        ("tf_fused_full_skip", "images", "train images 514  val images 16  downsample 1  crop None"),
        ("tf_collapsed_full_skip", "params", "BandTransformer parameters: 491,191 (0.491M)"),  # the control without the skip
        ("tf_fused_full_skip", "skip", "head skip  on  1x1 conv 224->8 from the model input, at its resolution"),
        ("tf_collapsed_full_skip", "tokenizer", "transformer  tokenizer band  blocks 2+6" + TOKENIZER_TAIL),
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
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3], done=False, running=True)
    w.run(LAUNCH)
    assert w.launched == []  # a live run is left alone
    with pytest.raises(RuntimeError, match=r"tf_fused_full_skip is stuck: .* for 3900 s \(limit 3600 s\)"):
        w.run(MONITOR)
    assert w.clock.sleeps == 13  # 13 polls of 300 s: the first time the quiet time is over the 3600 s floor


def test_monitor_names_a_dead_run_and_still_finishes_the_others(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full_skip", [0.1, 0.2], done=False)  # stopped at epoch 2; the recorder starts nothing
    w.folder("tf_collapsed_full_skip", [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    assert w.launched == [("tf_fused_full_skip", w.flags("tf_fused_full_skip"))]
    with pytest.raises(RuntimeError, match=r"\['tf_fused_full_skip'\] died"):
        w.run(MONITOR)
    assert "tf_fused_full_skip is not running and not done" in capsys.readouterr().out
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
    assert "EXTEND tf_fused_full_skip" in out and "tf_fused_full_skip     running  epoch   3/5" in out
    assert not (w.runs / "tf_fused_full_skip" / gates.DONE_FILE).exists()
    assert (w.runs / "tf_fused_full_skip" / "DONE.after_3_epochs.json").is_file()
    assert w.launched == [("tf_fused_full_skip", w.flags("tf_fused_full_skip"))]  # one at a time: the control waits its turn


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


def _line_with(out, text):
    found = [line for line in out.splitlines() if text in line]
    assert len(found) == 1, (text, found)
    return found[0].split()


def _block(out, title):
    """The lines of one comparison: from its title to the next blank line."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    stop = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
    return "\n".join(lines[start:stop])


def test_table_without_reference_folders_says_so_and_still_compares_the_arms(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3])
    w.folder("tf_collapsed_full_skip", [0.1, 0.2, 0.25])
    w.run(LAUNCH)
    w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    assert w.launched == [] and w.ns["REFERENCE_FOUND"] == []
    assert _line_with(out, "tf_fused_full_skip     done") == ["tf_fused_full_skip", "done", "0.300", "3", "3", "0.200", "60.0", "492,599", "ok", "ok"]
    for tag, skip_tag in zip(REFERENCE, RUN):
        assert f"{tag}: no folder in {w.runs}." in out
        assert f"there is no 'skip minus no skip' line for {skip_tag}." in out
    assert "skip minus no skip:" not in out
    assert "WARNING" not in out
    assert "fused minus control, with the skip: tf_fused_full_skip minus tf_collapsed_full_skip" in out
    assert "without the skip (notebook 09's runs)" not in out
    assert "best epoch of each (fused 3, control 3)" in out
    assert "n/a (needs 20 epochs; has 3)" in out
    assert list(w.ns["COMPARISONS"]) == ["with skip"]
    assert "No reference folder was found" in out
    assert not w.runs.joinpath("tf_fused_full").exists()  # reading did not create it


def test_table_with_reference_folders_prints_skip_minus_no_skip_over_shared_epochs(world, capsys):
    w = world(FULL_EPOCHS="30")
    skip = [0.20] * 30
    skip[11] = 0.30  # epoch 12: the best of epochs 1 to 25
    skip[25:] = [0.50] * 5  # epochs 26 to 30: the reference does not have them
    reference = [0.25] * 25
    reference[19] = 0.28  # epoch 20
    w.folder("tf_fused_full_skip", skip)
    w.folder("tf_fused_full", reference)  # notebook 09's run stopped at 25 epochs
    w.folder("tf_collapsed_full_skip", [0.40] * 30)
    w.folder("tf_collapsed_full", [0.30] * 30)
    before = _listing(w.runs)
    w.run(LAUNCH)
    w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    assert w.launched == [] and _listing(w.runs) == before
    assert w.ns["REFERENCE_FOUND"] == REFERENCE

    # The rows: the run tags first, then the references. Each with its own parameter count.
    rows = [line.split()[0] for line in out.splitlines() if re.match(r"tf_\w+ +done +\d", line)]
    assert rows == [*RUN, *REFERENCE]
    assert _line_with(out, "tf_fused_full_skip     done") == ["tf_fused_full_skip", "done", "0.500", "26", "30", "0.350", "60.0", "492,599", "ok", "ok"]
    assert _line_with(out, "tf_fused_full          done") == ["tf_fused_full", "done", "0.280", "20", "25", "0.253", "60.0", "488,487", "ok", "ok"]

    # Fused arm: 30 against 25 epochs. Only epochs 1 to 25 of each are compared, with a warning.
    fused = _block(out, "skip minus no skip: tf_fused_full_skip minus tf_fused_full,")
    assert "epochs run: tf_fused_full_skip 30, tf_fused_full 25" in fused
    assert "WARNING: the two runs have different epoch counts (30 and 25)" in fused
    assert "uses epochs 1 to 25 of each" in fused
    # Best epoch of each: 0.30 - 0.28. Last 10 of each (epochs 16 to 25): 0.20 - 0.253.
    assert _line_with(fused, "best epoch of each (skip 12, no skip 20)")[-8:] == ["+2.0"] * 8
    assert _line_with(fused, "mean of the last 10 epochs of each")[-8:] == ["-5.3"] * 8
    comparison = w.ns["COMPARISONS"]["skip:tf_fused_full"]
    assert comparison["epochs"] == 25
    assert (comparison["skip"]["best_epoch"], comparison["no skip"]["best_epoch"]) == (12, 20)

    # Control arm: equal lengths, no warning.
    control = _block(out, "skip minus no skip: tf_collapsed_full_skip minus tf_collapsed_full,")
    assert "WARNING" not in control
    assert _line_with(control, "best epoch of each (skip 1, no skip 1)")[-8:] == ["+10.0"] * 8
    assert _line_with(control, "mean of the last 10 epochs of each")[-8:] == ["+10.0"] * 8

    # Fused minus control, with the skip (30 epochs each) and without (25 against 30).
    with_skip = _block(out, "fused minus control, with the skip")
    assert "WARNING" not in with_skip
    assert _line_with(with_skip, "best epoch of each (fused 26, control 1)")[-8:] == ["+10.0"] * 8
    without = _block(out, "fused minus control, without the skip (notebook 09's runs)")
    assert "epochs run: tf_fused_full 25, tf_collapsed_full 30" in without and "WARNING" in without
    assert _line_with(without, "best epoch of each (fused 20, control 1)")[-8:] == ["-2.0"] * 8
    assert list(w.ns["COMPARISONS"]) == ["skip:tf_fused_full", "skip:tf_collapsed_full", "with skip", "no skip"]

    # Still rising: the last 10 against the 10 before, over each run's own epochs.
    assert "0.210 -> 0.350  +14.0 points  still rising" in out
    for text in (
        "One seed, validation only, constant learning rate.",
        "tf_fused_full_skip 30, tf_collapsed_full_skip 30, tf_fused_full 25, tf_collapsed_full 30",
        "up to 200 epochs, three seeds", "warmup-then-cosine", "a screen, not the protocol run",
        "it does not settle the hypothesis", "chosen on this validation set",
        "'tokens zeroed (skip path only)' row means the skip alone does it",
        "an ablation at test time, not retrained models",
        "The reference rows are notebook 09's runs, read from their folders and not trained again here.",
    ):
        assert text in out
    assert "No reference folder was found" not in out


def test_table_with_one_reference_folder_compares_that_arm_only(world, capsys):
    w = world(FULL_EPOCHS="3")
    for tag in (*RUN, "tf_collapsed_full"):
        w.folder(tag, [0.1, None, 0.3])
    w.run(LAUNCH)
    w.run(MONITOR)
    w.run(TABLE)
    out = capsys.readouterr().out
    assert w.ns["REFERENCE_FOUND"] == ["tf_collapsed_full"]
    assert "tf_fused_full: no folder in" in out
    assert "skip minus no skip: tf_collapsed_full_skip minus tf_collapsed_full," in out
    assert "skip minus no skip: tf_fused_full_skip" not in out
    assert "without the skip (notebook 09's runs)" not in out  # needs both references
    assert list(w.ns["COMPARISONS"]) == ["skip:tf_collapsed_full", "with skip"]


def test_table_marks_rows_from_other_flags_and_a_reference_log_with_the_skip(world, capsys):
    w = world(FULL_EPOCHS="3")
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3])
    # A "reference" that was really trained with the skip: its flags and its log both say so.
    w.folder(
        "tf_fused_full", [0.1, 0.2, 0.3], args=w.flags("tf_fused_full_skip"),
        banner=[*w.banner("tf_fused_full"), SKIP_LINE],
    )
    # A reference at another batch size.
    other = [("4" if before == "--batch-size" else flag) for before, flag in zip(["", *w.flags("tf_collapsed_full")], w.flags("tf_collapsed_full"))]
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3], args=other)
    assert w.ns.get("log_check") is None
    w.run(LAUNCH)  # the launch checks the run tags only; the references do not stop it
    w.run(MONITOR)
    w.run(TABLE)
    out = capsys.readouterr().out
    assert w.ns["log_check"]("tf_fused_full") == (True, [SKIP_LINE])
    assert re.search(r"tf_fused_full_skip +done .* ok +ok", out)
    assert re.search(r"tf_fused_full +done .* WRONG +DIFFER", out)
    assert re.search(r"tf_collapsed_full +done .* ok +DIFFER", out)
    assert "WARNING: ['tf_fused_full', 'tf_collapsed_full'] were started with flags that differ" in out

    # A run tag started without the skip stops the launch before any change.
    w2 = world(FULL_RUNS_DIR=str(w.runs.parent / "runs2"), FULL_EPOCHS="3")
    w2.folder("tf_fused_full_skip", [0.1, 0.2, 0.3], args=w2.flags("tf_fused_full"))
    before = _listing(w2.runs)
    with pytest.raises(RuntimeError, match="tf_fused_full_skip: .* was started with different flags") as raised:
        w2.run(LAUNCH)
    assert "--head-skip" in str(raised.value)
    assert _listing(w2.runs) == before


def test_table_warns_about_a_reference_whose_log_shows_another_model(world, capsys):
    """A reference folder with no ARGS.json has no flags to compare, so its log is the only check."""
    w = world(FULL_EPOCHS="3")
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3])
    out_dir = w.folder("tf_fused_full", [0.1, 0.2, 0.3], banner=[*w.banner("tf_fused_full"), SKIP_LINE])
    (out_dir / gates.ARGS_FILE).unlink()
    w.folder("tf_collapsed_full", [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    assert re.search(r"tf_fused_full +done .* WRONG +no record", out)
    assert "started with flags that differ" not in out
    warning = next(line for line in out.splitlines() if line.startswith("WARNING: the train.log of"))
    assert "['tf_fused_full'] shows another model" in warning
    assert warning.endswith("Do not read the comparison lines that use them.")
    assert out.index("WARNING: the train.log of") < out.index("skip minus no skip")  # before the lines it is about

    # No such line when every log is right.
    w2 = world(FULL_RUNS_DIR=str(w.runs.parent / "runs2"), FULL_EPOCHS="3")
    for tag in [*RUN, *REFERENCE]:
        w2.folder(tag, [0.1, 0.2, 0.3])
    w2.run(LAUNCH)
    w2.run(MONITOR)
    capsys.readouterr()
    w2.run(TABLE)
    assert "WARNING: the train.log of" not in capsys.readouterr().out


def test_log_check_needs_the_head_skip_line_for_a_run_tag(world):
    w = world(FULL_EPOCHS="3")
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3])
    w.run(LAUNCH)
    w.run(MONITOR)
    log = w.runs / "tf_fused_full_skip" / "train.log"
    assert w.ns["log_check"]("tf_fused_full_skip") == (True, [])
    log.write_text(log.read_text().replace(SKIP_LINE + "\n", ""))
    assert w.ns["log_check"]("tf_fused_full_skip") == (False, [])  # the banner is not complete without it


def test_probe_runs_on_finished_run_tags_only_and_reads_the_runs_own_folder(world, capsys):
    w = world(FULL_EPOCHS="3")
    w.folder("tf_fused_full_skip", [0.1, 0.2, 0.3], best=True)
    w.folder("tf_collapsed_full_skip", [0.1, 0.2], done=False, best=True)  # stopped early: not probed
    w.folder("tf_fused_full", [0.1, 0.2, 0.3], best=True)  # a reference has no skip to probe
    w.run(LAUNCH)
    with pytest.raises(RuntimeError, match="died"):  # the recorder starts nothing, so the stopped tag reads as dead
        w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    best = w.runs / "tf_fused_full_skip" / "best.pt"
    assert w.probed == [[
        "python", "-m", "sw.skip_probe", "--checkpoint", str(best), "--data-root", str(w.ns["SW_ROOT"]),
        "--device", "cpu", "--num-workers", "2",
    ]]
    # A probe that hangs must not hang the table: the call has the notebook's time limit.
    assert w.ns["PROBE_TIMEOUT"] == 1800
    assert [kwargs["timeout"] for kwargs in w.probe_kwargs] == [w.ns["PROBE_TIMEOUT"]]
    assert "--- tf_fused_full_skip\nnormal 0.500\nskip zeroed 0.400\ntokens zeroed 0.100" in out
    assert "--- tf_collapsed_full_skip: not finished, or no best.pt; not probed" in out
    assert "--stats-path" not in _code_cells()[TABLE]  # the probe finds band_stats.pt beside the checkpoint


@pytest.mark.parametrize(
    "result, text",
    [
        (types.SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nRuntimeError: the normal row does not reproduce"),
         "the probe failed; the table above does not depend on it. RuntimeError: the normal row does not reproduce"),
        (types.SimpleNamespace(returncode=2, stdout="", stderr=""), "the probe failed; the table above does not depend on it. exit code 2"),
        (TimeoutError("timed out after 1800 seconds"), "the probe could not run; the table above does not depend on it. TimeoutError: timed out"),
        (subprocess.TimeoutExpired(["python", "-m", "sw.skip_probe"], 1800),
         "the probe could not run; the table above does not depend on it. TimeoutExpired: "),
    ],
)
def test_a_failed_probe_is_printed_and_the_table_still_ends(world, capsys, result, text):
    w = world(FULL_EPOCHS="3")
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3], best=True)
    w.probe_result = result
    w.run(LAUNCH)
    w.run(MONITOR)
    capsys.readouterr()
    w.run(TABLE)
    out = capsys.readouterr().out
    assert len(w.probed) == 2  # the second tag is still tried
    for tag in RUN:
        assert f"--- {tag}: {text}" in out
    assert "One seed, validation only, constant learning rate." in out  # the closing lines still print
    assert out.rstrip().splitlines()[-1].startswith("notebook ran in")


def test_probe_can_be_turned_off(world, capsys):
    w = world(FULL_EPOCHS="3", FULL_PROBE="0")
    for tag in RUN:
        w.folder(tag, [0.1, 0.2, 0.3], best=True)
    w.run(LAUNCH)
    w.run(MONITOR)
    w.run(TABLE)
    assert w.probed == []
    assert "skipped (PROBE is off)" in capsys.readouterr().out
