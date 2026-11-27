"""notebooks/08_screen_three_arms.ipynb: valid, and its flags build the three models.

The notebook is committed with the outputs of its Colab run. They are the
record of that run, so no test asks for them to be empty.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from sw.train import _build_parser, build_model, config_from_args

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "08_screen_three_arms.ipynb"
PARAMS = {"tf_fused_std": 488_487, "tf_collapsed_std": 491_191, "cnn_std": 571_879}
SCREEN_ENV = (
    "SW_BRANCH", "SCREEN_RUNS_DIR", "SCREEN_WORK", "SCREEN_DATA", "SCREEN_ZIP", "SCREEN_DEVICE",
    "SCREEN_WORKERS", "SCREEN_POLL", "SCREEN_EPOCHS", "SCREEN_TAGS", "SCREEN_ONE_AT_A_TIME",
    "SCREEN_STALL_FLOOR",
)


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells():
    return ["".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


@pytest.fixture
def settings(monkeypatch):
    """The names the Settings cell defines, with no environment overrides."""
    for name in SCREEN_ENV:
        monkeypatch.delenv(name, raising=False)
    names: dict = {}
    exec(compile(_code_cells()[0], "settings cell", "exec"), names)
    return names


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
            # Outputs stay: they are the record of the run.
            assert isinstance(cell["outputs"], list)
            ast.parse("".join(cell["source"]))
    badge = "".join(nb["cells"][0]["source"])
    # The badge opens the file on the branch the Settings cell clones.
    assert "/blob/wp7-screen-notebook/notebooks/08_screen_three_arms.ipynb" in badge


def test_default_settings(settings):
    assert settings["BRANCH"] == "wp6-collapsed-tokenizer"
    assert settings["RUNS"] == Path("/content/drive/MyDrive/sw_runs/screen")
    assert settings["SCREEN_EPOCHS"] == 30
    assert settings["RUN_TAGS"] == ["tf_fused_std", "tf_collapsed_std", "cnn_std"]
    assert settings["STALL_FLOOR_SECONDS"] == 3600
    assert settings["ONE_AT_A_TIME"] is True
    assert set(settings["SCREEN"]) == set(PARAMS)


@pytest.mark.parametrize("tag", sorted(PARAMS))
def test_flags_parse_and_build_the_expected_model(settings, tag):
    flags = settings["SCREEN"][tag]
    assert flags.count("--epochs") == 1
    for flag in ("--stats-path", "--patience", "--amp", "--max-train-images", "--out-dir", "--resume"):
        assert flag not in flags
    args = _build_parser().parse_args(flags)
    cfg = config_from_args(args)
    assert (cfg.epochs, cfg.batch_size, cfg.downsample) == (30, 4, 2)
    assert cfg.standardize is True
    assert cfg.cosine is False
    assert cfg.stats_path is None
    assert cfg.lr == pytest.approx(1e-3 if tag == "cnn_std" else 3e-4)
    model = build_model(cfg, in_channels=args.in_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == PARAMS[tag]
    # The line the smoke cell and the monitor look for is the one sw.train prints.
    line = f"{model.__class__.__name__} parameters: {n_params:,} ({n_params / 1e6:.3f}M)"
    assert settings["PARAM_LINE"][tag] == line


def test_no_cell_appends_a_second_epochs_flag_or_waits_for_a_failed_state():
    later = "\n".join(_code_cells()[1:])
    assert "--epochs" not in later
    assert "--stats-path" not in later
    # gates.run_state never returns "failed"; a dead run reads "new" or "partial".
    assert '"failed"' not in later
    assert '("new", "partial")' in later


def test_guards_are_in_the_cells_that_need_them():
    setup, data, smoke, launch, monitor, table = _code_cells()[1:]
    # A branch whose sw.train lacks a flag stops in Setup, before the data step.
    assert "_build_parser().parse_args(flags)" in setup
    # A tag finished with fewer epochs is refused before anything is launched.
    assert launch.index("check_short()\n") < launch.rindex("start_due()")
    assert "< SCREEN_EPOCHS" in launch
    # A live run whose log stopped growing raises; a dead one does not block the next tag.
    assert "quiet > limit" in monitor and "is stuck" in monitor
    assert "if tag in DEAD:\n            continue" in launch
    # No claim about the teammate's flags.
    assert "at the same flags" not in "".join("".join(c["source"]) for c in _notebook()["cells"])
