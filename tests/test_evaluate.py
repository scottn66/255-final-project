"""Checkpoint scoring on tiny synthetic cubes. No real SpectralWaste files."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw.data import CLASS_NAMES
from sw.evaluate import cfg_from_checkpoint, evaluate_checkpoint, main as evaluate_main
from sw.train import main as train_main

MODEL_FLAGS = {
    "fused": ["--model", "transformer"],
    "control": [
        "--model", "transformer", "--tokenizer", "collapsed",
        "--depth-spectral", "0", "--depth-spatial", "8",
    ],
    "cnn": ["--model", "cnn"],
}
VARIANT_FLAGS = {
    "plain": [],
    # The recipe that goes with --standardize (docs/INPUT-BLIND.md).
    "standardize": ["--standardize", "--pe-init-std", "0.02"],
    "downsample2": ["--downsample", "2"],
}


N_VAL = 3
N_TEST = 5


def _write_split(root, split, n_images, seed, shape=(224, 32, 32)):
    """Cubes whose bands depend on the class, plus noise, so predictions vary by pixel.

    The masks are 8-pixel blocks, shifted by 3 rows and 1 column so the
    block edges do not sit on the downsample stride. Filament (class 5) is
    never drawn, so it is absent from the ground truth of every split.
    """
    rng = np.random.default_rng(seed)
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    channels, height, width = shape
    spectra = rng.uniform(0.1, 0.9, size=(7, channels))
    for index in range(n_images):
        blocks = rng.choice([0, 1, 2, 3, 4, 6], size=(height // 8 + 1, width // 8 + 1))
        mask = np.kron(blocks, np.ones((8, 8), dtype=np.int64)).astype(np.uint8)
        mask = np.ascontiguousarray(mask[3 : 3 + height, 1 : 1 + width])
        cube = spectra[mask].transpose(2, 0, 1) + rng.normal(0.0, 0.05, size=shape)
        cube = (np.clip(cube, 0.0, 1.0) * 65535).astype(np.uint16)
        tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", cube, photometric="minisblack")
        Image.fromarray(mask, mode="L").save(labels / f"{split}_{index:02d}.png")


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    # These cubes are tiny. One thread is faster than several, most of all on a busy machine.
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    # The helpers in the other test files write one test image. This one writes five,
    # so test and val differ in size and in class counts.
    root = tmp_path_factory.mktemp("sw")
    _write_split(root, "train", 4, seed=0)
    _write_split(root, "val", N_VAL, seed=1)
    _write_split(root, "test", N_TEST, seed=2)
    return root


def _class_counts(root, split, stride=1):
    """Pixels per class in the masks of ``split``, read from the files, at ``[::stride]``."""
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for path in sorted((root / "labels_hyper_lt" / split).glob("*.png")):
        mask = np.asarray(Image.open(path))[::stride, ::stride]
        counts += np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
    return counts.tolist()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, data_root):
    """Train each (model, variant) once per module and hand back its run folder."""
    done = {}

    def get(kind, variant, extra=()):
        key = (kind, variant, tuple(extra))
        if key not in done:
            out = tmp_path_factory.mktemp(f"run_{kind}_{variant}")
            train_main(
                MODEL_FLAGS[kind]
                + VARIANT_FLAGS[variant]
                + list(extra)
                + [
                    "--data-root", str(data_root), "--out-dir", str(out), "--device", "cpu",
                    "--epochs", "12", "--no-cosine", "--batch-size", "2", "--num-workers", "0",
                    "--seed", "0",
                ]
            )
            done[key] = out
        return done[key]

    return get


def _best_row(run, epoch):
    with (run / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return next(row for row in rows if row["epoch"] == epoch)


def _evaluate(checkpoint, split, data_root, **kwargs):
    # The synthetic splits are not the official size. A real test score needs the flag.
    if split == "test" and "max_images" not in kwargs:
        kwargs.setdefault("allow_partial_split", True)
    return evaluate_checkpoint(checkpoint, split, data_root=str(data_root), device="cpu", **kwargs)


def _copy_checkpoint(run, folder):
    folder.mkdir()
    shutil.copy(run / "best.pt", folder / "best.pt")
    return folder / "best.pt"


@pytest.mark.parametrize("variant", list(VARIANT_FLAGS))
@pytest.mark.parametrize("kind", list(MODEL_FLAGS))
def test_val_reproduces_the_training_log(kind, variant, runs, data_root):
    run = runs(kind, variant)
    result = _evaluate(run / "best.pt", "val", data_root)
    row = _best_row(run, result["checkpoint_epoch"])

    assert result["fg_miou"] == pytest.approx(row["val_fg_miou"], abs=1e-12)
    assert result["checkpoint_val_fg_miou"] == pytest.approx(row["val_fg_miou"], abs=1e-12)
    for name in CLASS_NAMES:
        logged = row["per_class_iou"][name]
        if logged is None:
            assert result["per_class_iou"][name] is None
        else:
            assert result["per_class_iou"][name] == pytest.approx(logged, abs=1e-12)

    assert result["val_check"]["status"] == "match"
    assert result["val_check"]["abs_diff"] <= 1e-12
    assert result["n_images"] == N_VAL
    # Rows of the matrix are the target. They must be the val masks at the run's stride.
    stride = 2 if variant == "downsample2" else 1
    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.sum(axis=1).tolist() == _class_counts(data_root, "val", stride)
    assert matrix.sum() == N_VAL * (32 // stride) ** 2
    assert 0.0 < result["pixel_accuracy"] < 1.0
    assert result["pixel_accuracy"] == pytest.approx(matrix.trace() / matrix.sum(), abs=1e-12)
    assert result["official_split"] is False
    assert any("NOT THE OFFICIAL SPLIT" in note for note in result["notes"])
    assert result["standardize"] is (variant == "standardize")
    assert result["model"] == ("MiniNetSeg" if kind == "cnn" else "BandTransformer")
    assert result["tokenizer"] == {"fused": "band", "control": "collapsed", "cnn": None}[kind]

    saved = json.loads((run / "eval_val.json").read_text(encoding="utf-8"))
    assert saved["fg_miou"] == result["fg_miou"]
    assert saved["per_class_iou"] == result["per_class_iou"]
    assert saved["cfg"] == torch.load(run / "best.pt", weights_only=False)["cfg"]
    assert len(saved["checkpoint_sha256"]) == 64
    # Validation is not the guarded split: scoring it again just rewrites the file.
    _evaluate(run / "best.pt", "val", data_root, batch_size=1)


def _shifted_stats(source, target):
    stats = torch.load(source, weights_only=False)
    stats["mean"] = stats["mean"] + 0.5
    torch.save(stats, target)
    return target


def test_the_statistics_file_matters(runs, data_root, capsys):
    """Other statistics give other predictions, so the match above tests the file."""
    run = runs("cnn", "standardize")
    result = _evaluate(run / "best.pt", "val", data_root)
    assert (np.asarray(result["confusion_matrix"]).sum(axis=0) > 0).sum() >= 2
    assert "val check: MATCH" in capsys.readouterr().out

    wrong = _shifted_stats(run / "band_stats.pt", run / "wrong_stats.pt")
    capped = _evaluate(run / "best.pt", "val", data_root, stats_path=wrong, max_images=N_VAL)
    assert capped["confusion_matrix"] != result["confusion_matrix"]
    assert capped["stats_path"] == str(wrong)
    assert capped["val_check"]["status"] == "not run"

    # An uncapped val run says so, loudly.
    other = _evaluate(run / "best.pt", "val", data_root, stats_path=wrong)
    assert other["val_check"]["status"] == "mismatch"
    assert "val check: MISMATCH" in capsys.readouterr().out

    # Test is not scored with a setup that does not reproduce val.
    with pytest.raises(RuntimeError, match="val check MISMATCH.*Test was not scored"):
        _evaluate(run / "best.pt", "test", data_root, stats_path=wrong)
    assert not (run / "eval_test.json").exists()
    allowed = _evaluate(run / "best.pt", "test", data_root, stats_path=wrong, allow_val_mismatch=True)
    assert allowed["val_check"]["status"] == "mismatch"
    saved = json.loads((run / "eval_test.json").read_text(encoding="utf-8"))
    assert any("val check MISMATCH" in note for note in saved["notes"])
    # Leave the run folder as a later test expects it.
    (run / "eval_test.json").unlink()
    _evaluate(run / "best.pt", "val", data_root)


def test_missing_statistics_file_is_an_error(runs, data_root, tmp_path):
    run = runs("fused", "standardize")
    checkpoint = _copy_checkpoint(run, tmp_path / "moved")
    with pytest.raises(FileNotFoundError, match="band statistics file is missing"):
        _evaluate(checkpoint, "val", data_root)
    assert not (checkpoint.parent / "band_stats.pt").exists()
    assert not (checkpoint.parent / "eval_val.json").exists()

    # The run's own file, named by hand, gives the logged score again.
    result = _evaluate(checkpoint, "val", data_root, stats_path=run / "band_stats.pt")
    row = _best_row(run, result["checkpoint_epoch"])
    assert result["fg_miou"] == pytest.approx(row["val_fg_miou"], abs=1e-12)


def test_a_run_with_its_own_stats_path_uses_only_that_file(runs, data_root, tmp_path):
    shared = tmp_path / "shared_stats.pt"
    run = runs("cnn", "standardize", extra=("--stats-path", str(shared)))
    assert shared.is_file() and not (run / "band_stats.pt").exists()
    row = _best_row(run, torch.load(run / "best.pt", weights_only=False)["epoch"])

    # A band_stats.pt next to the checkpoint is not the run's file. It is not used.
    _shifted_stats(shared, run / "band_stats.pt")
    result = _evaluate(run / "best.pt", "val", data_root)
    assert result["stats_path"] == str(shared)
    assert result["fg_miou"] == pytest.approx(row["val_fg_miou"], abs=1e-12)
    assert result["val_check"]["status"] == "match"

    # With the run's file gone the command stops. It does not fall back to the neighbour.
    (run / "eval_val.json").unlink()
    shared.rename(tmp_path / "elsewhere.pt")
    with pytest.raises(FileNotFoundError, match="band statistics file is missing"):
        _evaluate(run / "best.pt", "val", data_root)
    assert not (run / "eval_val.json").exists()
    result = _evaluate(run / "best.pt", "val", data_root, stats_path=tmp_path / "elsewhere.pt")
    assert result["val_check"]["status"] == "match"


def test_test_split_is_written_once(runs, data_root, tmp_path, capsys, monkeypatch):
    run = runs("fused", "plain")
    checkpoint = _copy_checkpoint(run, tmp_path / "final")
    out = checkpoint.parent / "eval_test.json"

    capped = _evaluate(checkpoint, "test", data_root, max_images=2)
    assert capped["capped"] and capped["n_images"] == 2 and capped["written"] is None
    assert not out.exists()
    assert "not a score" in capsys.readouterr().out
    # A cap that covers the split, or a large one, would be a free look at the test score.
    for too_many in (N_TEST, 9, 99):
        with pytest.raises(ValueError, match="on test is refused"):
            _evaluate(checkpoint, "test", data_root, max_images=too_many)

    with monkeypatch.context() as patch:
        # The limit of 8 images, shrunk to 1 so this small split can show it.
        patch.setattr("sw.evaluate.TEST_CAP", 1)
        with pytest.raises(ValueError, match="at most 1 images"):
            _evaluate(checkpoint, "test", data_root, max_images=2)
    assert not out.exists()

    # Five images are not the official 171, so the plain command refuses.
    command = ["--checkpoint", str(checkpoint), "--split", "test", "--data-root", str(data_root),
               "--device", "cpu", "--num-workers", "0"]
    with pytest.raises(ValueError, match="official split has 171"):
        evaluate_main(command)
    assert not out.exists()
    capsys.readouterr()

    first = evaluate_main(command + ["--allow-partial-split"])
    printed = capsys.readouterr().out
    assert "test FG mIoU" in printed and "left out of the FG mean" in printed
    assert "val check: MATCH" in printed and "NOT THE OFFICIAL SPLIT" in printed
    assert first["n_images"] == N_TEST and first["split"] == "test"
    # The targets are the test masks, not another split's.
    matrix = np.asarray(first["confusion_matrix"])
    assert matrix.sum(axis=1).tolist() == _class_counts(data_root, "test")
    assert _class_counts(data_root, "test") != _class_counts(data_root, "val")
    assert first["val_check"]["status"] == "match"
    assert first["val_check"]["val_fg_miou"] == pytest.approx(first["checkpoint_val_fg_miou"], abs=1e-12)
    before = out.read_text(encoding="utf-8")
    saved = json.loads(before)
    assert saved["fg_miou"] == first["fg_miou"]
    assert saved["official_split"] is False and saved["notes"] == first["notes"]
    assert not any("--force" in note for note in saved["notes"])
    # A test run scores val to check itself, but writes only the test file.
    assert not (checkpoint.parent / "eval_val.json").exists()

    with pytest.raises(FileExistsError, match="already scored on test.*score test once"):
        _evaluate(checkpoint, "test", data_root)
    assert out.read_text(encoding="utf-8") == before
    # A capped look is still allowed, and still writes nothing.
    _evaluate(checkpoint, "test", data_root, max_images=1)
    assert out.read_text(encoding="utf-8") == before

    again = _evaluate(checkpoint, "test", data_root, force=True)
    assert again["fg_miou"] == first["fg_miou"]
    assert any("--force" in note for note in again["notes"])
    # The file on disk is the new one.
    assert json.loads(out.read_text(encoding="utf-8"))["notes"] == again["notes"]


def test_a_test_file_from_another_checkpoint_is_called_stale(runs, data_root, tmp_path):
    run = runs("fused", "plain")
    checkpoint = _copy_checkpoint(run, tmp_path / "retrained")
    out = checkpoint.parent / "eval_test.json"
    first = _evaluate(checkpoint, "test", data_root)

    # A retrain into the same folder leaves a new best.pt next to the old file.
    ckpt = torch.load(checkpoint, weights_only=False)
    ckpt["class_weights"] = list(ckpt["class_weights"]) + [0.0]
    torch.save(ckpt, checkpoint)
    before = out.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="belongs to another checkpoint.*stale") as raised:
        _evaluate(checkpoint, "test", data_root)
    assert first["checkpoint_sha256"][:12] in str(raised.value)
    assert "already scored on test" not in str(raised.value)
    assert out.read_text(encoding="utf-8") == before

    replaced = _evaluate(checkpoint, "test", data_root, force=True)
    assert replaced["checkpoint_sha256"] != first["checkpoint_sha256"]
    assert any("another checkpoint" in note for note in replaced["notes"])
    assert json.loads(out.read_text(encoding="utf-8"))["checkpoint_sha256"] == replaced["checkpoint_sha256"]


def test_last_pt_has_its_own_file_and_no_score_of_its_own(runs, data_root, tmp_path, capsys):
    run = runs("control", "plain")
    folder = tmp_path / "both"
    best = _copy_checkpoint(run, folder)
    last = folder / "last.pt"
    shutil.copy(run / "last.pt", last)
    ckpt = torch.load(last, weights_only=False)

    _evaluate(best, "test", data_root)
    capsys.readouterr()
    # best.pt was scored. last.pt is another checkpoint and is not blocked by that.
    result = _evaluate(last, "test", data_root)
    printed = capsys.readouterr().out
    assert result["written"] == str(folder / "eval_test_last.json")
    assert json.loads((folder / "eval_test.json").read_text(encoding="utf-8"))["checkpoint"] == str(best)
    assert result["checkpoint_val_fg_miou"] is None
    assert result["best_val_fg_miou"] == ckpt["best_val_fg_miou"]
    assert result["best_epoch"] == ckpt["best_epoch"]
    assert result["val_check"]["status"] == "skipped"
    assert f"best val so far {ckpt['best_val_fg_miou']:.4f} (epoch {ckpt['best_epoch']})" in printed
    assert "recorded val FG mIoU" not in printed and "val check: skipped" in printed

    # On val, last.pt gives the log row of its own epoch, which need not be the best one.
    on_val = _evaluate(last, "val", data_root)
    assert on_val["written"] == str(folder / "eval_val_last.json")
    assert on_val["fg_miou"] == pytest.approx(_best_row(run, ckpt["epoch"])["val_fg_miou"], abs=1e-12)


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path, data_root):
    torch.save({"weights": 1}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not a sw.train checkpoint"):
        _evaluate(tmp_path / "bad.pt", "val", data_root)


def test_cfg_without_tokenizer_is_the_band_model(runs, data_root, tmp_path):
    run = runs("fused", "plain")
    ckpt = torch.load(run / "best.pt", weights_only=False)
    del ckpt["cfg"]["tokenizer"]
    ckpt["cfg"]["some_later_flag"] = 1
    cfg, unknown = cfg_from_checkpoint(ckpt["cfg"])
    assert cfg.tokenizer == "band" and unknown == ["some_later_flag"]
    (tmp_path / "old").mkdir()
    torch.save(ckpt, tmp_path / "old" / "best.pt")

    result = _evaluate(tmp_path / "old" / "best.pt", "val", data_root)
    assert result["tokenizer"] == "band"
    assert result["fg_miou"] == pytest.approx(ckpt["val_fg_miou"], abs=1e-12)
    assert any("no tokenizer entry" in note for note in result["notes"])


def test_strict_loading_refuses_a_mismatched_checkpoint(runs, data_root, tmp_path):
    # The control's weights under the fused model's cfg.
    ckpt = torch.load(runs("control", "plain") / "best.pt", weights_only=False)
    ckpt["cfg"] = torch.load(runs("fused", "plain") / "best.pt", weights_only=False)["cfg"]
    (tmp_path / "mixed").mkdir()
    torch.save(ckpt, tmp_path / "mixed" / "best.pt")
    with pytest.raises(RuntimeError, match="loaded strictly"):
        _evaluate(tmp_path / "mixed" / "best.pt", "val", data_root)
    assert not (tmp_path / "mixed" / "eval_val.json").exists()

    # One missing tensor is refused too.
    ckpt = torch.load(runs("cnn", "plain") / "best.pt", weights_only=False)
    ckpt["model"].pop(next(iter(ckpt["model"])))
    torch.save(ckpt, tmp_path / "mixed" / "short.pt")
    with pytest.raises(RuntimeError, match="loaded strictly"):
        _evaluate(tmp_path / "mixed" / "short.pt", "val", data_root)
