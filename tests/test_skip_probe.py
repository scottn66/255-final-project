"""``sw.skip_probe``: a ``--head-skip`` checkpoint scored with each head path switched off.

The runs here are a small transformer on synthetic cubes with thin lines.
They check the probe's wiring. They say nothing about SpectralWaste.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw import skip_probe
from sw.data import CLASS_NAMES
from sw.metrics import ConfusionMeter
from sw.models.transformer import BandTransformer
from sw.train import main as train_main

THIN = ("video_tape", "filament")


def write_thin_split(root, split, n_images, seed, size=32, bands=224):
    """Cubes where every class has its own spectrum, and two classes are thin lines.

    Background and the four large classes fill 8 by 8 blocks. Then
    ``video_tape`` is drawn as lines one pixel wide and ``filament`` as
    lines two pixels wide, over the blocks. Each pixel's spectrum is its
    class's shape plus a little noise, so one pixel's spectrum is enough
    to name its class.
    """
    rng = np.random.default_rng(seed)
    shapes = np.random.default_rng(123).random((7, bands))
    large = [CLASS_NAMES.index(name) for name in CLASS_NAMES if name not in THIN]
    tape, filament = CLASS_NAMES.index("video_tape"), CLASS_NAMES.index("filament")
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    blocks = size // 8
    for index in range(n_images):
        mask = np.kron(rng.choice(large, size=(blocks, blocks)), np.ones((8, 8), dtype=np.int64))
        for _ in range(2):
            mask[int(rng.integers(0, size)), :] = tape  # a row, one pixel wide
            mask[:, int(rng.integers(0, size))] = tape  # a column, one pixel wide
        start = int(rng.integers(0, size - 1))
        mask[start : start + 2, :] = filament  # a row, two pixels wide
        cube = shapes[mask].transpose(2, 0, 1) + 0.05 * rng.random((bands, size, size))
        tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", (cube / 1.1 * 65535).astype(np.uint16), photometric="minisblack")
        Image.fromarray(mask.astype(np.uint8), mode="L").save(labels / f"{split}_{index:02d}.png")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One short ``--head-skip --standardize`` run and its data root."""
    base = tmp_path_factory.mktemp("probe")
    root = base / "data"
    write_thin_split(root, "train", 4, seed=0)
    write_thin_split(root, "val", 3, seed=1)
    out = base / "run"
    flags = [
        "--model", "transformer", "--depth-spectral", "1", "--depth-spatial", "1", "--head-skip", "--standardize",
        "--no-cosine", "--lr", "3e-3", "--epochs", "12", "--batch-size", "2", "--num-workers", "0",
        "--data-root", str(root), "--out-dir", str(out), "--device", "cpu", "--seed", "0",
    ]
    result = train_main(flags)
    return {"root": root, "out": out, "result": result, "flags": flags}


def _val_batches(run):
    """The validation cubes, standardized with the run's own statistics, and their masks."""
    from sw.data import SpectralWasteDataset
    from sw.standardize import BandStandardizer, load_band_stats

    standardizer = BandStandardizer(*load_band_stats(run["out"] / "band_stats.pt"))
    base = SpectralWasteDataset(str(run["root"]), split="val", random_flip=False)
    return [(standardizer(cube)[None], mask[None]) for cube, mask in base]


def _model(run):
    saved = torch.load(run["out"] / "best.pt", map_location="cpu", weights_only=False)
    model = BandTransformer(1, 1, head_skip=True).eval()
    model.load_state_dict(saved["model"])
    return model, saved


def test_normal_row_reproduces_the_logged_score(run, capsys):
    result = skip_probe.probe(run["out"] / "best.pt", device="cpu")
    saved = torch.load(run["out"] / "best.pt", map_location="cpu", weights_only=False)
    assert list(result["rows"]) == ["normal", "skip_zeroed", "tokens_zeroed"]
    normal = result["rows"]["normal"]
    # Same device, same code: the same numbers, not only close ones.
    assert normal["val_fg_miou"] == saved["val_fg_miou"] == run["result"]["best_val_fg_miou"]
    assert normal["per_class_iou"] == saved["per_class_iou"]
    assert result["logged_val_fg_miou"] == saved["val_fg_miou"] and result["difference"] == 0.0
    assert result["epoch"] == saved["epoch"] == run["result"]["best_epoch"]
    assert result["val_images"] == 3
    assert result["param_count"] == run["result"]["param_count"]
    assert result["stats_path"] == str(run["out"] / "band_stats.pt")
    for row in result["rows"].values():
        assert set(row["per_class_iou"]) == set(CLASS_NAMES)
    # The batch size and the device option do not change a score.
    again = skip_probe.probe(run["out"] / "best.pt", device="cpu", batch_size=1, num_workers=0)
    assert again["rows"] == result["rows"]


def test_ablated_rows_match_a_computation_written_by_hand(run):
    """The two ablations, recomputed without the probe's hook.

    Skip zeroed: the skip conv's weights are set to zero, and GELU(0) is 0.
    Tokens zeroed: the head's last layers are applied by hand to zeros
    joined with the skip of the image.
    """
    result = skip_probe.probe(run["out"] / "best.pt", device="cpu")
    batches = _val_batches(run)

    model, _ = _model(run)
    with torch.no_grad():
        model.head.skip[0].weight.zero_()
        model.head.skip[0].bias.zero_()
        meter = ConfusionMeter()
        for cube, mask in batches:
            meter.update(model(cube), mask)
    assert meter.fg_miou() == pytest.approx(result["rows"]["skip_zeroed"]["val_fg_miou"], abs=1e-12)

    model, _ = _model(run)
    head = model.head
    meter = ConfusionMeter()
    with torch.no_grad():
        for cube, mask in batches:
            joined = torch.cat([torch.zeros(1, head.hidden, *cube.shape[-2:]), head.skip(cube)], dim=1)
            meter.update(head.classifier(head.fuse(joined)), mask)
    assert meter.fg_miou() == pytest.approx(result["rows"]["tokens_zeroed"]["val_fg_miou"], abs=1e-12)
    values = [meter.per_class_iou()[CLASS_NAMES.index(name)] for name in CLASS_NAMES]
    for name, value in zip(CLASS_NAMES, values):
        logged = result["rows"]["tokens_zeroed"]["per_class_iou"][name]
        assert logged == pytest.approx(value, abs=1e-12)

    # The three rows are three different measurements on this run.
    scores = [result["rows"][mode]["val_fg_miou"] for mode in skip_probe.MODES]
    assert len(set(scores)) == 3


def test_ablate_head_path_changes_one_path_and_restores_the_model(run):
    model, _ = _model(run)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(2, 224, 30, 40)
    seen = []
    handle = model.head.fuse.register_forward_hook(lambda module, args, output: seen.append(args[0]))
    with torch.no_grad():
        normal = model(x)
        with skip_probe.ablate_head_path(model, "normal") as same:
            assert same is model and torch.equal(model(x), normal)
        with skip_probe.ablate_head_path(model, "skip_zeroed"):
            no_skip = model(x)
        with skip_probe.ablate_head_path(model, "tokens_zeroed"):
            no_tokens = model(x)
        with skip_probe.ablate_head_path(model, "tokens_zeroed", fill="mean"):
            mean_tokens = model(x)
        assert torch.equal(model(x), normal)  # the hook is gone
    handle.remove()
    full, a, b, c, _ = seen[1:]  # the first call was the plain forward
    assert full.shape == (2, 48, 30, 40)
    assert torch.equal(a[:, :32], full[:, :32]) and not a[:, 32:].any()  # skip zeroed, decoder map kept
    assert torch.equal(b[:, 32:], full[:, 32:]) and not b[:, :32].any()  # decoder map zeroed, skip kept
    assert torch.equal(c[:, 32:], full[:, 32:])
    assert torch.allclose(c[:, :32], full[:, :32].mean(dim=(2, 3), keepdim=True).expand(-1, -1, 30, 40))
    for out in (no_skip, no_tokens, mean_tokens):
        assert out.shape == normal.shape and not torch.equal(out, normal)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert not model.head.fuse._forward_pre_hooks

    # The hook is removed when the block raises, too.
    with pytest.raises(ZeroDivisionError):
        with skip_probe.ablate_head_path(model, "tokens_zeroed"):
            1 / 0
    assert not model.head.fuse._forward_pre_hooks
    with pytest.raises(ValueError, match="mode must be"):
        with skip_probe.ablate_head_path(model, "no_skip"):
            pass
    with pytest.raises(ValueError, match="fill must be"):
        with skip_probe.ablate_head_path(model, "skip_zeroed", fill="median"):
            pass
    with pytest.raises(ValueError, match="no head skip"):
        with skip_probe.ablate_head_path(BandTransformer(1, 1), "skip_zeroed"):
            pass


def test_with_tokens_zeroed_a_pixel_sees_only_its_own_spectrum(run):
    model, _ = _model(run)
    model = model.double()
    x = torch.randn(1, 224, 30, 40, dtype=torch.float64)
    changed = x.clone()
    changed[0, :, 11, 20] += 1.0
    with torch.no_grad():
        with skip_probe.ablate_head_path(model, "tokens_zeroed"):
            diff = (model(changed) - model(x)).abs().sum(dim=1)[0]
        assert diff[11, 20] > 0
        diff[11, 20] = 0
        assert not diff.any()  # no token, no neighbour
        # Without the ablation the same change reaches other pixels through the tokens.
        assert ((model(changed) - model(x)).abs().sum(dim=1)[0] > 0).sum() > 1


def test_probe_uses_the_runs_own_band_statistics_and_never_computes_them(run, tmp_path):
    import shutil

    copy = tmp_path / "moved"
    shutil.copytree(run["out"], copy)
    # A moved run folder still finds the statistics beside the checkpoint.
    moved = skip_probe.probe(copy / "best.pt", device="cpu")
    assert moved["stats_path"] == str(copy / "band_stats.pt")
    assert moved["rows"] == skip_probe.probe(run["out"] / "best.pt", device="cpu")["rows"]

    (copy / "band_stats.pt").rename(copy / "elsewhere.pt")
    with pytest.raises(FileNotFoundError, match="never computes new ones"):
        skip_probe.probe(copy / "best.pt", device="cpu")
    assert not (copy / "band_stats.pt").exists()  # nothing was computed or written
    assert skip_probe.probe(copy / "best.pt", device="cpu", stats_path=copy / "elsewhere.pt")["rows"] == moved["rows"]

    # Other statistics give another input. The normal row no longer matches the log, and the probe stops.
    wrong = torch.load(copy / "elsewhere.pt", map_location="cpu", weights_only=False)
    wrong["mean"] = wrong["mean"] + 3.0 * wrong["std"]
    torch.save(wrong, copy / "wrong.pt")
    with pytest.raises(RuntimeError, match="does not reproduce the logged score"):
        skip_probe.probe(copy / "best.pt", device="cpu", stats_path=copy / "wrong.pt")


def test_probe_stops_when_the_logged_score_is_not_reproduced(run, tmp_path):
    saved = torch.load(run["out"] / "best.pt", map_location="cpu", weights_only=False)
    (tmp_path / "band_stats.pt").write_bytes((run["out"] / "band_stats.pt").read_bytes())

    saved["val_fg_miou"] += 0.01
    torch.save(saved, tmp_path / "best.pt")
    with pytest.raises(RuntimeError, match="does not reproduce the logged score") as raised:
        skip_probe.probe(tmp_path / "best.pt", device="cpu")
    assert "No ablation row is reported" in str(raised.value)
    assert skip_probe.probe(tmp_path / "best.pt", device="cpu", tolerance=0.02)["difference"] == pytest.approx(-0.01)

    # A per-class value that is off stops it too, even when the mean is right.
    saved["val_fg_miou"] -= 0.01
    saved["per_class_iou"] = {**saved["per_class_iou"], "film": saved["per_class_iou"]["film"] - 0.05}
    torch.save(saved, tmp_path / "best.pt")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        skip_probe.probe(tmp_path / "best.pt", device="cpu")


def test_last_pt_is_checked_against_the_metrics_log(run, tmp_path):
    last = torch.load(run["out"] / "last.pt", map_location="cpu", weights_only=False)
    assert "val_fg_miou" not in last
    rows = [json.loads(line) for line in (run["out"] / "metrics.jsonl").read_text().splitlines()]
    result = skip_probe.probe(run["out"] / "last.pt", device="cpu")
    assert result["epoch"] == 12
    assert result["logged_val_fg_miou"] == rows[-1]["val_fg_miou"]
    assert result["rows"]["normal"]["val_fg_miou"] == rows[-1]["val_fg_miou"]

    # With no log beside it there is nothing to check against, and the report says so.
    for name in ("last.pt", "band_stats.pt"):
        (tmp_path / name).write_bytes((run["out"] / name).read_bytes())
    alone = skip_probe.probe(tmp_path / "last.pt", device="cpu")
    assert alone["logged_val_fg_miou"] is None and alone["difference"] is None
    assert alone["rows"] == result["rows"]
    assert "not checked against the log" in skip_probe.format_probe(alone)


def test_probe_refuses_a_checkpoint_without_the_skip(run, tmp_path):
    out = tmp_path / "plain"
    flags = [flag for flag in run["flags"] if flag != "--head-skip"]
    flags[flags.index("--out-dir") + 1] = str(out)
    flags[flags.index("--epochs") + 1] = "1"
    train_main(flags)
    with pytest.raises(ValueError, match="trained without --head-skip"):
        skip_probe.probe(out / "best.pt", device="cpu")
    torch.save({"epoch": 1}, tmp_path / "other.pt")
    with pytest.raises(KeyError, match="missing 'model' or 'cfg'"):
        skip_probe.probe(tmp_path / "other.pt", device="cpu")
    with pytest.raises(ValueError, match="without --standardize"):
        # --stats-path for a run that did not standardize is a mistake, not something to ignore.
        cfg_off = torch.load(run["out"] / "best.pt", map_location="cpu", weights_only=False)
        cfg_off["cfg"]["standardize"] = False
        torch.save(cfg_off, tmp_path / "nostd.pt")
        skip_probe.probe(tmp_path / "nostd.pt", device="cpu", stats_path=run["out"] / "band_stats.pt")


def test_cli_prints_three_rows_and_says_it_is_not_a_retrained_model(run, capsys):
    capsys.readouterr()
    result = skip_probe.main(["--checkpoint", str(run["out"] / "best.pt"), "--device", "cpu", "--data-root", str(run["root"])])
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[0] == f"skip probe  {run['out'] / 'best.pt'}  (epoch {result['epoch']})"
    assert f"parameters {result['param_count']:,}  head skip 224->16" in lines[1]
    assert "val images 3  downsample 1  crop None  standardize on" in lines[2]
    assert "the normal row reproduces it (difference +0.0e+00)" in lines[3]
    table = [line for line in lines if line.startswith(("normal", "skip zeroed", "tokens zeroed"))]
    assert [line.split("  ")[0] for line in table] == [
        "normal", "skip zeroed (transformer path only)", "tokens zeroed (skip path only)",
    ]
    for line, mode in zip(table, skip_probe.MODES):
        assert f"{result['rows'][mode]['val_fg_miou']:.3f}" in line
        assert len(line.split()[-7:]) == 7  # one value per class
    assert "ablation at test time, not a retrained model" in printed
    assert "set to 0 at the input of head.fuse" in printed
    assert "Limited run" not in printed  # this run used every validation image

    skip_probe.main(["--checkpoint", str(run["out"] / "best.pt"), "--device", "cpu", "--fill", "mean"])
    printed = capsys.readouterr().out
    assert "tokens set to mean (skip path only)" in printed and "zeroed" not in printed
    assert "still carries what the tokens saw" in printed
    assert "not a retrained model" in printed


def test_probe_follows_the_runs_downsample_and_image_limit(run, tmp_path, capsys):
    out = tmp_path / "half"
    flags = list(run["flags"])
    flags[flags.index("--out-dir") + 1] = str(out)
    flags[flags.index("--epochs") + 1] = "2"
    train_main([*flags, "--downsample", "2", "--max-val-images", "2", "--patch", "4"])
    capsys.readouterr()
    result = skip_probe.main(["--checkpoint", str(out / "best.pt"), "--device", "cpu"])
    printed = capsys.readouterr().out
    assert result["val_images"] == 2 and result["difference"] == 0.0
    assert "val images 2  downsample 2  crop None" in printed
    assert "Limited run: only the first 2 validation images" in printed
