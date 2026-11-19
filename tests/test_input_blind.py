"""Input-blind gates: standardise, PE scale, prior bias, diagnostic ratios."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sw.data import HYPER_LT_TRAIN_PIXEL_COUNTS
from sw.models.transformer import BandTransformer, init_seg_head_prior_bias
from sw.pe import IndexBandPE
from sw.standardize import (
    BandStandardizer,
    compute_band_stats,
    load_band_stats,
    load_or_compute_band_stats,
    save_band_stats,
)
from sw.train import TrainConfig, _build_parser, build_model, config_from_args, set_seed


REPO = Path(__file__).resolve().parents[1]


def _load_diag_module():
    path = REPO / "scripts" / "diag_input_blind.py"
    spec = importlib.util.spec_from_file_location("diag_input_blind", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ToyCubes(torch.utils.data.Dataset):
    """Tiny (C,H,W) cubes with a known per-band offset."""

    def __init__(self, n=4, c=8, h=4, w=4, offset=0.23, spatial_std=0.14, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cubes = []
        for _ in range(n):
            noise = torch.randn(c, h, w, generator=g)
            self.cubes.append(offset + spatial_std * noise)

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, index):
        return self.cubes[index], torch.zeros(self.cubes[index].shape[-2:], dtype=torch.int64)


def test_compute_band_stats_train_only_and_standardize_whitening():
    ds = _ToyCubes()
    mean, std = compute_band_stats(ds, split="train")
    assert mean.shape == (8,)
    assert std.shape == (8,)
    with pytest.raises(ValueError, match="train split only"):
        compute_band_stats(ds, split="val")
    with pytest.raises(ValueError, match="train split only"):
        compute_band_stats(ds, split="test")

    stdizer = BandStandardizer(mean, std)
    stacked = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
    out = stdizer(stacked)
    # Under the same stats, per-band mean ~0 and std ~1.
    out_mean = out.mean(dim=(0, 2, 3))
    out_std = out.std(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5)
    assert torch.allclose(out_std, torch.ones_like(out_std), atol=1e-4)


def test_save_load_rejects_non_train_label(tmp_path):
    mean = torch.zeros(3)
    std = torch.ones(3)
    path = tmp_path / "band_stats.pt"
    save_band_stats(path, mean, std, split="train")
    m2, s2 = load_band_stats(path)
    assert torch.equal(m2, mean)
    assert torch.equal(s2, std)
    with pytest.raises(ValueError, match="refusing"):
        save_band_stats(path, mean, std, split="val")
    # Corrupt a file to claim val and ensure load raises.
    bad = tmp_path / "bad.pt"
    torch.save({"mean": mean, "std": std, "split": "val"}, bad)
    with pytest.raises(ValueError, match="train split only"):
        load_band_stats(bad)


def test_load_or_compute_caches(tmp_path):
    ds = _ToyCubes(n=2, c=4)
    path = tmp_path / "stats.pt"
    m1, s1 = load_or_compute_band_stats(path, ds, split="train")
    assert path.is_file()
    # Second call must hit the cache even if the dataset would differ.
    other = _ToyCubes(n=2, c=4, offset=0.9, seed=99)
    m2, s2 = load_or_compute_band_stats(path, other, split="train")
    assert torch.equal(m1, m2)
    assert torch.equal(s1, s2)
    with pytest.raises(ValueError, match="train split only"):
        load_or_compute_band_stats(path, ds, split="val")


def test_pe_init_std_scales_embedding():
    torch.manual_seed(0)
    pe_full = IndexBandPE(14, 64, init_std=1.0)
    torch.manual_seed(0)
    pe_small = IndexBandPE(14, 64, init_std=0.02)
    # Same default draw, then x0.02 — std of weights scales by ~0.02.
    ratio = pe_small.table.weight.std().item() / pe_full.table.weight.std().item()
    assert ratio == pytest.approx(0.02, rel=1e-5, abs=1e-6)

    torch.manual_seed(1)
    a = BandTransformer(1, 1, pe_init_std=1.0)
    torch.manual_seed(1)
    b = BandTransformer(1, 1, pe_init_std=0.02)
    w_a = a.tokenizer.group_emb.table.weight
    w_b = b.tokenizer.group_emb.table.weight
    assert torch.allclose(w_b, w_a * 0.02)


def test_head_prior_bias_matches_log_priors():
    model = BandTransformer(0, 0)
    # Conv2d bias defaults to a small uniform draw, not zeros — still not log priors.
    before = model.head.classifier.bias.detach().clone()
    bias = init_seg_head_prior_bias(model.head, HYPER_LT_TRAIN_PIXEL_COUNTS)
    counts = torch.tensor(HYPER_LT_TRAIN_PIXEL_COUNTS, dtype=torch.float32)
    expected = (counts / counts.sum()).log()
    assert torch.allclose(bias, expected)
    assert torch.allclose(model.head.classifier.bias.cpu(), expected)
    assert not torch.allclose(before.cpu(), expected)

    cfg = config_from_args(
        _build_parser().parse_args(
            ["--model", "transformer", "--head-prior-bias", "--device", "cpu"]
        )
    )
    assert cfg.head_prior_bias is True
    set_seed(0)
    built = build_model(cfg)
    assert torch.allclose(built.head.classifier.bias.cpu(), expected)


def test_default_flags_match_parent_logits_and_loss():
    """Defaults keep today's init: same logits and CE under a fixed seed."""
    x = torch.randn(1, 224, 32, 32)
    # Parent-style constructor (no new kwargs).
    set_seed(0)
    parent = BandTransformer(
        depth_spectral=2,
        depth_spatial=6,
        dim=64,
        heads=4,
        mlp_ratio=4.0,
        num_classes=7,
        n_bands=224,
        group=16,
        patch=8,
    )
    set_seed(0)
    cfg = TrainConfig(model="transformer", pe_init_std=1.0, head_prior_bias=False)
    child = build_model(cfg, in_channels=224)
    parent.eval()
    child.eval()
    with torch.no_grad():
        logits_p = parent(x)
        logits_c = child(x)
    assert torch.equal(logits_p, logits_c)

    weights = torch.ones(7)
    crit = nn.CrossEntropyLoss(weight=weights)
    target = torch.zeros(1, 32, 32, dtype=torch.int64)
    assert crit(logits_p, target).item() == crit(logits_c, target).item()


def test_cli_defaults_preserve_today():
    cfg = config_from_args(_build_parser().parse_args(["--model", "transformer"]))
    assert cfg.standardize is False
    assert cfg.stats_path is None
    assert cfg.pe_init_std == 1.0
    assert cfg.head_prior_bias is False
    assert cfg.lr == 1e-3


def test_diagnostic_ratios_rise_with_standardize():
    """On the synthetic offset cube, --standardize must not be a no-op."""
    diag = _load_diag_module()
    g = torch.Generator().manual_seed(0)
    raw = diag.synthetic_offset_cube(batch=1, height=64, width=64, generator=g)
    mean = raw.mean(dim=(0, 2, 3))
    std = raw.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
    whitened = BandStandardizer(mean, std)(raw)

    model_raw = diag.build_diag_model(pe_init_std=1.0, seed=0)
    model_std = diag.build_diag_model(pe_init_std=1.0, seed=0)
    r_raw = diag.measure_input_blind_ratios(model_raw, raw)
    r_std = diag.measure_input_blind_ratios(model_std, whitened)

    assert r_std["tokenizer_spatial_std_rms"] > r_raw["tokenizer_spatial_std_rms"]
    assert r_std["head_in_spatial_std_rms"] > r_raw["head_in_spatial_std_rms"]
    assert r_std["logits_spatial_std_rms"] > r_raw["logits_spatial_std_rms"]

    # PE shrink on top of standardise should raise tokenizer ratio further.
    model_pe = diag.build_diag_model(pe_init_std=0.02, seed=0)
    r_pe = diag.measure_input_blind_ratios(model_pe, whitened)
    assert r_pe["tokenizer_spatial_std_rms"] > r_std["tokenizer_spatial_std_rms"]


def test_diag_cli_synthetic_runs(capsys):
    diag = _load_diag_module()
    ratios = diag.main(["--seed", "0", "--height", "64", "--width", "64", "--batch-size", "1"])
    assert "tokenizer_spatial_std_rms" in ratios
    out = capsys.readouterr().out
    assert "tokenizer+PE" in out
    ratios2 = diag.main(
        [
            "--seed",
            "0",
            "--height",
            "64",
            "--width",
            "64",
            "--standardize",
            "--pe-init-std",
            "0.02",
        ]
    )
    assert ratios2["tokenizer_spatial_std_rms"] > ratios["tokenizer_spatial_std_rms"]
