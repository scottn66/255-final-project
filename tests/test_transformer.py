"""Shape contracts for the band-token scaffold.

The five bodies raise ``NotImplementedError`` until they are filled in.
Those tests are ``xfail`` with ``strict=False``: the suite stays green
now, a wrong shape fails once the body exists, and a matching shape is
reported as XPASS. ``docs/TRANSFORMER-TODO.md`` says which test to run
after each piece.

Parameter counts are asserted only for the tokenizer. Pool (mean vs
learned token) and spatial PE (sin-cos vs a table) are open choices, and
the block bodies are still TODOs, so those counts are not locked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw.models.cnn import MiniNetSeg, count_parameters
from sw.models.tokenizer import BandTokenizer
from sw.models.transformer import (
    BandTransformer,
    SegHead,
    SpatialBlock,
    SpatialPE,
    SpectralBlock,
    SpectralPool,
)
from sw.train import _build_parser, build_model, config_from_args, main

# A NotImplementedError is the expected failure. Any other exception, including
# a wrong shape, fails the run. strict=False keeps an XPASS (body filled, shape
# matches) from failing the suite.
todo = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=False,
    reason="body is still a TODO; NotImplementedError until it is filled in",
)


def test_blocks_reject_heads_that_do_not_divide_dim():
    with pytest.raises(ValueError, match="divisible"):
        SpectralBlock(dim=64, heads=3, mlp_ratio=4)
    with pytest.raises(ValueError, match="divisible"):
        SpatialBlock(dim=64, heads=3, mlp_ratio=4)
    with pytest.raises(ValueError, match="divisible"):
        BandTransformer(depth_spectral=1, depth_spatial=1, dim=64, heads=3)


def test_band_transformer_wiring_and_tokenizer_param_count():
    """Tokenizer count is fixed by BandTokenizer. The TODO modules are not."""
    import sw.models as models
    import sw.models.transformer as transformer

    model = BandTransformer(depth_spectral=2, depth_spatial=6, dim=64, heads=4, mlp_ratio=4)
    assert models.BandTransformer is BandTransformer
    assert models.MiniNetSeg is MiniNetSeg
    assert isinstance(model.tokenizer, BandTokenizer)
    assert model.tokenizer.n_groups == 14
    assert model.tokenizer.group == 16
    assert count_parameters(model.tokenizer) == 66_624
    assert len(model.spectral) == 2
    assert all(isinstance(block, SpectralBlock) for block in model.spectral)
    assert isinstance(model.pool, SpectralPool)
    assert isinstance(model.pos, SpatialPE)
    assert len(model.spatial) == 6
    assert all(isinstance(block, SpatialBlock) for block in model.spatial)
    assert isinstance(model.head, SegHead)
    assert model.head.num_classes == 7
    assert model.head.patch == 8

    source = Path(transformer.__file__).read_text(encoding="utf-8")
    assert "sw.attention" in source
    assert "nn.MultiheadAttention(" not in source
    assert "nn.TransformerEncoder" not in source
    assert "import timm" not in source


def test_default_model_is_cnn_and_transformer_flags_build():
    cnn_args = _build_parser().parse_args([])
    cnn_cfg = config_from_args(cnn_args)
    assert cnn_cfg.model == "cnn"
    assert cnn_cfg.out_dir == "runs/cnn"
    assert isinstance(build_model(cnn_cfg, in_channels=cnn_args.in_channels), MiniNetSeg)

    args = _build_parser().parse_args(["--model", "transformer"])
    cfg = config_from_args(args)
    assert cfg.model == "transformer"
    assert cfg.depth_spectral == 2
    assert cfg.depth_spatial == 6
    assert cfg.dim == 64
    assert cfg.heads == 4
    assert cfg.mlp_ratio == 4.0
    assert cfg.patch == 8
    assert cfg.group == 16
    assert cfg.out_dir == "runs/transformer"
    built = build_model(cfg, in_channels=224)
    assert isinstance(built, BandTransformer)
    assert built.tokenizer.n_groups == 14
    assert len(built.spectral) == 2
    assert len(built.spatial) == 6

    # --smoke still shortens the run and keeps its own output directory.
    smoke = config_from_args(
        _build_parser().parse_args(["--model", "transformer", "--smoke", "--device", "cpu"])
    )
    assert smoke.epochs == 1
    assert smoke.batch_size == 2
    assert smoke.max_train_steps == 2
    assert smoke.downsample == 4
    assert smoke.out_dir == "runs/smoke"



def test_spectral_block_shape():
    block = SpectralBlock(dim=64, heads=4, mlp_ratio=4)
    tokens = torch.randn(2, 16, 14, 64)  # (B, N, G, D)
    out = block(tokens)
    assert out.shape == (2, 16, 14, 64)
    assert torch.isfinite(out).all()



def test_spectral_pool_shape():
    pool = SpectralPool(dim=64)
    tokens = torch.randn(2, 16, 14, 64)  # (B, N, G, D)
    out = pool(tokens)
    assert out.shape == (2, 16, 64)  # (B, N, D)
    assert torch.isfinite(out).all()


@todo
def test_spatial_pe_shape():
    pe = SpatialPE(dim=64)
    # Zeros so a real embedding has to be what makes two sites differ.
    h, w = 3, 5
    tokens = torch.zeros(2, h * w, 64)  # (B, N, D)
    out = pe(tokens, h, w)
    assert out.shape == (2, h * w, 64)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0, 0], out[0, 1])


@todo
def test_spatial_block_shape():
    block = SpatialBlock(dim=64, heads=4, mlp_ratio=4)
    tokens = torch.randn(2, 16, 64)  # (B, N, D)
    out = block(tokens)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


@todo
def test_seg_head_shape():
    head = SegHead(dim=64, num_classes=7, patch=8)
    # 30x40 is not a multiple of patch 8: grid is 3x5, full size stays 30x40.
    h, w = 3, 5
    tokens = torch.randn(2, h * w, 64)
    logits = head(tokens, h, w, out_size=(30, 40))
    assert logits.shape == (2, 7, 30, 40)
    assert torch.isfinite(logits).all()


@todo
def test_band_transformer_shape():
    model = BandTransformer(
        depth_spectral=1,
        depth_spatial=1,
        dim=64,
        heads=4,
        mlp_ratio=4,
        num_classes=7,
    )
    images = torch.randn(2, 224, 30, 40)
    logits = model(images)
    assert logits.shape == (2, 7, 30, 40)
    assert torch.isfinite(logits).all()
    logits.sum().backward()


def _synthetic_root(root: Path) -> None:
    """Four train cubes and two val cubes. No real SpectralWaste files."""
    shape = (224, 32, 32)
    _, height, width = shape
    for split, n_images in (("train", 4), ("val", 2)):
        hyper = root / "hyper" / split
        labels = root / "labels_hyper_lt" / split
        hyper.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(n_images):
            cube = np.zeros(shape, dtype=np.uint16)
            cube[0, 0, 0] = 1000 * (index + 1)
            tifffile.imwrite(hyper / f"{split}_{index:02d}.tiff", cube, photometric="minisblack")
            mask = np.zeros((height, width), dtype=np.uint8)
            for cls in range(7):
                mask[:, cls::7] = cls
            Image.fromarray(mask, mode="L").save(labels / f"{split}_{index:02d}.png")


@todo
def test_transformer_smoke(tmp_path):
    """``--model transformer --smoke`` on a tiny synthetic root."""
    root = tmp_path / "data"
    _synthetic_root(root)
    out = tmp_path / "run"
    result = main(
        [
            "--model",
            "transformer",
            "--smoke",
            "--data-root",
            str(root),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--seed",
            "0",
        ]
    )
    assert result["epochs_ran"] == 1
    # Tokenizer parameters are always present. Blocks only add more.
    assert result["param_count"] >= 66_624
    with (out / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 1
    assert rows[0]["steps"] == 2
