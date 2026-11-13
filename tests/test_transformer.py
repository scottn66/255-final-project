"""Contracts for the band-token transformer.

The body that still raises ``NotImplementedError`` (``SegHead``) has
``xfail`` tests with ``strict=False``: the suite stays green now, a wrong
shape fails once the body exists, and a matching shape is reported as
XPASS. ``docs/TRANSFORMER-TODO.md`` says which test to run after each piece.

``SpatialPE`` and ``SpatialBlock`` are tested on values, not only shapes:
hand-computed sin-cos codes, a functional-only reference for the block,
the token order against the tokenizer, the calls inside
``BandTransformer``, and the dtype and device paths. The ``SpectralBlock``
and ``SpectralPool`` tests are still shape-only. Parameter counts are
locked for the tokenizer, ``SpatialPE`` (zero, a design choice) and
``SpatialBlock`` (from the layer formula).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw.models.cnn import MiniNetSeg, count_parameters
from sw.models.tokenizer import BandTokenizer
from sw.attention import ResidualSelfAttentionBlock
import torch.nn.functional as F
from sw.models.transformer import (
    BandTransformer,
    DropPath,
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


def _pe_code(pe, h, w, dim=64):
    """The embedding at each site, read through the public forward on zero tokens."""
    return pe(torch.zeros(1, h * w, dim), h, w)[0]


def _golden_code(i, j, dim):
    """Site (i, j) from the docstring formula, in plain ``math``.

    This is the module's recipe written a second time, so it only checks
    the tensor code against the math. The hand-written D = 8 and D = 6
    cases in the test below are the independent check of the recipe itself.
    """
    n_freq = -(-dim // 4)
    omega = [10000.0 ** (-k / n_freq) for k in range(n_freq)]

    def half(p):
        values = [math.sin(p * o) for o in omega] + [math.cos(p * o) for o in omega]
        return values[: dim // 2]

    return half(i) + half(j)


def test_spatial_pe_shape():
    pe = SpatialPE(dim=64)
    # Zeros so a real embedding has to be what makes two sites differ.
    h, w = 3, 5
    tokens = torch.zeros(2, h * w, 64)  # (B, N, D)
    out = pe(tokens, h, w)
    assert out.shape == (2, h * w, 64)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert torch.equal(out[0], out[1])  # one code per site, shared across the batch
    with pytest.raises(ValueError, match="grid"):
        pe(tokens, 5, 5)  # N = 15 tokens cannot be a 5 by 5 grid


def test_spatial_pe_matches_hand_computed_codes():
    """Values, not just shapes. Catches swapped sin/cos, wrong frequencies,
    swapped row and column halves, and a scaled embedding."""
    h, w = 3, 5

    # D = 8: omega = (1, 0.01). Written out by hand.
    code = _pe_code(SpatialPE(8), h, w, dim=8).double()
    for i in range(h):
        for j in range(w):
            expected = [
                math.sin(i), math.sin(i / 100), math.cos(i), math.cos(i / 100),
                math.sin(j), math.sin(j / 100), math.cos(j), math.cos(j / 100),
            ]
            assert torch.allclose(code[i * w + j], torch.tensor(expected, dtype=torch.float64), atol=1e-6)

    # D = 6 (D % 4 == 2): each half keeps both sines and the first cosine.
    code = _pe_code(SpatialPE(6), h, w, dim=6).double()
    for i in range(h):
        for j in range(w):
            expected = [math.sin(i), math.sin(i / 100), math.cos(i), math.sin(j), math.sin(j / 100), math.cos(j)]
            assert torch.allclose(code[i * w + j], torch.tensor(expected, dtype=torch.float64), atol=1e-6)

    # D = 64 on the full-resolution 32 by 32 grid, at the corners and inside.
    code = _pe_code(SpatialPE(64), 32, 32).double()
    for i, j in ((0, 0), (0, 31), (31, 0), (7, 19), (31, 31)):
        expected = torch.tensor(_golden_code(i, j, 64), dtype=torch.float64)
        assert torch.allclose(code[i * 32 + j], expected, atol=1e-5)


def test_spatial_pe_adds_to_tokens_and_leaves_them_unchanged():
    pe = SpatialPE(dim=64)
    x = torch.randn(2, 15, 64)
    before = x.clone()
    out = pe(x, 3, 5)
    assert torch.equal(x, before)  # out of place: the caller's tensor is not modified
    assert torch.allclose(out - x, _pe_code(pe, 3, 5).expand(2, -1, -1), atol=1e-6)  # the tokens are kept

    # The embedding is a constant, so the gradient passes straight through.
    x.requires_grad_(True)
    upstream = torch.randn(2, 15, 64)
    pe(x, 3, 5).backward(upstream)
    assert torch.equal(x.grad, upstream)


def test_spatial_pe_is_fixed_row_major_and_grid_stable():
    pe = SpatialPE(dim=64)
    assert count_parameters(pe) == 0
    assert len(pe.state_dict()) == 0  # nothing ties a checkpoint to one grid size

    # Row-major and grid-stable: site (i, j) is token i * w + j, and a site
    # keeps its code when the grid grows. A non-square grid catches a
    # column-major flatten; 16 -> 32 is the downsample 2 -> full case.
    h, w = 3, 5
    grid = _pe_code(pe, h, w).reshape(h, w, 64)
    big = _pe_code(pe, 6, 9).reshape(6, 9, 64)
    assert torch.allclose(big[:3, :5], grid, atol=1e-6)
    full = _pe_code(pe, 32, 32)
    assert torch.allclose(full.reshape(32, 32, 64)[:16, :16], _pe_code(pe, 16, 16).reshape(16, 16, 64), atol=1e-6)

    # Every site on the full-resolution grid has its own code.
    assert (torch.cdist(full, full) + torch.eye(1024)).min() > 1e-3

    # Even widths that are not a multiple of 4 still build. Odd widths would
    # make the row and column halves unequal, and widths below 4 nearly
    # repeat codes on a 32 by 32 grid, so both raise.
    for dim in (4, 6, 10, 62):
        code = _pe_code(SpatialPE(dim), 7, 9, dim)
        assert code.shape == (63, dim)
        assert (torch.cdist(code, code) + torch.eye(63)).min() > 1e-3
    for dim in (0, 2, 5, 7, 63):
        with pytest.raises(ValueError, match="dim"):
            SpatialPE(dim)


def test_spatial_pe_keeps_dtype_and_device():
    pe = SpatialPE(dim=64)
    # Half precision (training with --amp) and float64 (gradcheck) come back
    # in their own dtype. A float32 table added without a cast would turn
    # bf16 tokens into float32.
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        out = pe(torch.zeros(1, 15, 64, dtype=dtype), 3, 5)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
    # Float64 tokens get a table computed in float64.
    code64 = pe(torch.zeros(1, 1024, 64, dtype=torch.float64), 32, 32)[0]
    assert torch.allclose(code64[-1], torch.tensor(_golden_code(31, 31, 64), dtype=torch.float64), rtol=0, atol=1e-12)
    # The table is built on the tokens' device ("meta" stands in for CUDA here).
    assert pe(torch.zeros(1, 15, 64, device="meta"), 3, 5).device.type == "meta"


def test_spatial_pe_half_precision_values_and_input_guards():
    pe = SpatialPE(dim=64)
    # Half-precision tokens get the float32 table cast down, not a table
    # built in half precision (that one is off by up to 0.067 in bf16).
    reference = pe(torch.zeros(1, 1024, 64), 32, 32)[0]
    for dtype in (torch.float16, torch.bfloat16):
        assert torch.equal(pe(torch.zeros(1, 1024, 64, dtype=dtype), 32, 32)[0], reference.to(dtype))

    # The temperature argument is used: at D = 8 and temperature 100,
    # omega = (1, 0.1).
    code = _pe_code(SpatialPE(8, temperature=100.0), 1, 3, dim=8).double()
    j = 2
    expected = [0.0, 0.0, 1.0, 1.0, math.sin(j), math.sin(j / 10), math.cos(j), math.cos(j / 10)]
    assert torch.allclose(code[j], torch.tensor(expected, dtype=torch.float64), atol=1e-6)

    # Bad inputs raise a clear ValueError instead of a broadcast error later.
    with pytest.raises(ValueError, match="expected"):
        pe(torch.zeros(1, 15, 32), 3, 5)  # wrong width D
    with pytest.raises(ValueError, match="expected"):
        pe(torch.zeros(15, 64), 3, 5)  # no batch axis
    with pytest.raises(ValueError, match="grid"):
        pe(torch.zeros(1, 0, 64), 0, 5)  # empty grid: N = 0 matches h * w, still rejected
    for temperature in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            SpatialPE(64, temperature=temperature)
    with pytest.raises(ValueError, match="dtype"):
        pe.table(3, 5, dtype=torch.bfloat16)  # half-precision tables are refused


def test_band_transformer_adds_the_pe_on_the_tokenizer_grid():
    """The call in BandTransformer.forward: the PE runs, on (h, w) in that order.

    A 24 by 40 input is a 3 by 5 grid, so passing (w, h) would add the
    5 by 3 table instead. depth_spatial=0 stops before the spatial-block
    TODO; the head may still be a TODO, which is fine here.
    """
    torch.manual_seed(0)
    model = BandTransformer(depth_spectral=1, depth_spatial=0).eval()
    calls = []

    def record(module, args, output):
        tokens, h, w = args
        calls.append((h, w, (output - tokens).detach()))

    handle = model.pos.register_forward_hook(record)
    try:
        with torch.no_grad():
            model(torch.rand(1, 224, 24, 40))
    except NotImplementedError:
        pass  # SegHead is still a TODO; the PE call already happened
    finally:
        handle.remove()
    assert len(calls) == 1
    h, w, added = calls[0]
    assert (h, w) == (3, 5)
    assert torch.allclose(added[0], _pe_code(SpatialPE(64), 3, 5), atol=1e-6)


def test_tokenizer_token_order_is_row_major():
    """SpatialPE assumes token n is patch (n // w, n % w). Check the tokenizer agrees."""
    torch.manual_seed(0)
    tokenizer = BandTokenizer().eval()
    x = torch.rand(1, 224, 24, 40)  # patch 8: a 3 by 5 grid, non-square on purpose
    base, (h, w) = tokenizer(x)
    assert (h, w) == (3, 5)
    for i in range(h):
        for j in range(w):
            poked = x.clone()
            poked[:, :, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] += torch.randn(224, 8, 8)
            tokens, _ = tokenizer(poked)
            changed = (tokens - base).abs().flatten(2).amax(dim=2)[0]  # (N,) largest change per token
            assert int(changed.argmax()) == i * w + j
            assert int((changed > 1e-6).sum()) == 1  # no other token moved


def test_spatial_block_shape():
    block = SpatialBlock(dim=64, heads=4, mlp_ratio=4)
    tokens = torch.randn(2, 16, 64)  # (B, N, D)
    out = block(tokens)
    assert isinstance(out, torch.Tensor)  # tokens only, not (tokens, weights)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="expected"):
        block(torch.randn(2, 16, 32))  # wrong width
    with pytest.raises(ValueError, match="expected"):
        block(torch.randn(2, 4, 4, 64))  # still has a group axis


def _reference_block(block, x):
    """Pre-norm attention + MLP written with torch.nn.functional only.

    Independent of sw.attention's code paths: the heads are split by hand,
    the softmax is F.softmax, and the MLP is F.linear + F.gelu. It reads
    the block's weights, so any difference is a difference in the math.
    """
    b, n, d = x.shape
    heads = block.heads

    def split(t):  # (B, N, D) -> (B, H, N, D / H)
        return t.reshape(b, n, heads, d // heads).permute(0, 2, 1, 3)

    a = block.attn
    y = F.layer_norm(x, (d,), block.ln1.weight, block.ln1.bias, block.ln1.eps)
    q = split(F.linear(y, a.w_q.weight, a.w_q.bias))
    k = split(F.linear(y, a.w_k.weight, a.w_k.bias))
    v = split(F.linear(y, a.w_v.weight, a.w_v.bias))
    weights = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // heads), dim=-1)  # (B, H, N, N)
    mixed = (weights @ v).permute(0, 2, 1, 3).reshape(b, n, d)
    x = x + F.linear(mixed, a.w_o.weight, a.w_o.bias)
    y = F.layer_norm(x, (d,), block.ln2.weight, block.ln2.bias, block.ln2.eps)
    f = block.ffn
    return x + F.linear(F.gelu(F.linear(y, f.fc1.weight, f.fc1.bias)), f.fc2.weight, f.fc2.bias)


def test_spatial_block_matches_independent_references():
    """Values, not shapes, in float64 so 1e-12 is meaningful.

    The functional reference is the independent check. The merged
    ResidualSelfAttentionBlock shares MultiHeadSelfAttention and
    FeedForward with this block, so agreeing with it shows the composition
    (pre-norm order, residuals, names) matches, not the attention math.
    """
    torch.manual_seed(0)
    block = SpatialBlock(dim=16, heads=4, mlp_ratio=4).double().eval()
    # Non-default LayerNorm parameters, so a missing or swapped LayerNorm shows.
    for norm in (block.ln1, block.ln2):
        torch.nn.init.normal_(norm.weight, 1.0, 0.3)
        torch.nn.init.normal_(norm.bias, 0.0, 0.3)
    x = torch.randn(3, 9, 16, dtype=torch.float64)
    out = block(x)
    assert torch.allclose(out, _reference_block(block, x), rtol=0, atol=1e-12)

    # The merged block has the same layers and names, so the weights load
    # with strict=True and the outputs agree.
    merged = ResidualSelfAttentionBlock(16, 4).double().eval()
    merged.load_state_dict(block.state_dict(), strict=True)
    assert torch.allclose(out, merged(x)[0], rtol=0, atol=1e-12)


def test_spatial_block_param_count_follows_mlp_ratio():
    def expected(d, hidden):
        layer_norms = 2 * (2 * d)  # two LayerNorms, weight + bias each
        attention = 4 * (d * d + d)  # Q, K, V, O projections
        mlp = (d * hidden + hidden) + (hidden * d + d)  # fc1 + fc2
        return layer_norms + attention + mlp

    for dim, mlp_ratio, hidden in (
        (64, 4, 256), (64, 2, 128), (64, 1.5, 96),
        (12, 0.65, 8),  # 7.8 rounds up (floor would give 7)
        (12, 0.7, 8),  # 8.4 rounds down (ceil would give 9)
        (12, 0.625, 8),  # 7.5: Python's round goes to the even width
    ):
        block = SpatialBlock(dim=dim, heads=4, mlp_ratio=mlp_ratio)
        assert count_parameters(block) == expected(dim, hidden), (dim, mlp_ratio)
    with pytest.raises(ValueError, match="round"):
        SpatialBlock(dim=12, heads=4, mlp_ratio=0.01)  # 0.12 would round to an empty MLP
    # At mlp_ratio 4 the spatial and spectral blocks are matched (49,984 each).
    spatial = SpatialBlock(dim=64, heads=4, mlp_ratio=4)
    assert count_parameters(spatial) == count_parameters(SpectralBlock(dim=64, heads=4, mlp_ratio=4)) == 49_984
    assert count_parameters(SpatialBlock(dim=64, heads=4, drop_path=0.3)) == 49_984  # drop path is free


def test_spatial_block_attends_across_tokens():
    torch.manual_seed(0)
    block = SpatialBlock(dim=64, heads=4).eval()
    tokens = torch.randn(2, 9, 64)
    out = block(tokens)
    # No position term inside the block: reordering the tokens reorders the
    # output and changes nothing else (SpatialPE is what tells sites apart).
    order = torch.randperm(9)
    assert torch.allclose(block(tokens[:, order]), out[:, order], atol=1e-5)
    # Attention mixes tokens: changing token 0 moves the others. (A random
    # poke, not a constant shift, which the pre-norm would remove.)
    poked = tokens.clone()
    poked[:, 0] += torch.randn(64)
    assert (block(poked)[:, 1:] - out[:, 1:]).abs().max() > 1e-3
    # Images in a batch do not mix.
    other = tokens.clone()
    other[1] = torch.randn(9, 64)
    assert torch.equal(block(other)[0], out[0])


def test_spatial_block_gradcheck():
    torch.manual_seed(0)
    block = SpatialBlock(dim=8, heads=2, mlp_ratio=2).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(block, (x,), eps=1e-6, atol=1e-6)


def test_drop_path_drops_whole_samples_and_keeps_the_mean():
    # p = 0.25, not 0.5: at 0.5, dropping with probability "keep" by mistake
    # would look the same.
    drop = DropPath(0.25)
    assert count_parameters(drop) == 0
    x = torch.ones(4000, 3, 2)
    drop.eval()
    assert drop(x) is x  # eval: identity
    drop.train()
    torch.manual_seed(0)
    out = drop(x)
    per_sample = out.reshape(4000, -1)
    dropped = (per_sample == 0).all(dim=1)
    kept = torch.isclose(per_sample, torch.tensor(1 / 0.75)).all(dim=1)  # kept values are divided by keep = 0.75
    assert (dropped | kept).all()  # each sample is dropped whole or kept whole
    assert 0.22 < dropped.float().mean() < 0.28  # about p of them (4000 draws, sd 0.007)
    assert abs(out.mean().item() - 1.0) < 0.05  # expected value matches eval
    assert DropPath(0.0).train()(x) is x  # p = 0: identity in training too
    for p in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="drop path"):
            DropPath(p)
    half = DropPath(0.5).train()(torch.ones(8, 4, dtype=torch.bfloat16))
    assert half.dtype == torch.bfloat16

    # Any rank: the output keeps the input's shape, each sample is dropped
    # or kept whole, and the input itself is not modified.
    for shape in ((400, 4), (400, 3, 2), (400, 2, 3, 5)):
        x = torch.ones(shape)
        out = DropPath(0.25).train()(x)
        assert out.shape == x.shape
        assert torch.equal(x, torch.ones(shape))  # out of place
        per_sample = out.reshape(400, -1)
        assert ((per_sample == 0).all(dim=1) | (per_sample == per_sample[:, :1]).all(dim=1)).all()
        assert (per_sample == 0).all(dim=1).any() and not (per_sample == 0).all(dim=1).all()


def test_spatial_block_drop_path_is_per_image_and_off_in_eval():
    torch.manual_seed(0)
    tokens = torch.randn(256, 4, 16)
    plain = SpatialBlock(dim=16, heads=2, drop_path=0.0)
    assert torch.equal(plain.train()(tokens), plain.eval()(tokens))  # p = 0: train mode changes nothing

    block = SpatialBlock(dim=16, heads=2, drop_path=0.5)
    block.eval()
    reference = block(tokens)
    assert torch.equal(block(tokens), reference)  # eval is deterministic

    block.train()
    torch.manual_seed(1)
    out = block(tokens)
    attn_out = block.attn(block.ln1(tokens))[0]
    # Each image ends up in one of four states, one per (attention kept?,
    # MLP kept?) pair. Check that every image matches one of them exactly.
    states = []
    for i in range(256):
        x = tokens[i : i + 1]
        a = attn_out[i : i + 1]
        candidates = []
        for keep_attn in (0.0, 2.0):  # dropped, or kept and divided by 0.5
            y = x + keep_attn * a
            for keep_mlp in (0.0, 2.0):
                candidates.append(y + keep_mlp * block.ffn(block.ln2(y)))
        matches = [k for k, c in enumerate(candidates) if torch.allclose(out[i : i + 1], c, atol=1e-5)]
        assert matches, f"image {i} is not a whole-image drop of either branch"
        states.append(matches[0])
    assert set(states) == {0, 1, 2, 3}  # all four combinations occur in 256 images


def test_band_transformer_runs_the_spatial_blocks_in_order():
    """The spatial blocks get the PE'd tokens, in sequence, at full size,
    and the gradient reaches the tokenizer. The head may still be a TODO."""
    torch.manual_seed(0)
    model = BandTransformer(depth_spectral=1, depth_spatial=3).train()
    seen = []
    handles = [model.pos.register_forward_hook(lambda m, a, o: seen.append(("pos", o)))]
    for index, block in enumerate(model.spatial):
        handles.append(block.register_forward_hook(lambda m, a, o, index=index: seen.append((index, a[0], o))))
    try:
        model(torch.rand(1, 224, 256, 256))
    except NotImplementedError:
        pass  # SegHead
    finally:
        for handle in handles:
            handle.remove()
    assert [entry[0] for entry in seen] == ["pos", 0, 1, 2]
    previous = seen[0][1]
    for _, inputs, outputs in seen[1:]:
        assert inputs is previous  # each block reads the previous output
        assert outputs.shape == (1, 1024, 64)  # 32 by 32 tokens at full resolution
        assert torch.isfinite(outputs).all()
        previous = outputs
    previous.square().mean().backward()
    assert model.tokenizer.proj.weight.grad.abs().sum() > 0

    # mlp_ratio reaches the spatial blocks. (SpectralBlock still requires 4,
    # so this uses depth_spectral=0.)
    narrow = BandTransformer(depth_spectral=0, depth_spatial=2, mlp_ratio=2)
    for block in narrow.spatial:
        assert count_parameters(block) == count_parameters(SpatialBlock(64, 4, mlp_ratio=2)) == 33_472


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
