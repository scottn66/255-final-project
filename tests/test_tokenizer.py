"""Unit tests for BandTokenizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sw.models.tokenizer import BandTokenizer
from sw.models.tokenizer import CollapsedTokenizer


def test_tokenizer_shape_and_finite():
    tokens, (h, w) = BandTokenizer()(torch.randn(2, 224, 256, 256))
    assert (h, w) == (32, 32)
    assert tokens.shape == (2, 1024, 14, 64)
    assert torch.isfinite(tokens).all()

def test_tokenizer_param_count():
    # conv 65,600 + layernorm 128 + group embedding 14*64 = 896
    n_params = sum(p.numel() for p in BandTokenizer().parameters())
    assert n_params == 66_624


# --- CollapsedTokenizer: the tokenizer of the band-collapsed control ---


def test_collapsed_tokenizer_shape_and_finite():
    tokenizer = CollapsedTokenizer()
    assert tokenizer.n_groups == 1  # the band axis is gone before any attention
    # 30 by 40 is not a multiple of patch 8: the grid is 3 by 5.
    x = torch.randn(2, 224, 30, 40)
    tokens, (h, w) = tokenizer(x)
    assert (h, w) == (3, 5)
    assert tokens.shape == (2, 15, 1, 64)  # (B, h*w, G = 1, D)
    assert torch.isfinite(tokens).all()
    assert BandTokenizer()(x)[1] == (3, 5)  # the same grid as the band tokenizer
    # Rows 24 to 29 are past the last full patch. They are dropped: changing
    # them moves no token (measured: exactly 0).
    poked = x.clone()
    poked[:, :, 24:] += 100 * torch.randn(2, 224, 6, 40)
    assert (tokenizer(poked)[0] - tokens).abs().max() < 1e-6


def test_collapsed_tokenizer_param_count():
    # 1x1 conv 224*16 + 16 = 3,600 + patch conv 65,600 + layernorm 128
    tokenizer = CollapsedTokenizer()
    n_params = sum(p.numel() for p in tokenizer.parameters())
    assert n_params == 3_600 + 65_600 + 128 == 69_328
    # reduce=16 gives the patch conv the band tokenizer's shape. That is the parameter match.
    assert tokenizer.proj.weight.shape == BandTokenizer().proj.weight.shape == (64, 16, 8, 8)
    # One group has no order to encode, so there is no group embedding.
    assert list(tokenizer.state_dict()) == [
        "reduce.weight", "reduce.bias", "proj.weight", "proj.bias", "norm.weight", "norm.bias",
    ]
    # The 224 -> 32 of docs/RECIPE.md doubles the patch conv: 7,200 + 131,136 + 128.
    assert sum(p.numel() for p in CollapsedTokenizer(reduce=32).parameters()) == 138_464


def test_collapsed_tokenizer_rejects_bad_arguments():
    with pytest.raises(ValueError, match="bands"):
        CollapsedTokenizer()(torch.randn(1, 200, 16, 16))  # wrong band count, like BandTokenizer
    with pytest.raises(ValueError, match="bands"):
        BandTokenizer()(torch.randn(1, 200, 16, 16))
    with pytest.raises(ValueError, match="bands"):
        CollapsedTokenizer(n_bands=200)(torch.randn(1, 224, 16, 16))
    for reduce in (0, -1):
        with pytest.raises(ValueError, match="reduce"):
            CollapsedTokenizer(reduce=reduce)
    # Other sizes build: the arguments reach the layers.
    tokens, (h, w) = CollapsedTokenizer(n_bands=32, d_model=24, reduce=1, patch=4)(torch.randn(1, 32, 8, 12))
    assert (h, w) == (2, 3)
    assert tokens.shape == (1, 6, 1, 24)


def test_collapsed_tokenizer_matches_hand_written_reference():
    """Values, not only shapes: a per-pixel mix of the bands, a linear map
    of each patch, then LayerNorm, with nothing in between. In float64, so
    1e-12 is meaningful. Small sizes; the patch grid is 3 by 5."""
    torch.manual_seed(0)
    tokenizer = CollapsedTokenizer(n_bands=32, d_model=24, reduce=5, patch=4).double().eval()
    # Non-default LayerNorm parameters, so a missing LayerNorm shows.
    torch.nn.init.normal_(tokenizer.norm.weight, 1.0, 0.3)
    torch.nn.init.normal_(tokenizer.norm.bias, 0.0, 0.3)
    x = torch.randn(2, 32, 12, 20, dtype=torch.float64)
    tokens, (h, w) = tokenizer(x)
    assert (h, w) == (3, 5)

    mix, patch = tokenizer.reduce, tokenizer.proj
    # 1x1 conv: every pixel's 32 bands go through the same 5 by 32 matrix.
    mixed = torch.einsum("rc,bchw->brhw", mix.weight[:, :, 0, 0], x) + mix.bias[None, :, None, None]
    for i in range(h):
        for j in range(w):
            window = mixed[:, :, i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4]  # (B, 5, 4, 4): one patch
            embedded = torch.einsum("drkl,brkl->bd", patch.weight, window) + patch.bias
            expected = F.layer_norm(embedded, (24,), tokenizer.norm.weight, tokenizer.norm.bias, tokenizer.norm.eps)
            assert torch.allclose(tokens[:, i * w + j, 0], expected, rtol=0, atol=1e-12), (i, j)


def test_collapsed_tokenizer_token_order_is_row_major():
    """Token n is patch (n // w, n % w), exactly like BandTokenizer.

    Poke one 8 by 8 input patch at a time: only that patch's token moves.
    """
    torch.manual_seed(0)
    tokenizer = CollapsedTokenizer().eval()
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
