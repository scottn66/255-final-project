"""Unit tests for BandTokenizer."""

from __future__ import annotations

import torch

from sw.models.tokenizer import BandTokenizer


def test_tokenizer_shape_and_finite():
    tokens, (h, w) = BandTokenizer()(torch.randn(2, 224, 256, 256))
    assert (h, w) == (32, 32)
    assert tokens.shape == (2, 1024, 14, 64)
    assert torch.isfinite(tokens).all()

def test_tokenizer_param_count():
    # conv 65,600 + layernorm 128 + group embedding 14*64 = 896
    n_params = sum(p.numel() for p in BandTokenizer().parameters())
    assert n_params == 66_624
