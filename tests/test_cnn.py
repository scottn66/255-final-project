"""MiniNetSeg: full-resolution logits, and a parameter count in the control band."""

from __future__ import annotations

import torch

from sw.models.cnn import MiniNetSeg, count_parameters


def test_mininet_output_shape_and_param_count():
    import sw

    assert sw.MiniNetSeg is MiniNetSeg
    model = MiniNetSeg()
    n_params = count_parameters(model)
    # Scaffold band for the CNN control. The checkup bar (52.8 / 54.3) is a
    # score target, not a parameter target. 571_879 is this default width.
    assert n_params == 571_879
    assert 400_000 <= n_params <= 800_000

    # Non-square on purpose: the head must follow the input, not a fixed 256.
    images = torch.randn(2, 224, 30, 40)
    logits = model(images)
    assert logits.shape == (2, 7, 30, 40)
    assert torch.isfinite(logits).all()
    logits.sum().backward()

    stem = model.stem[0]
    assert isinstance(stem, torch.nn.Conv2d)
    assert stem.kernel_size == (1, 1)
    assert stem.in_channels == 224
    assert stem.out_channels < 224


def test_mininet_does_not_pull_in_a_pretrained_backbone():
    import sw.models.cnn as cnn

    source = open(cnn.__file__, encoding="utf-8").read()
    assert "nn.MultiheadAttention(" not in source
    assert "import timm" not in source
    assert "torchvision.models" not in source
    assert "load_state_dict" not in source
