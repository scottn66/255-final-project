"""Compact MiniNet-family CNN for SpectralWaste.

MiniNet (Alonso et al.) is a small encoder-decoder: depthwise separable
convolutions, plus a module that runs several dilated depthwise convolutions
in parallel and mixes them with a 1x1. This file is a narrowed version of
that idea, sized for 224-band cubes.

It is the CNN control row for the Nov 4 checkup. The published hyperspectral
bar is MiniNet-v2 at 52.8 FG mIoU and SegFormer-B0 at 54.3. This scaffold
does not try to hit that bar.

The first layer is a 1x1 convolution. A 3x3 on 224 bands would spend the
parameter budget on the stem (224 * width * 9) before any spatial mixing.
The 1x1 mixes bands down to a few dozen channels. Every later spatial layer
is depthwise separable, which is what keeps the model inside 0.4–0.8M
parameters.

No pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    """Number of parameters, including BatchNorm scale and bias."""
    return sum(p.numel() for p in model.parameters())


class SeparableConv2d(nn.Module):
    """3x3 depthwise convolution, then a 1x1 pointwise convolution.

    A normal 3x3 costs ``in * out * 9``. This costs ``in * 9 + in * out``,
    which is the saving MiniNet is built on.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x


class MultiDilation(nn.Module):
    """MiniNet module: parallel dilated depthwise 3x3s, then a 1x1 mix.

    Dilation 1, 2, 4, and 8 see different context without stacking more
    layers or downsampling further. The 1x1 puts the branches back to
    ``channels``. A residual add makes the block safe to stack.
    """

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilations
            ]
        )
        mixed = channels * len(dilations)
        self.bn = nn.BatchNorm2d(mixed)
        self.mix = nn.Conv2d(mixed, channels, kernel_size=1, bias=False)
        self.mix_bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [branch(x) for branch in self.branches]
        mixed = self.act(self.bn(torch.cat(parts, dim=1)))
        mixed = self.act(self.mix_bn(self.mix(mixed)))
        return x + mixed


class EncoderStage(nn.Module):
    """Halve the map with a stride-2 separable conv, then a MiniNet module."""

    def __init__(self, in_ch: int, out_ch: int, dilations: tuple[int, ...]):
        super().__init__()
        self.down = SeparableConv2d(in_ch, out_ch, stride=2)
        self.block = MultiDilation(out_ch, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class DecoderStage(nn.Module):
    """Bilinear upsample to the skip, concatenate, 1x1 reduce, separable refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = SeparableConv2d(out_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(self.reduce(x))


class MiniNetSeg(nn.Module):
    """Encoder-decoder segmentation CNN.

    Args:
        in_channels: spectral bands. Real cubes have 224.
        num_classes: 7 (background plus six object classes).
        stem_channels: width after the 1x1 band mix.
        widths: channel count after each stride-2 stage. Four stages take
            256 pixels down to 16, which is the MiniNet-style bottleneck.
            The default widths are 571,879 parameters.
        dilations: rates inside every MiniNet module.

    Input ``(N, in_channels, H, W)`` returns logits ``(N, num_classes, H, W)``.
    """

    def __init__(
        self,
        in_channels: int = 224,
        num_classes: int = 7,
        stem_channels: int = 48,
        widths: tuple[int, ...] = (64, 96, 128, 176),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be >= 1")
        if len(widths) < 1:
            raise ValueError("widths must contain at least one stage")
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1x1 band reduction, then one separable conv so the full-resolution
        # skip has seen a 3x3 neighborhood, not only a spectral mix.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.stem_spatial = SeparableConv2d(stem_channels, stem_channels, stride=1)

        stages = []
        prev = stem_channels
        for width in widths:
            stages.append(EncoderStage(prev, width, dilations))
            prev = width
        self.encoder = nn.ModuleList(stages)
        # A second MiniNet module at the bottleneck, where dilation is cheapest.
        self.bottleneck = MultiDilation(widths[-1], dilations)

        # Skip channels from shallow to deep, excluding the bottleneck itself.
        skip_channels = [stem_channels, *widths[:-1]]
        decoder = []
        in_ch = widths[-1]
        for skip_ch in reversed(skip_channels):
            decoder.append(DecoderStage(in_ch, skip_ch, skip_ch))
            in_ch = skip_ch
        self.decoder = nn.ModuleList(decoder)
        self.head = nn.Conv2d(stem_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.stem_spatial(self.stem(x))
        skips.append(h)
        for stage in self.encoder:
            h = stage(h)
            skips.append(h)
        h = self.bottleneck(h)
        # skips[:-1] is every map except the bottleneck resolution.
        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)
        logits = self.head(h)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def main() -> None:
    model = MiniNetSeg()
    n = count_parameters(model)
    print(f"MiniNetSeg parameters: {n:,} ({n / 1e6:.3f}M)")
    sample = torch.zeros(1, model.in_channels, 64, 64)
    with torch.no_grad():
        logits = model(sample)
    print(f"forward {tuple(sample.shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
