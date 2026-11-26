#!/usr/bin/env python3
"""Measure spatial signal vs magnitude at init (input-blind diagnosis).

Builds a BandTransformer at its default initialisation and prints the ratio
of spatial std to RMS after the tokenizer (+ IndexBandPE), at the SegHead
input (after spatial blocks), and on the logits. A high shared per-band
offset with LayerNorm + large PE makes that ratio small; standardisation
and a smaller ``--pe-init-std`` raise it.

Default input is a synthetic cube with a realistic offset
(``0.23 + 0.14 * noise``, shape ``(B, 224, 128, 128)``). Pass ``--data-root``
to load real train cubes instead (first ``--max-images``, optional
``--downsample``).

Examples::

    PYTHONPATH=src python scripts/diag_input_blind.py
    PYTHONPATH=src python scripts/diag_input_blind.py --standardize --pe-init-std 0.02
    PYTHONPATH=src python scripts/diag_input_blind.py --data-root "$SW_ROOT" --downsample 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sw.data import SpectralWasteDataset
from sw.models.transformer import BandTransformer, init_seg_head_prior_bias
from sw.standardize import BandStandardizer, compute_band_stats
from sw.train import prepare_sample


def spatial_std_rms_ratio(x: torch.Tensor, spatial_dims) -> float:
    """Mean spatial std over non-spatial axes, divided by the tensor RMS."""
    rms = x.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)
    std = x.detach().float().std(dim=spatial_dims, unbiased=False).mean()
    return float((std / rms).item())


def synthetic_offset_cube(
    batch: int = 1,
    n_bands: int = 224,
    height: int = 128,
    width: int = 128,
    offset: float = 0.23,
    spatial_std: float = 0.14,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``offset + spatial_std * N(0,1)``, clamped to a soft [0, 1]-ish range."""
    noise = torch.randn(
        batch, n_bands, height, width, generator=generator, dtype=torch.float32
    )
    return offset + spatial_std * noise


@torch.no_grad()
def measure_input_blind_ratios(
    model: BandTransformer,
    x: torch.Tensor,
) -> dict[str, float]:
    """Forward hooks: tokenizer out, head input, logits spatial-std / RMS."""
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def on_tokenizer(_module, _inputs, output):
        tokens, _grid = output
        captured["tokenizer"] = tokens.detach()

    def on_head(_module, inputs, _output):
        # SegHead.forward(tokens, h, w, out_size)
        captured["head_in"] = inputs[0].detach()

    h_tok = model.tokenizer.register_forward_hook(on_tokenizer)
    h_head = model.head.register_forward_hook(on_head)
    try:
        logits = model(x)
    finally:
        h_tok.remove()
        h_head.remove()

    tok = captured["tokenizer"]  # (B, N, G, D)
    head_in = captured["head_in"]  # (B, N, D)
    return {
        "tokenizer_spatial_std_rms": spatial_std_rms_ratio(tok, spatial_dims=1),
        "head_in_spatial_std_rms": spatial_std_rms_ratio(head_in, spatial_dims=1),
        "logits_spatial_std_rms": spatial_std_rms_ratio(logits, spatial_dims=(-2, -1)),
        "input_mean": float(x.mean().item()),
        "input_spatial_std": float(x.std(dim=(-2, -1), unbiased=False).mean().item()),
        "logits_abs_mean": float(logits.abs().mean().item()),
    }


def build_diag_model(
    *,
    pe_init_std: float = 1.0,
    head_prior_bias: bool = False,
    seed: int = 0,
) -> BandTransformer:
    torch.manual_seed(seed)
    model = BandTransformer(
        depth_spectral=2,
        depth_spatial=6,
        dim=64,
        heads=4,
        mlp_ratio=4.0,
        num_classes=7,
        n_bands=224,
        group=16,
        patch=8,
        pe_init_std=pe_init_std,
    )
    if head_prior_bias:
        init_seg_head_prior_bias(model.head)
    return model


def _load_real_batch(
    data_root: str,
    *,
    max_images: int,
    downsample: int,
    standardize: bool,
) -> tuple[torch.Tensor, BandStandardizer | None]:
    ds = SpectralWasteDataset(data_root, split="train", random_flip=False)
    n = min(len(ds), max_images)
    if n < 1:
        raise RuntimeError("train split is empty")
    cubes = []
    for i in range(n):
        cube, mask = ds[i]
        cube, _ = prepare_sample(
            cube, mask, train=False, crop_size=None, downsample=downsample
        )
        cubes.append(cube)
    batch = torch.stack(cubes, dim=0)
    standardizer = None
    if standardize:
        # Stats from the same train cubes we diagnose (still train-only).
        class _ListDS(torch.utils.data.Dataset):
            def __len__(self):
                return len(cubes)

            def __getitem__(self, index):
                return cubes[index], torch.zeros(1, dtype=torch.int64)

        mean, std = compute_band_stats(_ListDS(), split="train")
        standardizer = BandStandardizer(mean, std)
        batch = standardizer(batch)
    return batch, standardizer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--offset", type=float, default=0.23)
    p.add_argument("--spatial-std", type=float, default=0.14)
    p.add_argument("--data-root", default=None, help="Optional real SW_ROOT; else synthetic.")
    p.add_argument("--max-images", type=int, default=2)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--pe-init-std", type=float, default=1.0)
    p.add_argument("--head-prior-bias", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return p


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = _build_parser().parse_args(argv)
    model = build_diag_model(
        pe_init_std=args.pe_init_std,
        head_prior_bias=args.head_prior_bias,
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = model.to(device)

    if args.data_root is not None:
        x, _ = _load_real_batch(
            args.data_root,
            max_images=args.max_images,
            downsample=args.downsample,
            standardize=args.standardize,
        )
        x = x.to(device)
    else:
        g = torch.Generator()
        g.manual_seed(args.seed)
        x = synthetic_offset_cube(
            batch=args.batch_size,
            height=args.height,
            width=args.width,
            offset=args.offset,
            spatial_std=args.spatial_std,
            generator=g,
        ).to(device)
        if args.standardize:
            # Per-band stats on this synthetic batch (stand-in for train-only).
            mean = x.mean(dim=(0, 2, 3))
            std = x.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
            x = BandStandardizer(mean, std)(x)

    ratios = measure_input_blind_ratios(model, x)
    print(
        f"input mean={ratios['input_mean']:.4f}  "
        f"spatial_std={ratios['input_spatial_std']:.4f}"
    )
    print(
        f"tokenizer+PE  spatial_std/rms = {ratios['tokenizer_spatial_std_rms']:.4f}"
    )
    print(f"head input    spatial_std/rms = {ratios['head_in_spatial_std_rms']:.4f}")
    print(
        f"logits        spatial_std/rms = {ratios['logits_spatial_std_rms']:.4f}  "
        f"|logits| mean={ratios['logits_abs_mean']:.4f}"
    )
    print(
        f"flags: standardize={args.standardize}  "
        f"pe_init_std={args.pe_init_std}  head_prior_bias={args.head_prior_bias}"
    )
    return ratios


if __name__ == "__main__":
    main()
