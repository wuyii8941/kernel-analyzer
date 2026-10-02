#!/usr/bin/env python3
"""Probe the reviewed fused-RoPE arithmetic choices on identical operands.

This is a source diagnostic only.  It does not claim that synthetic operands
represent a training-state population or that a nonzero output difference is
the cause of the recorded Ministral trajectory.  The probe keeps the formula,
storage and indexing fixed while changing only the trigonometric implementation
used by the Triton program.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernel_analyzer.rotary_trig_variants import fp64_reference, rotary_variant


@triton.jit
def _order_variant_kernel(
    query, frequency, position, output, n: tl.constexpr, block: tl.constexpr,
    mode: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    dim = index % 128
    token = (index // 128) % 128
    base = index - dim
    value = tl.load(query + index, mask=mask, other=0.0).to(tl.float32)
    partner_dim = tl.where(dim < 64, dim + 64, dim - 64)
    partner = tl.load(query + base + partner_dim, mask=mask, other=0.0).to(tl.float32)
    rotated = tl.where(dim < 64, -partner, partner)
    phase = tl.load(frequency + (dim % 64)).to(tl.float32) * tl.load(position + token).to(tl.float32)
    cosine = tl.cos(phase)
    sine = tl.sin(phase)
    scale = 1.0 + 0.1 * tl.log(1.0 + tl.floor(tl.load(position + token).to(tl.float32) / 16384.0))
    first = value * cosine
    second = rotated * sine
    combined = tl.where(mode == 0, (first + second) * scale,
                        tl.where(mode == 1, first * scale + second * scale,
                                 (first + second).to(tl.bfloat16).to(tl.float32) * scale))
    tl.store(output + index, combined, mask=mask)


def order_variant(
    query: torch.Tensor, frequency: torch.Tensor, position: torch.Tensor, mode: int
) -> torch.Tensor:
    output = torch.empty_like(query)
    block = 256
    _order_variant_kernel[(triton.cdiv(query.numel(), block),)](
        query, frequency, position, output, n=query.numel(), block=block, mode=mode
    )
    return output


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int | bool]:
    delta = candidate.float() - reference.float()
    norm = float(torch.linalg.vector_norm(delta).item())
    ref_norm = float(torch.linalg.vector_norm(reference.float()).item())
    return {
        "exact": bool(torch.equal(candidate, reference)),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "total_effect_l2": norm,
        "reference_l2": ref_norm,
        "relative_l2": norm / ref_norm if ref_norm else math.inf,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.double().mean().item()) if delta.numel() else 0.0,
    }


def fp32_reference(
    query: torch.Tensor, frequency: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    """Evaluate the same expression with explicit FP32 intermediates."""
    query32 = query.float()
    phase = position.float()[:, None] * frequency.float()[None, :]
    cosine = torch.cat((torch.cos(phase), torch.cos(phase)), dim=-1)
    sine = torch.cat((torch.sin(phase), torch.sin(phase)), dim=-1)
    rotated = torch.cat((-query32[..., 64:], query32[..., :64]), dim=-1)
    scale = 1.0 + 0.1 * torch.log(
        1.0 + torch.floor(position.float() / 16384.0)
    )
    return ((query32 * cosine[None] + rotated * sine[None]) * scale[None, :, None]).to(
        torch.bfloat16
    )


def run(device: torch.device, seed: int, position_scale: float) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # The reviewed kernel has [32, 128, 128] BF16 query storage, FP32
    # frequencies and int64 positions.  Position values are deliberately
    # nonuniform so both low and high phase regimes are exercised.
    query = torch.randn((32, 128, 128), device=device, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    frequency = torch.linspace(0.001, 1.2, 64, device=device, dtype=torch.float32)
    position = torch.arange(128, device=device, dtype=torch.int64)
    if position_scale != 1.0:
        position = torch.round(position.float() * position_scale).to(torch.int64)
    native = rotary_variant(query, frequency, position, trig="tl_math")
    libdevice = rotary_variant(query, frequency, position, trig="libdevice")
    scale_terms = order_variant(query, frequency, position, mode=1)
    rounded_sum = order_variant(query, frequency, position, mode=2)
    fp32 = fp32_reference(query, frequency, position)
    high_precision = fp64_reference(query, frequency, position)
    return {
        "position_scale": position_scale,
        "position_min": int(position.min().item()),
        "position_max": int(position.max().item()),
        "tl_math_vs_libdevice": metrics(native, libdevice),
        "tl_math_vs_scale_terms_order": metrics(native, scale_terms),
        "tl_math_vs_rounded_sum_materialization": metrics(native, rounded_sum),
        "tl_math_vs_fp32_formula": metrics(native, fp32),
        "tl_math_vs_fp64_formula": metrics(native, high_precision),
        "libdevice_vs_fp64_formula": metrics(libdevice, high_precision),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the Triton source probe")
    torch.cuda.set_device(device)
    rows = [run(device, args.seed + index, scale) for index, scale in enumerate((1.0, 128.0, 1024.0))]
    payload = {
        "schema": "rotary-arithmetic-source-probe-v1",
        "status": "COMPLETE",
        "formula": "fused RoPE plus position scaling, BF16 output",
        "source_variants": ["tl_math", "libdevice"],
        "reference": "FP64 formula evaluated then BF16 materialization",
        "input_policy": "synthetic identical operands; not a training-state population",
        "rows": rows,
        "interpretation": (
            "This probe separates trigonometric implementation effects on the same operands. "
            "It cannot by itself attribute the recorded Ministral training difference, which also "
            "depends on fused operation order, materialization and real model operands."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("output must be inside the repository")
        if output.exists():
            raise ValueError("refusing to overwrite an existing probe result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
