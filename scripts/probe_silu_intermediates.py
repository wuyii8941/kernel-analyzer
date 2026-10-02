#!/usr/bin/env python3
"""Probe the first differing intermediate in the reviewed SiLU expression.

This source-level probe uses synthetic operands and stores selected FP32
intermediates before the final BF16 write.  It is a mechanism probe, not a
natural-state or training-population result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.triton_helpers import libdevice


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/root_cause_closure_v1/silu_intermediate_probe_v1.json"


@triton.jit
def _probe(
    grad, gate, up, exp_out, sum_out, sigmoid_out, product_out, derivative_out,
    n: tl.constexpr, block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    g = tl.load(grad + index, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(gate + index, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + index, mask=mask, other=0.0).to(tl.float32)
    one = tl.full([1], 1.0, tl.float32)
    exp_value = libdevice.exp(-x)
    denominator = exp_value + one
    sigmoid = one / denominator
    product = (g * u) * sigmoid
    derivative_factor = x * (one - sigmoid) + one
    derivative = product * derivative_factor
    tl.store(exp_out + index, exp_value, mask=mask)
    tl.store(sum_out + index, denominator, mask=mask)
    tl.store(sigmoid_out + index, sigmoid, mask=mask)
    tl.store(product_out + index, product, mask=mask)
    tl.store(derivative_out + index, derivative, mask=mask)


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int | bool]:
    delta = candidate.double() - reference.double()
    norm = float(torch.linalg.vector_norm(delta).item())
    ref_norm = float(torch.linalg.vector_norm(reference.double()).item())
    return {
        "exact": bool(torch.equal(candidate, reference)),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": norm / ref_norm if ref_norm else None,
        "max_abs": float(delta.abs().max().item()),
        "signed_mean": float(delta.mean().item()),
    }


def run(seed: int, scale: float) -> dict[str, object]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n = 1_572_864
    grad = torch.randn(n, device=device, generator=generator, dtype=torch.bfloat16)
    gate = (torch.randn(n, device=device, generator=generator, dtype=torch.bfloat16) * scale).to(torch.bfloat16)
    up = torch.randn(n, device=device, generator=generator, dtype=torch.bfloat16)
    outputs = [torch.empty(n, device=device, dtype=torch.float32) for _ in range(5)]
    _probe[(triton.cdiv(n, 1024),)](
        grad, gate, up, *outputs, n=n, block=1024
    )
    torch.cuda.synchronize()
    g = grad.float()
    x = gate.float()
    u = up.float()
    one = torch.ones_like(x)
    exp_value = torch.exp(-x)
    denominator = exp_value + one
    sigmoid = one / denominator
    product = (g * u) * sigmoid
    derivative = product * (x * (one - sigmoid) + one)
    names = ("exp", "denominator", "sigmoid", "product", "derivative")
    references = (exp_value, denominator, sigmoid, product, derivative)
    return {
        "seed": seed,
        "scale": scale,
        "intermediates": {
            name: _metrics(output, reference)
            for name, output, reference in zip(names, outputs, references)
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [run(seed, scale) for seed, scale in ((20260940, 1.0), (20260941, 4.0), (20260942, 8.0))]
    first_difference = {}
    names = ("exp", "denominator", "sigmoid", "product", "derivative")
    for name in names:
        first_difference[name] = any(
            row["intermediates"][name]["nonzero_coordinates"] > 0 for row in rows
        )
    payload = {
        "schema": "kernel-analyzer-silu-intermediate-probe-v1",
        "status": "COMPLETE_SYNTHETIC_INTERMEDIATE_SOURCE_PROBE",
        "source_boundary": "reviewed_gated_silu_backward_expression",
        "input_policy": "synthetic BF16 operands; not a natural-state or training-population claim",
        "rows": rows,
        "first_difference_candidates": first_difference,
        "interpretation": (
            "The first nonzero intermediate identifies where the reviewed Triton expression "
            "departs from the direct FP32 Torch expression on the tested synthetic operands. "
            "It separates exp implementation and downstream arithmetic from the final BF16 "
            "write, but does not prove that the same component dominates the natural training case."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
