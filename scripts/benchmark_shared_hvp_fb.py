"""Benchmark shared all-block HVP screening on real F+B semantic cuts.

This is an engineering feasibility benchmark, not a model correctness result.
The cuts exercise loss-head, RMSNorm, and attention forward/backward semantics,
then project the resulting parameter gradient into one scalar effective-update
coordinate.  All declared local injection blocks are screened together.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import statistics
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from kernel_analyzer.bias_oracle_feasibility import shared_block_hvp_sketch


def _peak_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    # Linux reports KiB. This is process peak RSS, hence conservative and
    # cumulative across cuts rather than an allocation-perfect measurement.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _factor(block: torch.Tensor, *, rank: int, scale: float, generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn(
            block.numel(), rank, dtype=block.dtype, device=block.device,
            generator=generator,
        )
        * (scale / math.sqrt(rank))
    )


def _loss_head(device: torch.device, dtype: torch.dtype, generator: torch.Generator):
    tokens, hidden, vocabulary = 16, 32, 128
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(hidden, vocabulary, device=device, dtype=dtype, generator=generator, requires_grad=True)
    local = torch.zeros(tokens, vocabulary, device=device, dtype=dtype, requires_grad=True)
    targets = torch.arange(tokens, device=device) % vocabulary
    loss = F.cross_entropy(x @ weight + local, targets)
    return loss, [weight], [local]


def _rmsnorm(device: torch.device, dtype: torch.dtype, generator: torch.Generator):
    tokens, hidden = 32, 64
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=generator)
    upstream = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(hidden, device=device, dtype=dtype, generator=generator, requires_grad=True)
    local = torch.zeros_like(x, requires_grad=True)
    value = x + local
    normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
    loss = (normalized * weight * upstream).sum() / tokens
    return loss, [weight], [local]


def _attention(device: torch.device, dtype: torch.dtype, generator: torch.Generator):
    tokens, hidden, heads = 16, 32, 4
    head_dim = hidden // heads
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=generator)
    weights = [
        torch.randn(hidden, hidden, device=device, dtype=dtype, generator=generator, requires_grad=True)
        for _ in range(3)
    ]
    q, k, v = [
        (x @ weight).reshape(tokens, heads, head_dim).transpose(0, 1)
        for weight in weights
    ]
    score_error = torch.zeros(heads, tokens, tokens, device=device, dtype=dtype, requires_grad=True)
    probability_error = torch.zeros_like(score_error, requires_grad=True)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim) + score_error
    probability = torch.softmax(scores, dim=-1) + probability_error
    output = (probability @ v).transpose(0, 1).reshape(tokens, hidden)
    upstream = torch.randn(output.shape, device=device, dtype=dtype, generator=generator)
    loss = (output * upstream).sum() / tokens
    return loss, weights, [score_error, probability_error]


CUTS: dict[str, Callable[..., Any]] = {
    "loss_head_cross_entropy": _loss_head,
    "rmsnorm_backward": _rmsnorm,
    "attention_backward": _attention,
}


def _ordinary_fb(builder: Callable[..., Any], device: torch.device, dtype: torch.dtype, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss, parameters, _ = builder(device, dtype, generator)
    torch.autograd.grad(loss, parameters, create_graph=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "wall_seconds": time.perf_counter() - start,
        "peak_memory_bytes": _peak_memory(device),
    }


def _shared_hvp(
    builder: Callable[..., Any], device: torch.device, dtype: torch.dtype,
    seed: int, probes: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss, parameters, blocks = builder(device, dtype, generator)
    parameter_gradients = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
    output_codes = [
        torch.randn(gradient.shape, device=device, dtype=dtype, generator=generator)
        for gradient in parameter_gradients
    ]
    scalar_response = sum(
        (gradient * code).sum()
        for gradient, code in zip(parameter_gradients, output_codes, strict=True)
    )
    means = [
        torch.randn(block.shape, device=device, dtype=dtype, generator=generator) * 1e-4
        for block in blocks
    ]
    factors = [
        _factor(block, rank=4, scale=1e-3, generator=generator)
        for block in blocks
    ]
    sketch = shared_block_hvp_sketch(
        scalar_response, blocks, means, factors, probes=probes, seed=seed
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    finite = all(math.isfinite(value) for value in (
        *sketch.transported_mean_projections,
        *sketch.curvature_projections,
        *sketch.curvature_standard_errors,
    ))
    return {
        "status": "SUPPORTED" if finite else "INVALID_NONFINITE",
        "wall_seconds": elapsed,
        "peak_memory_bytes": _peak_memory(device),
        "declared_block_count": len(blocks),
        "declared_block_coordinates": [block.numel() for block in blocks],
        "parameter_coordinates": sum(parameter.numel() for parameter in parameters),
        "sketch": sketch.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(
        "results/property/bias_oracle_feasibility/shared_hvp_fb_benchmark.json"
    ))
    args = parser.parse_args()
    if args.probes < 1 or args.repeats < 1:
        raise SystemExit("--probes and --repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # Initialize the context before resetting allocator statistics. Some
        # torch/CUDA combinations reject reset_peak_memory_stats pre-init.
        torch.empty(1, device=device)
    torch.set_num_threads(1)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    rows = []
    for index, (cut_id, builder) in enumerate(CUTS.items()):
        seed = 20260820 + index
        # Warm both paths once so lazy library initialization is not charged
        # exclusively to the ordinary or HVP arm.
        _ordinary_fb(builder, device, dtype, seed)
        try:
            _shared_hvp(builder, device, dtype, seed, args.probes)
        except (RuntimeError, ValueError):
            pass
        ordinary_runs = [
            _ordinary_fb(builder, device, dtype, seed) for _ in range(args.repeats)
        ]
        ordinary = {
            "wall_seconds_median": statistics.median(
                row["wall_seconds"] for row in ordinary_runs
            ),
            "wall_seconds_min": min(row["wall_seconds"] for row in ordinary_runs),
            "peak_memory_bytes": max(row["peak_memory_bytes"] for row in ordinary_runs),
            "repeats": args.repeats,
        }
        try:
            screened_runs = [
                _shared_hvp(builder, device, dtype, seed, args.probes)
                for _ in range(args.repeats)
            ]
            screened = screened_runs[-1]
            screened["wall_seconds_median"] = statistics.median(
                row["wall_seconds"] for row in screened_runs
            )
            screened["wall_seconds_min"] = min(
                row["wall_seconds"] for row in screened_runs
            )
            screened["peak_memory_bytes"] = max(
                row["peak_memory_bytes"] for row in screened_runs
            )
            screened["repeats"] = args.repeats
        except (RuntimeError, ValueError) as exc:
            screened = {
                "status": "UNSUPPORTED_DOUBLE_BACKWARD",
                "reason": str(exc),
            }
        if screened["status"] == "SUPPORTED":
            screened["wall_time_multiple_vs_ordinary_fb"] = (
                screened["wall_seconds_median"] / ordinary["wall_seconds_median"]
            )
        rows.append({
            "cut_id": cut_id,
            "ordinary_fb": ordinary,
            "shared_hvp_screen": screened,
        })
    output = {
        "schema": "kernel-analyzer-shared-hvp-fb-benchmark-v1",
        "scientific_role": "ENGINEERING_FEASIBILITY_NOT_MODEL_CORRECTNESS",
        "device": str(device),
        "dtype": str(dtype),
        "probes": args.probes,
        "cuts": rows,
        "all_cuts_support_double_backward": all(
            row["shared_hvp_screen"]["status"] == "SUPPORTED" for row in rows
        ),
        "cost_scope": (
            "probe count is shared across all declared blocks inside one graph; "
            "this benchmark does not establish compiled/custom-kernel coverage"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
