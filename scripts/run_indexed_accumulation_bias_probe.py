#!/usr/bin/env python3
"""Run a small, hash-free indexed-accumulation bias probe.

The probe compares CUDA ``index_add`` with an explicit input-order reference
on the same operands.  It is deliberately a declared stress distribution,
not a claim about a natural model distribution: repeated destinations and a
large positive dynamic range make non-associative accumulation visible.
Both BF16 and FP32 modes are useful.  BF16 shows the larger practical effect;
FP32 demonstrates that the order effect is not intrinsically a low-precision
phenomenon.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from kernel_analyzer.bias_checker import check_bias


def make_inputs(*, device: str, dtype: torch.dtype, order: str, width: int, groups: int):
    def factory(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = 2048 if dtype == torch.bfloat16 else 4096
        destinations = torch.arange(n, device=device, dtype=torch.long) % groups
        span = 20.0 if dtype == torch.float32 else 8.0
        exponents = torch.linspace(
            -span + 0.03 * index,
            span + 0.03 * index,
            n,
            device=device,
            dtype=torch.float32,
        )
        if order == "descending":
            exponents = exponents.flip(0)
        elif order == "alternating":
            exponents = exponents[::2].repeat_interleave(2)[:n]
        values = torch.pow(torch.tensor(2.0, device=device), exponents)
        column_scale = 1.0 + 0.03125 * torch.arange(width, device=device, dtype=torch.float32)
        source = (values[:, None] * column_scale[None, :]).to(dtype)
        return source, destinations

    return factory


def indexed_candidate(source: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        (int(destinations.max().item()) + 1, source.shape[1]),
        device=source.device,
        dtype=source.dtype,
    )
    return output.index_add(0, destinations, source)


def ordered_reference(source: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        (int(destinations.max().item()) + 1, source.shape[1]),
        device=source.device,
        dtype=source.dtype,
    )
    for row in range(source.shape[0]):
        output[destinations[row]] = output[destinations[row]] + source[row]
    return output


def compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    direction = stage.get("direction", {})
    interval = direction.get("confirmation_projection_interval")
    return {
        "decision": stage.get("decision"),
        "sample_count": stage.get("sample_count"),
        "total_rms": stage.get("total_rms"),
        "aligned_ratio_of_sums": stage.get("aligned_ratio_of_sums"),
        "direction_interval": interval,
        "sign_prevalence": direction.get("sign_prevalence"),
        "errors": stage.get("errors", []),
    }


def run_probe(*, device: str, dtype: torch.dtype, order: str, samples: int, backward: bool) -> dict[str, Any]:
    report = check_bias(
        indexed_candidate,
        ordered_reference,
        make_inputs(device=device, dtype=dtype, order=order, width=8, groups=8),
        samples=samples,
        calibration_samples=samples // 2,
        check_backward=backward,
        alpha=0.05,
    )
    return {
        "dtype": str(dtype),
        "order": order,
        "samples": samples,
        "check_backward": backward,
        "status": report.get("status"),
        "measurement_status": report.get("measurement_status"),
        "scope": report.get("scope"),
        "stages": {
            name: compact_stage(stage)
            for name, stage in report.get("stages", {}).items()
        },
        "limitations": report.get("limitations", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--order", choices=("ascending", "descending", "alternating", "all"), default="all")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for this probe")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    orders = ("ascending", "descending", "alternating") if args.order == "all" else (args.order,)
    result = {
        "schema": "indexed-accumulation-bias-probe-v1",
        "candidate": "CUDA index_add with repeated destinations",
        "reference": "explicit input-order accumulation with the same dtype",
        "root_cause_hypothesis": "non-associative accumulation and implementation reduction order",
        "claim_scope": "declared stress input distribution only; not a natural training-population claim",
        "results": [
            run_probe(device=args.device, dtype=dtype, order=order, samples=args.samples, backward=args.backward)
            for order in orders
        ],
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
