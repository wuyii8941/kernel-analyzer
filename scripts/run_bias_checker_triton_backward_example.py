#!/usr/bin/env python3
"""Run a callable-level backward bias check for an unfamiliar Triton kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from kernel_analyzer import check_bias


@triton.jit
def identity_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


class TritonCallableWithBiasedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(value)
        identity_kernel[(triton.cdiv(value.numel(), 128),)](
            value, output, value.numel(), BLOCK=128, num_warps=4
        )
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        # Deliberate, explicit backward positive control.  The Triton forward
        # is identity; the callable's backward reports a 1% scaling bias.
        return (gradient * 1.01,)


def candidate(value: torch.Tensor) -> torch.Tensor:
    return TritonCallableWithBiasedBackward.apply(value)


def reference(value: torch.Tensor) -> torch.Tensor:
    return value


def make_inputs(index: int) -> tuple[torch.Tensor]:
    value = torch.linspace(-1.0, 1.0, 64, device="cuda")
    return (value + 0.01 * index,)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    report = check_bias(
        candidate,
        reference,
        make_inputs,
        samples=8,
        calibration_samples=4,
        check_backward=True,
    )
    compact = {
        "schema": "kernel-analyzer-triton-bias-check-backward-example-v1",
        "status": report["status"],
        "measurement_status": report["measurement_status"],
        "output_status": report["stages"]["OUTPUT"]["decision"],
        "backward_status": report["stages"]["BACKWARD"]["decision"],
        "output_rms": report["stages"]["OUTPUT"].get("total_rms"),
        "backward_rms": report["stages"]["BACKWARD"].get("total_rms"),
        "output_samples": report["stages"]["OUTPUT"]["sample_count"],
        "backward_samples": report["stages"]["BACKWARD"]["sample_count"],
        "scope": "DECLARED_INPUT_DISTRIBUTION_ONLY",
        "note": "The backward scaling is a deliberate positive control, not a natural training case.",
    }
    output = Path("results/property/bias_checker_triton_backward_20260914.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_bias_checker_backward")
    main()
