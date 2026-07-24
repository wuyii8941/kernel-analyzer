#!/usr/bin/env python
"""Revision 2: original-candidate repairs for fused SiLU-multiply calls."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = "triton_poi_fused__unsafe_view_mul_silu_14"


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    in_out, other = values[:2]
    if in_out.numel() != other.numel():
        raise RuntimeError("fused SiLU-multiply buffers do not have equal numel")
    with torch.no_grad():
        other_view = other.reshape(in_out.shape)
        activated = torch.nn.functional.silu(in_out)
        in_out.copy_(activated * other_view)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-silu-mul-repair.v0.2",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_SILU_MUL_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused SiLU-multiply invocation, not separate SiLU and multiply attribution",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
