#!/usr/bin/env python
"""Original-candidate repairs for Qwen3's repeated fused SiLU-multiply kernel."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = "triton_poi_fused__unsafe_view_mul_silu_14"


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    in_out, other = values[:2]
    with torch.no_grad():
        # Preserve the eager semantic boundary: SiLU returns FP16 before the
        # following FP16 multiply, whereas the fused candidate may defer it.
        activated = torch.nn.functional.silu(in_out)
        in_out.copy_(activated * other)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-silu-mul-repair.v0.1",
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
