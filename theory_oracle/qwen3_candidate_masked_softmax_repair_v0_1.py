#!/usr/bin/env python
"""Original-candidate repairs for Qwen3's repeated masked safe-softmax kernel."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = "triton_red_fused__safe_softmax_add_prepare_softmax_online_view_8"


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    scores, mask = values[:2]
    with torch.no_grad():
        masked_scores = scores + mask
        probabilities = torch.ops.aten._safe_softmax.default(masked_scores, -1, None)
        scores.copy_(probabilities)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-masked-softmax-repair.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_MASKED_SOFTMAX_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused mask-add and safe-softmax invocation, not separate add/reduction attribution",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
