#!/usr/bin/env python
"""Original-candidate repairs for key head-repeat, layout and scaling kernels."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = (
    "triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_"
    "expand_mean_mul_pow_rsqrt_sin_transpose_unsqueeze_view_7"
)


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    if source.ndim != 4 or destination.ndim != 4:
        raise RuntimeError("key source and destination must both be four-dimensional")
    batch, key_heads, sequence, width = source.shape
    expected = (batch, key_heads * 2, width, sequence)
    if key_heads != 8 or width != 128 or tuple(destination.shape) != expected:
        raise RuntimeError(
            f"unexpected key layout source/destination {source.shape} -> {destination.shape}"
        )
    expected_stride = (key_heads * 2 * width * sequence, width * sequence, sequence, 1)
    if tuple(destination.stride()) != expected_stride:
        raise RuntimeError(f"unexpected key layout destination stride {destination.stride()}")

    with torch.no_grad():
        repeated = source.repeat_interleave(2, dim=1)
        repaired = repeated.permute(0, 1, 3, 2) * 0.29730177875068026
        destination.copy_(repaired)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-key-layout-repair.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_KEY_LAYOUT_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused key head-repeat plus transpose/layout plus scaling invocation",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
