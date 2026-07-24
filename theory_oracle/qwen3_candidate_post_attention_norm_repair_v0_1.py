#!/usr/bin/env python
"""Original-candidate repairs for repeated post-attention residual-RMSNorm kernels."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = "triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12"


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    residual, attention_output, weight, output = values[:4]
    if residual.numel() != attention_output.numel() or residual.numel() != output.numel():
        raise RuntimeError("post-attention RMSNorm buffers do not have equal numel")
    with torch.no_grad():
        shape = output.shape
        hidden = residual.reshape(shape) + attention_output.reshape(shape).float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        normalized = weight.reshape((1,) * (hidden.ndim - 1) + (-1,)) * (
            hidden * torch.rsqrt(variance + 1e-6)
        )
        output.copy_(normalized.to(output.dtype))


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-post-attention-norm-repair.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_POST_ATTENTION_NORM_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused attention-residual plus RMSNorm invocation, not constituent attribution",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
