#!/usr/bin/env python
"""Original-candidate repairs for fused query RMSNorm and rotary kernels."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = (
    "triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_"
    "mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2"
)


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    destination, projection, norm_weight, inv_freq = values[:4]
    if destination.ndim != 4:
        raise RuntimeError(f"expected four-dimensional destination, got {destination.shape}")
    batch, heads, sequence, width = destination.shape
    if heads != 16 or width != 128:
        raise RuntimeError(f"unexpected query destination shape {destination.shape}")
    if projection.numel() != batch * sequence * heads * width:
        raise RuntimeError("query projection size does not match destination")
    if norm_weight.numel() != width or inv_freq.numel() * 2 != width:
        raise RuntimeError("query norm or rotary frequency width mismatch")

    with torch.no_grad():
        query = projection.reshape(batch, sequence, heads, width).float()
        variance = query.pow(2).mean(-1, keepdim=True)
        query = query * torch.rsqrt(variance + 1e-6)
        query = query * norm_weight.reshape(1, 1, 1, width)
        query = query.permute(0, 2, 1, 3)

        positions = torch.arange(sequence, device=query.device, dtype=torch.float32)
        frequencies = positions[:, None] * inv_freq.float().reshape(1, -1)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().reshape(1, 1, sequence, width)
        sine = angles.sin().reshape(1, 1, sequence, width)
        rotate_half = torch.cat((-query[..., width // 2 :], query[..., : width // 2]), dim=-1)
        repaired = (query * cosine + rotate_half * sine) * 0.29730177875068026
        destination.copy_(repaired)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-query-norm-rotary-repair.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_QUERY_NORM_ROTARY_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused query RMSNorm plus rotary invocation, not constituent attribution",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
