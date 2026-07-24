#!/usr/bin/env python
"""Original-candidate repairs for fused key RMSNorm and rotary kernels."""

from __future__ import annotations

from typing import Any

from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


KERNEL = (
    "triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_"
    "mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4"
)


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    destination, projection, norm_weight, inv_freq = values[:4]
    if destination.ndim != 4:
        raise RuntimeError(f"expected four-dimensional destination, got {destination.shape}")
    batch, heads, sequence, width = destination.shape
    if heads != 8 or width != 128:
        raise RuntimeError(f"unexpected key destination shape {destination.shape}")
    if projection.numel() != batch * sequence * heads * width:
        raise RuntimeError("key projection size does not match destination")
    if norm_weight.numel() != width or inv_freq.numel() * 2 != width:
        raise RuntimeError("key norm or rotary frequency width mismatch")

    with torch.no_grad():
        key = projection.reshape(batch, sequence, heads, width).float()
        variance = key.pow(2).mean(-1, keepdim=True)
        key = key * torch.rsqrt(variance + 1e-6)
        key = key * norm_weight.reshape(1, 1, 1, width)
        key = key.permute(0, 2, 1, 3)

        positions = torch.arange(sequence, device=key.device, dtype=torch.float32)
        frequencies = positions[:, None] * inv_freq.float().reshape(1, -1)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().reshape(1, 1, sequence, width)
        sine = angles.sin().reshape(1, 1, sequence, width)
        rotate_half = torch.cat((-key[..., width // 2 :], key[..., : width // 2]), dim=-1)
        repaired = key * cosine + rotate_half * sine
        destination.copy_(repaired)


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-key-norm-rotary-repair.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_KEY_NORM_ROTARY_REPAIR",
            "kernel_family": KERNEL,
            "expected_family_calls": 28,
            "selected_call_indices": [0, 14, 27],
            "claim_limits": [
                "generated fused key RMSNorm plus rotary invocation, not constituent attribution",
                "repair direction relative to eager, not correctness",
                "repair only; no injection or sufficiency claim",
                "selected matched state only",
            ],
        },
        repair,
    )
