#!/usr/bin/env python
"""Separate original-candidate repairs for seven singleton Qwen3 kernels."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_multikernel_runner_v0_1 import run_campaign


def embedding_norm(torch: Any, values: tuple[Any, ...]) -> None:
    token_ids, embedding_weight, norm_weight, hidden_out, norm_out = values[:5]
    with torch.no_grad():
        ids = token_ids.reshape(hidden_out.shape[:-1])
        hidden = torch.nn.functional.embedding(ids, embedding_weight)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        normalized = norm_weight.reshape((1,) * (hidden.ndim - 1) + (-1,)) * (hidden * torch.rsqrt(variance + 1e-6))
        hidden_out.copy_(hidden.reshape(hidden_out.shape))
        norm_out.copy_(normalized.to(norm_out.dtype).reshape(norm_out.shape))


def causal_mask(torch: Any, values: tuple[Any, ...]) -> None:
    attention_mask, output = values[:2]
    batch, sequence = output.shape[0], output.shape[-1]
    with torch.no_grad():
        valid_keys = attention_mask.reshape(batch, sequence).bool()
        positions = torch.arange(sequence, device=output.device)
        causal = positions[None, :] <= positions[:, None]
        allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
        replacement = torch.where(
            allowed,
            torch.zeros((), dtype=output.dtype, device=output.device),
            torch.full((), float("-inf"), dtype=output.dtype, device=output.device),
        )
        output.copy_(replacement)


def zero_safe_softmax_buffer(torch: Any, values: tuple[Any, ...]) -> None:
    with torch.no_grad():
        values[0].zero_()


def final_rsqrt(torch: Any, values: tuple[Any, ...]) -> None:
    output, residual, attention_output, mlp_output = values[:4]
    if not (residual.numel() == attention_output.numel() == mlp_output.numel()):
        raise RuntimeError("final RMSNorm inputs do not have equal numel")
    with torch.no_grad():
        hidden = residual + attention_output.reshape(residual.shape).float() + mlp_output.reshape(residual.shape).float()
        rsqrt = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
        output.copy_(rsqrt.reshape(output.shape))


def cast_copy(torch: Any, values: tuple[Any, ...]) -> None:
    source, output = values[:2]
    if source.numel() != output.numel():
        raise RuntimeError("cast-copy buffers do not have equal numel")
    with torch.no_grad():
        output.copy_(source.reshape(output.shape).to(output.dtype))


def final_norm_slice(torch: Any, values: tuple[Any, ...]) -> None:
    weight, residual, attention_output, mlp_output, rsqrt, output = values[:6]
    if not (residual.numel() == attention_output.numel() == mlp_output.numel()):
        raise RuntimeError("final norm-slice hidden inputs do not have equal numel")
    if residual.ndim != 3 or output.ndim != 3:
        raise RuntimeError("unexpected final norm-slice rank")
    with torch.no_grad():
        hidden = residual + attention_output.reshape(residual.shape).float() + mlp_output.reshape(residual.shape).float()
        normalized = weight.reshape(1, 1, -1) * hidden * rsqrt.reshape(hidden.shape[:-1] + (1,))
        output.copy_(normalized[:, -output.shape[1]:, :].to(output.dtype))


LIMIT = ["named singleton generated kernel only", "one frozen state", "repair-only implementation-relative evidence", "no correctness claim"]


if __name__ == "__main__":
    treatments = [
        {"kernel_family": "triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0", "repair": embedding_norm, "valid_status": "VALID_ORIGINAL_CANDIDATE_EMBEDDING_NORM_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_poi_fused__to_copy_add_arange_bitwise_and_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_5", "repair": causal_mask, "valid_status": "VALID_ORIGINAL_CANDIDATE_CAUSAL_MASK_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_poi_fused__safe_softmax_9", "repair": zero_safe_softmax_buffer, "valid_status": "VALID_ORIGINAL_CANDIDATE_ZERO_SOFTMAX_BUFFER_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_per_fused__unsafe_view_add_mean_pow_rsqrt_16", "repair": final_rsqrt, "valid_status": "VALID_ORIGINAL_CANDIDATE_FINAL_RSQRT_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_poi_fused__to_copy_t_17", "repair": cast_copy, "valid_status": "VALID_ORIGINAL_CANDIDATE_LM_HEAD_WEIGHT_CAST_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_poi_fused__to_copy__unsafe_view_add_mul_slice_18", "repair": final_norm_slice, "valid_status": "VALID_ORIGINAL_CANDIDATE_FINAL_NORM_SLICE_REPAIR", "claim_limits": LIMIT},
        {"kernel_family": "triton_poi_fused__to_copy__unsafe_view_19", "repair": cast_copy, "valid_status": "VALID_ORIGINAL_CANDIDATE_LOGITS_CAST_REPAIR", "claim_limits": LIMIT},
    ]
    run_campaign(treatments, {
        "description": __doc__,
        "schema_version": "forkcert.qwen3-candidate-singleton-kernel-campaign.v0.1",
    })
