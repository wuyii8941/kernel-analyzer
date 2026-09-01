"""FP32 reductions that differ only in their addition order.

The functions in this module are intentionally simple.  They are used by the
same-dtype reduction experiment, where both implementations receive FP32
operands and return FP32 results.  The only permitted difference is the order
in which reduction terms are added.
"""

from __future__ import annotations

import torch


def sequential_fp32_sum(values: torch.Tensor) -> torch.Tensor:
    """Reduce the leading dimension from left to right in FP32."""

    if values.dtype != torch.float32 or values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("sequential reduction requires a nonempty FP32 tensor")
    result = values[0].clone()
    for index in range(1, values.shape[0]):
        result = result + values[index]
    return result


def balanced_fp32_sum(values: torch.Tensor) -> torch.Tensor:
    """Reduce the leading dimension with a deterministic balanced tree."""

    if values.dtype != torch.float32 or values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("balanced reduction requires a nonempty FP32 tensor")
    level = [values[index].clone() for index in range(values.shape[0])]
    while len(level) > 1:
        next_level: list[torch.Tensor] = []
        for index in range(0, len(level) - 1, 2):
            next_level.append(level[index] + level[index + 1])
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def permuted_sequential_fp32_sum(
    values: torch.Tensor, permutation: torch.Tensor
) -> torch.Tensor:
    """Apply one frozen legal permutation, then use the sequential reducer."""

    if permutation.dtype != torch.int64 or permutation.ndim != 1:
        raise ValueError("permutation must be a one-dimensional int64 tensor")
    if permutation.numel() != values.shape[0]:
        raise ValueError("permutation length differs from reduction length")
    expected = torch.arange(values.shape[0], dtype=torch.int64, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("reduction order is not a permutation")
    return sequential_fp32_sum(values.index_select(0, permutation.to(values.device)))


class _OrderedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        reduction: str,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("the same-dtype experiment requires FP32 hidden and weight")
        if hidden.shape[-1] != weight.numel():
            raise ValueError("RMSNorm weight does not match the hidden dimension")
        if reduction not in {"sequential", "balanced", "permuted_sequential"}:
            raise ValueError(f"unknown reduction order: {reduction}")
        variance = hidden.square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(variance + float(epsilon))
        normalized = hidden * inverse_rms
        ctx.reduction = reduction
        ctx.save_for_backward(normalized, inverse_rms, weight, permutation)
        return normalized * weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        normalized, inverse_rms, weight, permutation = ctx.saved_tensors
        weighted = grad_output * weight
        projection = (weighted * normalized).mean(dim=-1, keepdim=True)
        grad_hidden = inverse_rms * (weighted - normalized * projection)
        contributions = (grad_output * normalized).reshape(-1, weight.numel())
        if ctx.reduction == "sequential":
            grad_weight = sequential_fp32_sum(contributions)
        elif ctx.reduction == "balanced":
            grad_weight = balanced_fp32_sum(contributions)
        else:
            grad_weight = permuted_sequential_fp32_sum(contributions, permutation)
        return grad_hidden, grad_weight, None, None, None


def ordered_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
    reduction: str,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm with an explicitly selected FP32 weight-gradient sum order."""

    if permutation is None:
        permutation = torch.empty(0, dtype=torch.int64, device=hidden.device)
    return _OrderedRMSNorm.apply(hidden, weight, epsilon, reduction, permutation)
