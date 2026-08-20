"""Exact source decomposition for a low-precision tensor endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class RoundingRepair:
    """One source-aligned low-precision materialization counterfactual.

    ``delivered`` always has the original low-precision ABI.  ``base`` is the
    quantized FP32 reference before an optional, measured kernel residual is
    restored.  Keeping those two tensors separate is important: replacing an
    MM by ``fp32_mm(...).to(bfloat16)`` repairs kernel arithmetic, but it does
    *not* repair deterministic output rounding.
    """

    delivered: torch.Tensor
    base: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    upper_probability: torch.Tensor


def decompose_low_precision_output(
    actual_low: torch.Tensor, high_reference: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Split same-operands error into local-kernel and output-rounding terms.

    ``high_reference`` must be the mathematical operation evaluated with the
    same operands promoted to FP32. The returned FP32 tensors satisfy
    ``kernel + output_rounding == total`` coordinate by coordinate.
    """
    if actual_low.shape != high_reference.shape:
        raise ValueError("actual and reference shapes differ")
    if not actual_low.is_floating_point() or not high_reference.is_floating_point():
        raise TypeError("precision decomposition requires floating tensors")
    if high_reference.dtype != torch.float32:
        raise TypeError("high reference must be FP32")
    rounded = high_reference.to(actual_low.dtype).float()
    actual = actual_low.detach().float()
    high = high_reference.detach().float()
    kernel = actual - rounded
    output_rounding = rounded - high
    total = actual - high
    return {
        "kernel": kernel,
        "output_rounding": output_rounding,
        "total": total,
    }


def low_precision_neighbors(
    high_reference: torch.Tensor, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return adjacent representable values bracketing an FP32 tensor.

    The probability is for the upper neighbor and makes the quantizer
    unbiased coordinate-wise in exact arithmetic.  Exact representable values
    have identical neighbors and probability zero.
    """

    if high_reference.dtype != torch.float32:
        raise TypeError("high reference must be FP32")
    if dtype not in {torch.bfloat16, torch.float16}:
        raise TypeError("source-aligned rounding repair requires BF16 or FP16 ABI")
    if not bool(torch.isfinite(high_reference).all()):
        raise ValueError("rounding repair requires finite FP32 values")

    nearest = high_reference.to(dtype)
    nearest_f32 = nearest.float()
    neg_inf = torch.full((), -float("inf"), dtype=dtype, device=nearest.device)
    pos_inf = torch.full((), float("inf"), dtype=dtype, device=nearest.device)
    previous = torch.nextafter(nearest, neg_inf)
    following = torch.nextafter(nearest, pos_inf)
    lower = torch.where(nearest_f32 > high_reference, previous, nearest)
    upper = torch.where(nearest_f32 < high_reference, following, nearest)
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("FP32 reference lies outside finite low-precision support")

    lower_f32 = lower.float()
    upper_f32 = upper.float()
    width = upper_f32 - lower_f32
    exact = width == 0
    probability = torch.where(
        exact,
        torch.zeros_like(high_reference),
        (high_reference - lower_f32) / width,
    ).clamp_(0.0, 1.0)
    if not bool((lower_f32 <= high_reference).all()):
        raise RuntimeError("lower rounding neighbor does not bracket reference")
    if not bool((high_reference <= upper_f32).all()):
        raise RuntimeError("upper rounding neighbor does not bracket reference")
    return lower, upper, probability


def stochastic_round_to_low_precision(
    high_reference: torch.Tensor,
    dtype: torch.dtype,
    *,
    generator: torch.Generator,
    stratum_index: int | None = None,
    stratum_count: int | None = None,
) -> RoundingRepair:
    """Stochastically materialize FP32 values through the declared ABI.

    This removes the conditional mean of deterministic nearest rounding while
    retaining the original dtype and support.  It is a debiasing intervention,
    not an exact FP32 shadow: a single low-precision tensor cannot represent an
    arbitrary FP32 result.
    """

    lower, upper, probability = low_precision_neighbors(high_reference, dtype)
    draw = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    if stratum_index is not None or stratum_count is not None:
        if stratum_index is None or stratum_count is None:
            raise ValueError("rounding stratum index and count must be supplied together")
        if stratum_count < 1 or not 0 <= stratum_index < stratum_count:
            raise ValueError("invalid rounding stratum")
        draw = (draw + float(stratum_index)) / float(stratum_count)
    delivered = torch.where(draw < probability, upper, lower)
    return RoundingRepair(
        delivered=delivered,
        base=delivered,
        lower=lower,
        upper=upper,
        upper_probability=probability,
    )


def source_aligned_mm_output(
    actual_low: torch.Tensor,
    high_reference: torch.Tensor,
    mode: str,
    *,
    generator: torch.Generator | None = None,
    rounding_stratum_index: int | None = None,
    rounding_stratum_count: int | None = None,
) -> RoundingRepair:
    """Construct one MM source intervention without confusing its sources.

    Modes:

    ``SHAM``
        Reconstruct the observed tensor from the exact decomposition.
    ``KERNEL_ONLY``
        Remove local kernel arithmetic while retaining deterministic output
        rounding (the historical FP32-accumulate/BF16-cast intervention).
    ``ROUNDING_ONLY``
        Replace nearest rounding by unbiased stochastic rounding while
        retaining the measured local-kernel residual where representable.
    ``JOINT``
        Remove local-kernel arithmetic and replace deterministic rounding by
        unbiased stochastic rounding.
    """

    terms = decompose_low_precision_output(actual_low, high_reference)
    rounded = high_reference.to(actual_low.dtype)
    if mode == "SHAM":
        delivered = (rounded.float() + terms["kernel"]).to(actual_low.dtype)
        base = rounded
    elif mode == "KERNEL_ONLY":
        delivered = rounded
        base = rounded
    elif mode in {"ROUNDING_ONLY", "JOINT"}:
        if generator is None:
            raise ValueError(f"{mode} requires an explicit generator")
        stochastic = stochastic_round_to_low_precision(
            high_reference, actual_low.dtype, generator=generator,
            stratum_index=rounding_stratum_index,
            stratum_count=rounding_stratum_count,
        )
        base = stochastic.delivered
        delivered = base if mode == "JOINT" else (
            base.float() + terms["kernel"]
        ).to(actual_low.dtype)
        return RoundingRepair(
            delivered=delivered,
            base=base,
            lower=stochastic.lower,
            upper=stochastic.upper,
            upper_probability=stochastic.upper_probability,
        )
    else:
        raise ValueError(f"unknown MM source intervention: {mode}")
    return RoundingRepair(
        delivered=delivered,
        base=base,
        lower=rounded,
        upper=rounded,
        upper_probability=torch.zeros_like(high_reference),
    )
