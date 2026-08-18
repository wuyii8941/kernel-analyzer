"""Exact source decomposition for a low-precision tensor endpoint."""

from __future__ import annotations

from typing import Mapping

import torch


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
