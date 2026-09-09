"""Value-independent fixed-coordinate summaries for signed tensor differences.

This is the maintained copy used by current analysis code.  The schema name is
kept stable so historical records remain readable after the old ``forkcert``
source tree was archived.
"""

from __future__ import annotations

from typing import Any

import torch


SCHEMA_VERSION = "forkcert.directional-error-sketch.v1"


def fixed_flat_coordinate_indices(count: int, *, sample_size: int = 64) -> torch.Tensor:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if count < 1:
        raise ValueError("cannot sketch an empty tensor")
    size = min(sample_size, count)
    if size == 1:
        return torch.zeros(1, dtype=torch.int64)
    return torch.arange(size, dtype=torch.int64) * (count - 1) // (size - 1)


def fixed_coordinate_tensor_sketch(value: torch.Tensor, *, sample_size: int = 64) -> dict[str, Any]:
    positions = fixed_flat_coordinate_indices(int(value.numel()), sample_size=sample_size)
    sampled = value.detach().reshape(-1)[positions.to(value.device)].double().cpu()
    value_double = value.detach().double()
    return {
        "selection_rule": "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES",
        "sample_size": int(positions.numel()),
        "tensor_numel": int(value.numel()),
        "flat_coordinate_indices": [int(item) for item in positions.tolist()],
        "values": [float(item) for item in sampled.tolist()],
        "all_values_finite": bool(torch.isfinite(value.detach()).all().cpu()),
        "rms": float(torch.sqrt(torch.mean(value_double.square())).cpu()),
        "max_abs": float(value_double.abs().max().cpu()),
        "values_used_to_select_coordinates": False,
    }


def fixed_coordinate_error_sketch(
    candidate: torch.Tensor, reference: torch.Tensor, *, sample_size: int = 64,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    positions = fixed_flat_coordinate_indices(int(candidate.numel()), sample_size=sample_size)
    left = candidate.detach().reshape(-1)[positions.to(candidate.device)].double().cpu()
    right = reference.detach().reshape(-1)[positions.to(reference.device)].double().cpu()
    delta = left - right
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES",
        "sample_size": int(positions.numel()),
        "tensor_numel": int(candidate.numel()),
        "flat_coordinate_indices": [int(item) for item in positions.tolist()],
        "candidate_values": [float(item) for item in left.tolist()],
        "reference_values": [float(item) for item in right.tolist()],
        "signed_delta_values": [float(item) for item in delta.tolist()],
        "all_sampled_values_finite": bool(
            torch.isfinite(left).all() and torch.isfinite(right).all() and torch.isfinite(delta).all()
        ),
        "candidate_values_used_to_select_coordinates": False,
    }
