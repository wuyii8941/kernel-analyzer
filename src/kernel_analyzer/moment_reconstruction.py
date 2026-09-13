"""Exact bookkeeping for how stored AdamW moment errors enter later steps."""
from __future__ import annotations

from typing import Any


def squared_norm(value) -> float:
    return float(value.detach().double().square().sum().item())


def reconstruction_terms(
    *, reference_previous, reference_current, candidate_previous_read,
    candidate_current_unstored, candidate_current_read, beta: float,
) -> dict[str, Any]:
    """Decompose a stored-state difference without assuming exact FP arithmetic.

    ``floating_remainder`` retains all evaluation differences between the
    candidate recurrence and ``reference_current + beta * previous_error``.
    ``storage_error`` is what is added by quantizing/reconstructing the newly
    computed candidate state.  Their sum is an algebraic reconstruction, not
    an independence or non-zero-mean assumption.
    """
    previous_error = candidate_previous_read - reference_previous
    transported_previous = previous_error * beta
    floating_remainder = (
        candidate_current_unstored - reference_current - transported_previous
    )
    storage_error = candidate_current_read - candidate_current_unstored
    actual = candidate_current_read - reference_current
    reconstructed = transported_previous + floating_remainder + storage_error
    residual = actual - reconstructed
    denominator = max(squared_norm(actual), 1e-300)
    return {
        "previous_error_energy": squared_norm(previous_error),
        "transported_previous_energy": squared_norm(transported_previous),
        "floating_remainder_energy": squared_norm(floating_remainder),
        "storage_error_energy": squared_norm(storage_error),
        "actual_error_energy": squared_norm(actual),
        "reconstructed_error_energy": squared_norm(reconstructed),
        "reconstruction_residual_energy": squared_norm(residual),
        "relative_reconstruction_residual": (squared_norm(residual) / denominator) ** 0.5,
    }


def update_component_terms(*, reference, candidate, first_only, second_only) -> dict:
    """Record first-moment, second-moment and nonlinear joint write effects."""
    total = candidate - reference
    first = first_only - reference
    second = second_only - reference
    interaction = total - first - second
    denominator = max(squared_norm(reference), 1e-300)
    return {
        "total_relative_rms": (squared_norm(total) / denominator) ** 0.5,
        "first_moment_only_relative_rms": (squared_norm(first) / denominator) ** 0.5,
        "second_moment_only_relative_rms": (squared_norm(second) / denominator) ** 0.5,
        "joint_nonlinear_remainder_relative_rms": (
            squared_norm(interaction) / denominator
        ) ** 0.5,
        "component_reconstruction_residual": squared_norm(
            total - first - second - interaction
        ),
    }
