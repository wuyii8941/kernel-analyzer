"""Exact two-factor decomposition for numerical source interventions.

The four outputs must be measured on identical operands and training state:
candidate, repair factor A only, repair factor B only, and repair both factors.
The function reports a symmetric decomposition and never interprets a large
component as a population bias or a training consequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and nonempty")
    return result.reshape(-1)


def _summary(component: np.ndarray, total: np.ndarray) -> dict[str, float]:
    total_energy = float(total @ total)
    energy = float(component @ component)
    return {
        "energy": energy,
        "rms": float(np.sqrt(energy / component.size)),
        "projection_on_total": (
            float(component @ total / total_energy) if total_energy else 0.0
        ),
    }


def two_factor_source_decomposition(
    candidate: Any,
    repair_a: Any,
    repair_b: Any,
    repair_both: Any,
    *,
    factor_a: str,
    factor_b: str,
) -> dict[str, Any]:
    """Decompose candidate-minus-joint-repair into A and B contributions.

    This is the two-player Shapley decomposition of the measured four-corner
    intervention. It is an algebraic attribution for this fixed comparison,
    not an assumption that the factors are independent.
    """
    values = {
        "candidate": _vector(candidate, "candidate"),
        "repair_a": _vector(repair_a, "repair_a"),
        "repair_b": _vector(repair_b, "repair_b"),
        "repair_both": _vector(repair_both, "repair_both"),
    }
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1:
        raise ValueError("all four outputs must have identical coordinates")
    c, a, b, ab = (values[name] for name in (
        "candidate", "repair_a", "repair_b", "repair_both"
    ))
    total = c - ab
    contribution_a = 0.5 * ((c - a) + (b - ab))
    contribution_b = 0.5 * ((c - b) + (a - ab))
    interaction = c - a - b + ab
    closure = total - contribution_a - contribution_b
    return {
        "schema": "two-factor-numerical-source-decomposition-v1",
        "factor_a": factor_a,
        "factor_b": factor_b,
        "coordinate_count": int(total.size),
        "total": _summary(total, total),
        "factor_a_contribution": _summary(contribution_a, total),
        "factor_b_contribution": _summary(contribution_b, total),
        "factorial_interaction": _summary(interaction, total),
        "closure_max_abs": float(np.max(np.abs(closure))),
        "closure_l2": float(np.linalg.norm(closure)),
        "scope": "FIXED_OPERANDS_FOUR_CORNER_ATTRIBUTION",
        "does_not_establish": [
            "population mean bias",
            "unique physical cause outside the declared factors",
            "training loss consequence",
        ],
    }
