"""Finite-sample population inference for a predeclared directional sign.

This endpoint asks how often a matched implementation effect has positive
inner product with a predeclared per-unit direction.  It does not estimate a
vector mean or the magnitude of the effect.  Exact binomial inference is valid
when the units are iid draws from the declared population.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from .training_equivalence import exact_binomial_one_sided_bounds


def population_positive_direction_prevalence(
    signed_inner_products: Sequence[float],
    *,
    null_positive_probability: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """Test whether positive directional effects occur more often than null.

    Zero is not counted as positive.  The direction and null probability must
    be fixed before observing the confirmation units.
    """

    values = np.asarray(signed_inner_products, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("signed inner products must be a nonempty finite vector")
    if (
        not math.isfinite(null_positive_probability)
        or not 0.0 < null_positive_probability < 1.0
    ):
        raise ValueError("null_positive_probability must lie in (0, 1)")
    positives = int(np.count_nonzero(values > 0.0))
    lower, upper = exact_binomial_one_sided_bounds(
        positives, int(values.size), alpha=alpha
    )
    decision = (
        "DIRECTION_PREVALENCE_CONFIRMED"
        if lower > null_positive_probability
        else "OPPOSITE_DIRECTION_PREVALENCE_CONFIRMED"
        if upper < null_positive_probability
        else "INCONCLUSIVE"
    )
    return {
        "decision": decision,
        "estimand": "PROBABILITY_OF_POSITIVE_PREDECLARED_DIRECTIONAL_INNER_PRODUCT",
        "positive_count": positives,
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "independent_unit_count": int(values.size),
        "observed_positive_fraction": positives / int(values.size),
        "one_sided_probability_bounds": [lower, upper],
        "null_positive_probability": float(null_positive_probability),
        "alpha_per_decision_direction": float(alpha),
        "bounds_are_simultaneous": False,
        "assumption_scope": "FINITE_SAMPLE_IID_BERNOULLI_DIRECTION_INDICATORS",
        "magnitude_or_mean_vector_guarantee": False,
    }
