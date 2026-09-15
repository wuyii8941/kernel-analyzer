"""Small, explicit summaries that keep bias separate from error energy."""
from __future__ import annotations

import math
from collections.abc import Sequence

from .mean_inference import mean_test_p, student_quantile
from .population_direction import population_positive_direction_prevalence
from .training_equivalence import exact_binomial_one_sided_bounds


def binary_prevalence_summary(
    successes: Sequence[bool], *, null_probability: float = .5, alpha: float = .05,
    estimand: str,
) -> dict:
    if not successes:
        raise ValueError("prevalence inference needs at least one unit")
    if not 0 < null_probability < 1 or not 0 < alpha < .5:
        raise ValueError("invalid probability or alpha")
    count = sum(bool(value) for value in successes)
    lower, upper = exact_binomial_one_sided_bounds(count, len(successes), alpha=alpha)
    return {
        "decision": (
            "PREVALENCE_ABOVE_NULL" if lower > null_probability
            else "PREVALENCE_BELOW_NULL" if upper < null_probability
            else "INCONCLUSIVE"
        ),
        "estimand": estimand,
        "success_count": count,
        "independent_unit_count": len(successes),
        "observed_fraction": count / len(successes),
        "one_sided_probability_bounds": [lower, upper],
        "null_probability": null_probability,
        "alpha_per_decision_direction": alpha,
        "bounds_are_simultaneous": False,
        "assumption_scope": "FINITE_SAMPLE_IID_BERNOULLI_INDICATORS",
    }


def statewise_aligned_summary(
    inner_products: Sequence[float], repair_energies: Sequence[float], *, alpha: float = .05
) -> dict:
    """Summarize per-unit aligned gain without calling it a vector-mean test."""
    if len(inner_products) != len(repair_energies) or len(inner_products) < 2:
        raise ValueError("aligned inference needs matching sequences with at least two units")
    values = []
    for inner, energy in zip(inner_products, repair_energies):
        if not math.isfinite(inner) or not math.isfinite(energy) or energy <= 0:
            raise ValueError("aligned sufficient statistics must be finite with positive repair energy")
        values.append(inner / energy)
    n = len(values)
    estimate = math.fsum(values) / n
    variance = math.fsum((value - estimate) ** 2 for value in values) / (n - 1)
    standard_error = math.sqrt(variance / n)
    critical = student_quantile(n - 1, 1 - alpha / 2)
    frequency = population_positive_direction_prevalence(
        inner_products, null_positive_probability=.5, alpha=alpha
    )
    return {
        "estimand": "MEAN_STATEWISE_REPAIR_ALIGNED_GAIN",
        "estimate": estimate,
        "conditional_student_interval": [
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ],
        "conditional_student_p": mean_test_p(estimate, standard_error, n - 1),
        "student_assumption": "independent units and a normal or adequate small-sample approximation",
        "positive_direction_frequency": frequency,
        "ratio_of_sums_descriptive": (
            math.fsum(inner_products) / math.fsum(repair_energies)
        ),
        "vector_mean_nonzero_established": False,
    }


def total_rms_from_gram(joint_gram: dict, indices: range) -> float:
    effects = joint_gram["effect_effect"]
    repairs = joint_gram["repair_repair"]
    numerator = math.fsum(effects[index][index] for index in indices)
    denominator = math.fsum(repairs[index][index] for index in indices)
    if numerator < 0 or denominator <= 0:
        raise ValueError("invalid Gram diagonal energy")
    return math.sqrt(numerator / denominator)
