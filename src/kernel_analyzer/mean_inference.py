"""Shared numerical routines for conditional Student mean inference.

Accurate quantiles do not remove normality or asymptotic assumptions. A
constant observed sample does not establish a degenerate population.
"""
import math


def student_quantile(df: int, probability: float) -> float:
    if not math.isfinite(df) or df < 1:
        raise ValueError("degrees of freedom must be finite and positive")
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    from scipy.stats import t

    return float(t.ppf(probability, df))


def mean_test_p(estimate: float, standard_error: float, df: int) -> float:
    if not math.isfinite(estimate) or not math.isfinite(standard_error) or standard_error < 0:
        raise ValueError("estimate and standard error must be finite; error must be nonnegative")
    if not math.isfinite(df) or df < 1:
        raise ValueError("degrees of freedom must be positive")
    if standard_error == 0:
        return 1.0
    from scipy.stats import t

    return float(2 * t.sf(abs(estimate) / standard_error, df))


def mean_inference_p(branch: dict) -> float:
    """Read only a mean-test endpoint, never substitute a symmetry diagnostic."""
    if branch.get("status", "").startswith("NOT_IDENTIFIABLE"):
        return 1.0
    if "raw_studentized_mean_p" not in branch:
        raise ValueError("Legacy profile lacks the current mean endpoint; recompute before aggregation")
    value = float(branch["raw_studentized_mean_p"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid mean-test probability")
    return value
