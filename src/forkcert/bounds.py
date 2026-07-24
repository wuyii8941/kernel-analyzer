from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

UNIT_ROUNDOFF = {
    "bf16": 2.0**-8,
    "bfloat16": 2.0**-8,
    "fp16": 2.0**-11,
    "float16": 2.0**-11,
    "fp32": 2.0**-24,
    "float32": 2.0**-24,
}


@dataclass
class ErrorSource:
    name: str
    mechanism: str
    dtype: str
    reduction_length: int = 1
    sum_abs: float = 0.0
    reduction: str = "tree"
    materialization_count_delta: int = 0
    local_scale: float = 0.0
    propagation: float = 1.0
    logprob_lipschitz: float = 2.0
    reduction_path_count: int = 2
    assumptions_verified: bool = False
    algorithm_order_known: bool = False
    input_norm_measured: bool = False
    propagation_certified: bool = False
    difference_injection: bool = False
    shared_rounding_cancelled: bool = False
    local_error_independent: bool = False
    propagation_empirically_calibrated: bool = False
    notes: str | None = None


@dataclass
class BoundResult:
    source_count: int
    activation_bound_worst: float
    activation_bound_prob: float
    logprob_bound_worst: float
    logprob_bound_prob: float
    delta: float
    per_source: list[dict]

    def to_json_dict(self) -> dict:
        return asdict(self)


def unit_roundoff(dtype: str) -> float:
    key = dtype.lower()
    if key not in UNIT_ROUNDOFF:
        raise ValueError(f"unsupported dtype for bound calculation: {dtype}")
    return UNIT_ROUNDOFF[key]


def reduction_error_worst(n: int, u: float, sum_abs: float, reduction: str) -> float:
    if n <= 1 or sum_abs == 0:
        return 0.0
    if reduction == "sequential":
        operations = n - 1
    elif reduction == "tree":
        operations = math.ceil(math.log2(n))
    else:
        raise ValueError(f"unsupported reduction type: {reduction}")
    ku = operations * u
    if ku >= 1.0:
        return math.inf
    gamma = ku / (1.0 - ku)
    return gamma * abs(sum_abs)


def reduction_error_prob(n: int, u: float, sum_abs: float, delta: float = 1e-6) -> float:
    if n <= 1 or sum_abs == 0:
        return 0.0
    lam = math.sqrt(2.0 * math.log(2.0 / delta))
    return lam * math.sqrt(n) * u * abs(sum_abs)


def materialization_error(count_delta: int, u: float, local_scale: float) -> float:
    count = abs(count_delta)
    if count == 0:
        return 0.0
    ku = count * u
    if ku >= 1.0:
        return math.inf
    return (ku / (1.0 - ku)) * abs(local_scale)


def source_bounds(source: ErrorSource, delta: float = 1e-6) -> tuple[float, float]:
    u = unit_roundoff(source.dtype)
    red_worst = source.reduction_path_count * reduction_error_worst(
        source.reduction_length, u, source.sum_abs, source.reduction
    )
    red_prob = source.reduction_path_count * reduction_error_prob(
        source.reduction_length, u, source.sum_abs, delta=delta
    )
    mat = materialization_error(source.materialization_count_delta, u, source.local_scale)
    worst = (red_worst + mat) * source.propagation
    probabilistic = (red_prob + mat) * source.propagation
    return worst, min(worst, probabilistic)


def local_source_bounds(source: ErrorSource, delta: float = 1e-6) -> tuple[float, float]:
    """Return the local injection bound before deterministic propagation."""
    u = unit_roundoff(source.dtype)
    red_worst = source.reduction_path_count * reduction_error_worst(
        source.reduction_length, u, source.sum_abs, source.reduction
    )
    red_prob = source.reduction_path_count * reduction_error_prob(
        source.reduction_length, u, source.sum_abs, delta=delta
    )
    mat = materialization_error(source.materialization_count_delta, u, source.local_scale)
    worst = red_worst + mat
    return worst, min(worst, red_prob + mat)


def assemble_semi_certified_probability_bound(
    sources: Iterable[ErrorSource], delta: float = 1e-6
) -> dict:
    """P4 assembly: RSS across independent local injections, never across propagation gains."""
    source_list = list(sources)
    failures = []
    for source in source_list:
        if not source.difference_injection:
            failures.append(f"{source.name}: not marked as a path-difference injection")
        if not source.shared_rounding_cancelled:
            failures.append(f"{source.name}: shared-rounding cancellation not established")
        if not source.local_error_independent:
            failures.append(f"{source.name}: cross-source local independence not established")
        if not source.propagation_empirically_calibrated:
            failures.append(f"{source.name}: propagation gain is not empirically calibrated")
        if source.propagation < 0 or not math.isfinite(source.propagation):
            failures.append(f"{source.name}: propagation must be finite and non-negative")
    per_source_delta = delta / len(source_list) if source_list else delta
    rows = []
    activation_worst_terms = []
    activation_prob_terms = []
    logprob_worst_terms = []
    logprob_prob_terms = []
    for source in source_list:
        local_worst, local_prob = local_source_bounds(source, delta=per_source_delta)
        propagated_worst = local_worst * source.propagation
        propagated_prob = local_prob * source.propagation
        logprob_worst = source.logprob_lipschitz * propagated_worst
        logprob_prob = source.logprob_lipschitz * propagated_prob
        activation_worst_terms.append(propagated_worst)
        activation_prob_terms.append(propagated_prob)
        logprob_worst_terms.append(logprob_worst)
        logprob_prob_terms.append(logprob_prob)
        row = asdict(source)
        row.update(
            {
                "local_bound_worst": local_worst,
                "local_bound_prob": local_prob,
                "probability_failure_budget": per_source_delta,
                "propagated_bound_worst": propagated_worst,
                "propagated_bound_prob": propagated_prob,
                "logprob_bound_worst_term": logprob_worst,
                "logprob_bound_prob_term": logprob_prob,
            }
        )
        rows.append(row)
    valid = not failures
    return {
        "certificate_kind": "semi_certified" if valid else "unverified_diagnostic",
        "source_count": len(source_list),
        "delta": delta,
        "assembly": "sqrt(sum((local_probability_bound * full_empirical_propagation_gain)^2))",
        "propagation_rss_forbidden": True,
        "validation": valid,
        "validation_failures": failures,
        "activation_bound_worst": sum(activation_worst_terms),
        "activation_bound_prob": math.sqrt(sum(value * value for value in activation_prob_terms)),
        "logprob_bound_worst": sum(logprob_worst_terms),
        "logprob_bound_prob": math.sqrt(sum(value * value for value in logprob_prob_terms)),
        "per_source": rows,
    }


def legal_sources_valid(sources: Iterable[ErrorSource]) -> tuple[bool, list[str]]:
    failures = []
    for source in sources:
        missing = [
            field
            for field in [
                "assumptions_verified",
                "algorithm_order_known",
                "input_norm_measured",
                "propagation_certified",
            ]
            if not getattr(source, field)
        ]
        if missing:
            failures.append(f"{source.name}: missing {','.join(missing)}")
        if source.propagation < 0 or not math.isfinite(source.propagation):
            failures.append(f"{source.name}: propagation must be finite and non-negative")
        if source.logprob_lipschitz < 0 or not math.isfinite(source.logprob_lipschitz):
            failures.append(f"{source.name}: logprob_lipschitz must be finite and non-negative")
    return not failures, failures


def assemble_logprob_bound(sources: Iterable[ErrorSource], delta: float = 1e-6) -> BoundResult:
    source_list = list(sources)
    per_source = []
    total_worst = 0.0
    total_prob = 0.0
    total_logprob_worst = 0.0
    total_logprob_prob = 0.0
    per_source_delta = delta / len(source_list) if source_list else delta
    for source in source_list:
        worst, prob = source_bounds(source, delta=per_source_delta)
        total_worst += worst
        total_prob += prob
        logprob_worst = source.logprob_lipschitz * worst
        logprob_prob = source.logprob_lipschitz * prob
        total_logprob_worst += logprob_worst
        total_logprob_prob += logprob_prob
        row = asdict(source)
        row.update(
            {
                "activation_bound_worst": worst,
                "activation_bound_prob": prob,
                "probability_failure_budget": per_source_delta,
                "logprob_bound_worst": logprob_worst,
                "logprob_bound_prob": logprob_prob,
            }
        )
        per_source.append(row)
    return BoundResult(
        source_count=len(per_source),
        activation_bound_worst=total_worst,
        activation_bound_prob=total_prob,
        logprob_bound_worst=total_logprob_worst,
        logprob_bound_prob=total_logprob_prob,
        delta=delta,
        per_source=per_source,
    )


def phase2_decision(logprob_bound_prob: float | None, empirical_delta_p99: float | None) -> str:
    if logprob_bound_prob is None or empirical_delta_p99 is None or empirical_delta_p99 <= 0:
        return "UNKNOWN: no empirical Phase 1 delta distribution supplied."
    ratio = logprob_bound_prob / empirical_delta_p99
    if logprob_bound_prob < empirical_delta_p99:
        return "VIOLATION: empirical p99(delta) exceeds probability bound; refine B, check confounds, or inspect bug candidates."
    if ratio <= 100:
        return "GO: probability bound is tight enough for three-way classification."
    if ratio > 1000:
        return "DOWNGRADE: bound is too loose as a classifier; use stable/unknown or empirical baseline."
    return "REVIEW: bound is loose but may still be useful after source refinement."
