"""Protocol-relative fixed-suite update-equivalence decisions.

The v2 decision combines three directional summaries with a mandatory energy
summary. Its interpretation depends on measurement provenance: an exact Gram
represents the retained coordinates, while a CountSketch Gram represents only
the sketch. A maximum across sketches is not a certified original-space norm
bound. Absence of statistical significance is never treated as equivalence.

The v1 classifier is retained for historical compatibility and internal use;
its three projections alone do not establish full-vector equivalence. Neither
version proves a bias mechanism or training-loss equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import NormalDist

import numpy as np


BRANCHES = ("additive", "repair_aligned", "residual_direction")


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return P[Binomial(trials, probability) <= successes].

    This small implementation keeps the exact prevalence endpoint independent
    of an optional scipy installation.  Log probabilities avoid overflow for
    the sample sizes used by the protocol.
    """

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials] and trials must be positive")
    if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
        raise ValueError("probability must lie in [0, 1]")
    if successes == trials or probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * math.log(probability)
        + (trials - index) * math.log1p(-probability)
        for index in range(successes + 1)
    ]
    largest = max(logs)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(value - largest) for value in logs))


def exact_binomial_one_sided_bounds(
    violations: int,
    independent_units: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Clopper--Pearson lower and upper one-sided bounds.

    The two returned bounds each have one-sided error probability ``alpha``;
    they are not advertised as one simultaneous ``1-alpha`` interval.
    """

    if independent_units < 1 or not 0 <= violations <= independent_units:
        raise ValueError("violations must lie in [0, independent_units]")
    if not 0.0 < alpha < 0.5 or not math.isfinite(alpha):
        raise ValueError("alpha must lie in (0, 0.5)")

    def solve_cdf(count: int, target: float) -> float:
        lower, upper = 0.0, 1.0
        for _ in range(80):
            middle = (lower + upper) / 2.0
            # For count < trials the CDF decreases with probability.
            if _binomial_cdf(count, independent_units, middle) > target:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2.0

    lower_bound = (
        0.0
        if violations == 0
        else solve_cdf(violations - 1, 1.0 - alpha)
    )
    upper_bound = (
        1.0
        if violations == independent_units
        else solve_cdf(violations, alpha)
    )
    return lower_bound, upper_bound


def population_statewise_rms_exceedance_equivalence(
    effect_energy_by_unit: Sequence[float],
    repair_energy_by_unit: Sequence[float],
    *,
    statewise_rms_margin: float,
    maximum_exceedance_probability: float,
    repair_energy_floor: float = 0.0,
    alpha: float = 0.05,
) -> dict:
    """Exact finite-sample inference for statewise tolerance exceedance.

    The population target is

    ``P(sqrt(X/B) >= statewise_rms_margin)``.

    It is deliberately distinct from the mean-energy target
    ``E[X] / E[B]``.  Under independent Bernoulli sampling of declared units,
    no upper bound on the magnitude of ``X`` is needed.  Every repair energy
    must exceed the floor fixed by the sampling protocol.
    """

    x = np.asarray(effect_energy_by_unit, dtype=np.float64)
    b = np.asarray(repair_energy_by_unit, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or x.shape != b.shape:
        raise ValueError("effect and repair energies must be matched nonempty vectors")
    if not np.isfinite(x).all() or not np.isfinite(b).all() or np.any(x < 0.0):
        raise ValueError("effect and repair energies must be finite and effect energies nonnegative")
    if (
        not math.isfinite(repair_energy_floor)
        or repair_energy_floor < 0.0
        or np.any(b <= repair_energy_floor)
    ):
        raise ValueError("every repair energy must exceed the predeclared repair-energy floor")
    if not math.isfinite(statewise_rms_margin) or statewise_rms_margin <= 0.0:
        raise ValueError("statewise_rms_margin must be finite and positive")
    if (
        not math.isfinite(maximum_exceedance_probability)
        or not 0.0 < maximum_exceedance_probability < 1.0
    ):
        raise ValueError("maximum_exceedance_probability must lie in (0, 1)")

    squared_margin = float(statewise_rms_margin) ** 2
    if not math.isfinite(squared_margin) or squared_margin == 0.0:
        raise ValueError("margin arithmetic overflow or underflow")
    # Equality is an exceedance because the equivalence region is strict.
    indicators = x >= squared_margin * b
    violations = int(np.count_nonzero(indicators))
    lower, upper = exact_binomial_one_sided_bounds(violations, int(x.size), alpha=alpha)
    decision = (
        "EQUIVALENT"
        if upper < maximum_exceedance_probability
        else "NON_EQUIVALENT"
        if lower > maximum_exceedance_probability
        else "INCONCLUSIVE"
    )
    return {
        "decision": decision,
        "estimand": "PROBABILITY_STATEWISE_RMS_AT_OR_ABOVE_MARGIN",
        "observed_exceedance_count": violations,
        "observed_exceedance_fraction": violations / int(x.size),
        "statewise_rms_margin": float(statewise_rms_margin),
        "maximum_exceedance_probability": float(maximum_exceedance_probability),
        "one_sided_probability_bounds": [lower, upper],
        "bounds_are_simultaneous": False,
        "alpha_per_decision_direction": float(alpha),
        "independent_unit_count": int(x.size),
        "repair_energy_floor": float(repair_energy_floor),
        "assumption_scope": "FINITE_SAMPLE_IID_BERNOULLI_EXCEEDANCE_INDICATORS",
        "mean_energy_q_guarantee": False,
        "interpretation": (
            "Bounds the prevalence of states outside a statewise RMS tolerance; "
            "it does not bound mean energy or the size of an exceedance."
        ),
    }


def _one_sided_mean_bounds(values: Sequence[float], alpha: float) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2 or not np.isfinite(samples).all():
        raise ValueError("one-sided inference requires at least two finite independent units")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    center = float(samples.mean())
    half = _t_critical(samples.size - 1, 1.0 - alpha) * float(
        samples.std(ddof=1) / math.sqrt(samples.size)
    )
    return center - half, center + half


def population_total_energy_equivalence(
    effect_energy_by_unit: Sequence[float],
    repair_energy_by_unit: Sequence[float],
    *,
    rms_margin: float,
    alpha: float = 0.05,
) -> dict:
    """Test E[X] < margin^2 E[B] without freezing a sample denominator.

    This is an asymptotic studentized-mean procedure over declared independent
    units.  Callers remain responsible for its independence and moment
    conditions; consecutive steps from one trajectory are not separate units.
    """

    x = np.asarray(effect_energy_by_unit, dtype=np.float64)
    b = np.asarray(repair_energy_by_unit, dtype=np.float64)
    if x.shape != b.shape or np.any(x < 0.0) or np.any(b < 0.0):
        raise ValueError("effect and repair energies must be matched and nonnegative")
    if rms_margin <= 0.0 or not math.isfinite(rms_margin):
        raise ValueError("rms_margin must be finite and positive")
    difference = x - float(rms_margin) ** 2 * b
    lower, upper = _one_sided_mean_bounds(difference, alpha)
    estimate = math.sqrt(float(x.mean() / b.mean())) if b.mean() > 0.0 else math.inf
    decision = "EQUIVALENT" if upper < 0.0 else "NON_EQUIVALENT" if lower > 0.0 else "INCONCLUSIVE"
    return {
        "decision": decision,
        "rms_ratio_estimate": estimate,
        "margin": float(rms_margin),
        "mean_boundary_difference": float(difference.mean()),
        "one_sided_mean_bounds": [lower, upper],
        "alpha": float(alpha),
        "independent_unit_count": int(x.size),
        "assumption_scope": "ASYMPTOTIC_STUDENTIZED_INDEPENDENT_UNITS",
    }


def bounded_population_total_energy_equivalence(
    effect_energy_by_unit: Sequence[float],
    repair_energy_by_unit: Sequence[float],
    *,
    rms_margin: float,
    effect_energy_upper_bound: float,
    repair_energy_upper_bound: float,
    bound_provenance: str,
    alpha: float = 0.05,
) -> dict:
    """Finite-sample test of ``E[X] < margin**2 E[B]`` under declared bounds.

    The bounds must follow from the data-generating protocol or another
    argument fixed before the units are observed.  A sample maximum is not a
    valid value for either bound.  Hoeffding's inequality is applied to
    ``D = X - margin**2 B``, whose declared range is
    ``[-margin**2 * B_max, X_max]``.  No normal approximation is used.
    """

    x = np.asarray(effect_energy_by_unit, dtype=np.float64)
    b = np.asarray(repair_energy_by_unit, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or x.shape != b.shape:
        raise ValueError("effect and repair energies must be matched nonempty vectors")
    if not np.isfinite(x).all() or not np.isfinite(b).all() or np.any(x < 0.0) or np.any(b < 0.0):
        raise ValueError("effect and repair energies must be finite and nonnegative")
    if rms_margin <= 0.0 or not math.isfinite(rms_margin):
        raise ValueError("rms_margin must be finite and positive")
    x_max = float(effect_energy_upper_bound)
    b_max = float(repair_energy_upper_bound)
    if not math.isfinite(x_max) or x_max < 0.0:
        raise ValueError("effect_energy_upper_bound must be finite and nonnegative")
    if not math.isfinite(b_max) or b_max <= 0.0:
        raise ValueError("repair_energy_upper_bound must be finite and positive")
    if not isinstance(bound_provenance, str) or not bound_provenance.strip():
        raise ValueError("bound_provenance must identify a pre-observation bound argument")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(x > x_max * (1.0 + tolerance)) or np.any(b > b_max * (1.0 + tolerance)):
        raise ValueError("observed energy exceeds its declared protocol bound")

    margin_squared = float(rms_margin) ** 2
    difference = x - margin_squared * b
    lower_support = -margin_squared * b_max
    upper_support = x_max
    support_width = upper_support - lower_support
    radius = support_width * math.sqrt(math.log(1.0 / alpha) / (2.0 * x.size))
    center = float(difference.mean())
    lower = center - radius
    upper = center + radius
    repair_mean = float(b.mean())
    estimate = math.sqrt(float(x.mean()) / repair_mean) if repair_mean > 0.0 else math.inf
    decision = "EQUIVALENT" if upper < 0.0 else "NON_EQUIVALENT" if lower > 0.0 else "INCONCLUSIVE"
    return {
        "decision": decision,
        "rms_ratio_estimate": estimate,
        "margin": float(rms_margin),
        "mean_boundary_difference": center,
        "one_sided_mean_bounds": [lower, upper],
        "hoeffding_radius": radius,
        "declared_support": [lower_support, upper_support],
        "effect_energy_upper_bound": x_max,
        "repair_energy_upper_bound": b_max,
        "bound_provenance": bound_provenance,
        "alpha": float(alpha),
        "independent_unit_count": int(x.size),
        "assumption_scope": "FINITE_SAMPLE_INDEPENDENT_UNITS_WITH_PREDECLARED_ENERGY_BOUNDS",
        "sample_extrema_are_not_valid_bound_provenance": True,
    }


def population_aligned_equivalence(
    effect_repair_inner_by_unit: Sequence[float],
    repair_energy_by_unit: Sequence[float],
    *,
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """Test |E[A]/E[B]| < margin using two matched mean inequalities."""

    a = np.asarray(effect_repair_inner_by_unit, dtype=np.float64)
    b = np.asarray(repair_energy_by_unit, dtype=np.float64)
    if a.shape != b.shape or np.any(b < 0.0):
        raise ValueError("inner products and repair energies must be matched")
    if margin <= 0.0 or not math.isfinite(margin) or b.mean() <= 0.0:
        raise ValueError("margin and mean repair energy must be positive")
    positive = a - float(margin) * b
    negative = -a - float(margin) * b
    positive_bounds = _one_sided_mean_bounds(positive, alpha)
    negative_bounds = _one_sided_mean_bounds(negative, alpha)
    equivalent = positive_bounds[1] < 0.0 and negative_bounds[1] < 0.0
    outside = positive_bounds[0] > 0.0 or negative_bounds[0] > 0.0
    return {
        "decision": "EQUIVALENT" if equivalent else "NON_EQUIVALENT" if outside else "INCONCLUSIVE",
        "ratio_of_means_estimate": float(a.mean() / b.mean()),
        "margin": float(margin),
        "positive_boundary_bounds": list(positive_bounds),
        "negative_boundary_bounds": list(negative_bounds),
        "alpha": float(alpha),
        "independent_unit_count": int(a.size),
        "assumption_scope": "ASYMPTOTIC_STUDENTIZED_INDEPENDENT_UNITS",
    }


def bounded_population_aligned_equivalence(
    effect_repair_inner_by_unit: Sequence[float],
    repair_energy_by_unit: Sequence[float],
    *,
    margin: float,
    effect_energy_upper_bound: float,
    repair_energy_upper_bound: float,
    bound_provenance: str,
    alpha: float = 0.05,
) -> dict:
    """Finite-sample test of ``|E[A] / E[B]| < margin``.

    Given protocol bounds ``X <= X_max`` and ``B <= B_max``, Cauchy--Schwarz
    gives ``|A| <= sqrt(X_max * B_max)``.  Hoeffding bounds are applied to the
    two matched inequalities ``A-margin*B`` and ``-A-margin*B``.  Passing is an
    intersection decision, so each component can use the declared ``alpha``;
    this function does not use an observed maximum as a population bound.
    """

    a = np.asarray(effect_repair_inner_by_unit, dtype=np.float64)
    b = np.asarray(repair_energy_by_unit, dtype=np.float64)
    if a.ndim != 1 or a.size < 1 or a.shape != b.shape:
        raise ValueError("inner products and repair energies must be matched nonempty vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b < 0.0):
        raise ValueError("inner products must be finite and repair energies nonnegative")
    if margin <= 0.0 or not math.isfinite(margin):
        raise ValueError("margin must be finite and positive")
    x_max = float(effect_energy_upper_bound)
    b_max = float(repair_energy_upper_bound)
    if not math.isfinite(x_max) or x_max < 0.0:
        raise ValueError("effect_energy_upper_bound must be finite and nonnegative")
    if not math.isfinite(b_max) or b_max <= 0.0:
        raise ValueError("repair_energy_upper_bound must be finite and positive")
    if not isinstance(bound_provenance, str) or not bound_provenance.strip():
        raise ValueError("bound_provenance must identify a pre-observation bound argument")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    a_max = math.sqrt(x_max * b_max)
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(np.abs(a) > a_max * (1.0 + tolerance)) or np.any(b > b_max * (1.0 + tolerance)):
        raise ValueError("observed inner product or repair energy exceeds its declared bound")

    positive = a - float(margin) * b
    negative = -a - float(margin) * b
    support = [-a_max - float(margin) * b_max, a_max]
    radius = (support[1] - support[0]) * math.sqrt(math.log(1.0 / alpha) / (2.0 * a.size))
    positive_bounds = [float(positive.mean()) - radius, float(positive.mean()) + radius]
    negative_bounds = [float(negative.mean()) - radius, float(negative.mean()) + radius]
    equivalent = positive_bounds[1] < 0.0 and negative_bounds[1] < 0.0
    outside = positive_bounds[0] > 0.0 or negative_bounds[0] > 0.0
    repair_mean = float(b.mean())
    estimate = float(a.mean() / repair_mean) if repair_mean > 0.0 else math.nan
    return {
        "decision": "EQUIVALENT" if equivalent else "NON_EQUIVALENT" if outside else "INCONCLUSIVE",
        "ratio_of_means_estimate": estimate,
        "margin": float(margin),
        "positive_boundary_bounds": positive_bounds,
        "negative_boundary_bounds": negative_bounds,
        "hoeffding_radius": radius,
        "declared_boundary_support": support,
        "derived_inner_product_abs_bound": a_max,
        "effect_energy_upper_bound": x_max,
        "repair_energy_upper_bound": b_max,
        "bound_provenance": bound_provenance,
        "alpha": float(alpha),
        "independent_unit_count": int(a.size),
        "assumption_scope": "FINITE_SAMPLE_INDEPENDENT_UNITS_WITH_PREDECLARED_ENERGY_BOUNDS",
        "sample_extrema_are_not_valid_bound_provenance": True,
    }


def fixed_suite_total_rms_from_joint_gram(
    joint_gram: Mapping[str, Sequence[Sequence[float]]],
    *,
    calibration_count: int = 16,
) -> float:
    """Return the confirmation RMS ratio in the supplied Gram's vector space.

    Unlike the three directional summaries, this quantity sees every direction
    represented by the saved update vectors. If they are sketches, this is not
    an exact original-coordinate ratio or an upper bound on that ratio.
    """

    uu = np.asarray(joint_gram["effect_effect"], dtype=np.float64)
    rr = np.asarray(joint_gram["repair_repair"], dtype=np.float64)
    if uu.shape != rr.shape or uu.ndim != 2 or uu.shape[0] != uu.shape[1]:
        raise ValueError("effect and repair Gram matrices must share one square shape")
    count = uu.shape[0]
    if not 1 < calibration_count < count - 1:
        raise ValueError("calibration split leaves too few confirmation states")
    conf = np.arange(calibration_count, count)
    effect_energy = float(np.trace(uu[np.ix_(conf, conf)]))
    repair_energy = float(np.trace(rr[np.ix_(conf, conf)]))
    if repair_energy <= 0.0:
        raise ValueError("confirmation repair energy must be positive")
    if effect_energy < 0.0:
        raise ValueError("confirmation effect energy must be nonnegative")
    return math.sqrt(effect_energy / repair_energy)


def _t_critical(df: int, probability: float) -> float:
    z = NormalDist().inv_cdf(probability)
    f = float(df)
    return (
        z
        + (z**3 + z) / (4.0 * f)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * f**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * f**3)
    )


def simultaneous_intervals_from_joint_gram(
    joint_gram: Mapping[str, Sequence[Sequence[float]]],
    *,
    calibration_count: int = 16,
    family_alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Recover three simultaneous held-out intervals from saved inner products.

    The saved matrices are sufficient to reproduce the fixed-direction,
    repair-aligned, and scaling-removed confirmation values without retaining
    the original high-dimensional parameter vectors.
    """

    uu = np.asarray(joint_gram["effect_effect"], dtype=np.float64)
    rr = np.asarray(joint_gram["repair_repair"], dtype=np.float64)
    ur = np.asarray(joint_gram["effect_repair"], dtype=np.float64)
    if uu.shape != rr.shape or uu.shape != ur.shape or uu.ndim != 2 or uu.shape[0] != uu.shape[1]:
        raise ValueError("joint Gram matrices must be square and share one shape")
    count = uu.shape[0]
    if not 1 < calibration_count < count - 1:
        raise ValueError("calibration split leaves too few confirmation states")
    cal = np.arange(calibration_count)
    conf = np.arange(calibration_count, count)
    repair_energy = np.diag(rr)
    if np.any(repair_energy <= 0.0):
        raise ValueError("repair energy must be positive for all states")
    repair_scale = math.sqrt(float(repair_energy[conf].mean()))

    calibration_norm = math.sqrt(max(float(uu[np.ix_(cal, cal)].mean()), 0.0))
    if calibration_norm <= 0.0:
        # Exact identity needs no learned direction: every implementation
        # effect is zero.  Other degenerate cases remain fail-closed.
        if np.count_nonzero(uu) == 0 and np.count_nonzero(ur) == 0:
            return {name: [0.0, 0.0] for name in BRANCHES}
        raise ValueError("calibration mean direction is not identifiable")
    additive = uu[np.ix_(conf, cal)].mean(axis=1) / calibration_norm / repair_scale

    gain = np.diag(ur) / repair_energy
    # The declared aligned estimand is the energy-weighted ratio of sums.
    # Linearized samples retain that center while allowing the existing
    # studentized interval calculation to account for state-to-state variation.
    aligned_numerator = np.diag(ur)[conf]
    aligned_denominator = repair_energy[conf]
    aligned_center = float(aligned_numerator.sum() / aligned_denominator.sum())
    aligned = aligned_center + (
        aligned_numerator - aligned_center * aligned_denominator
    ) / float(aligned_denominator.mean())
    residual_gram = (
        uu
        - ur * gain[None, :]
        - ur.T * gain[:, None]
        + rr * gain[:, None] * gain[None, :]
    )
    residual_norm = math.sqrt(max(float(residual_gram[np.ix_(cal, cal)].mean()), 0.0))
    if residual_norm <= 0.0:
        residual = np.zeros(conf.size, dtype=np.float64)
    else:
        residual = residual_gram[np.ix_(conf, cal)].mean(axis=1) / residual_norm / repair_scale

    values = {"additive": additive, "repair_aligned": aligned, "residual_direction": residual}
    critical = _t_critical(conf.size - 1, 1.0 - family_alpha / (2.0 * len(BRANCHES)))
    intervals = {}
    for name, samples in values.items():
        center = float(samples.mean())
        half = critical * float(samples.std(ddof=1) / math.sqrt(samples.size))
        intervals[name] = [center - half, center + half]
    return intervals


def classify_fixed_suite_update_equivalence(
    intervals: Mapping[str, Sequence[float]] | None,
    margins: Mapping[str, float],
    *,
    total_rms: float,
    total_rms_margin: float,
    consequence_status: str = "NOT_DECLARED",
    exact_identity_verified: bool = False,
    valid_protocol: bool = True,
) -> dict:
    """Classify update equivalence on one declared finite input suite.

    The total-RMS envelope is mandatory and closes directions not represented
    by the three predeclared profile summaries.  This is not a claim about a
    random state population or downstream training quality.
    """

    if not valid_protocol:
        return {
            "decision": "ABSTAIN",
            "reason": "PROTOCOL_OR_REPAIR_INVALID",
            "claim_scope": "FIXED_SUITE_UPDATE",
            "material_consequence_status": consequence_status,
        }
    total_rms = float(total_rms)
    total_rms_margin = float(total_rms_margin)
    if not math.isfinite(total_rms) or total_rms < 0.0:
        raise ValueError("total_rms must be finite and nonnegative")
    if not math.isfinite(total_rms_margin) or total_rms_margin <= 0.0:
        raise ValueError("total_rms_margin must be finite and positive")
    if consequence_status not in {"NOT_DECLARED", "PASSED", "FAILED"}:
        raise ValueError("unknown material consequence status")

    profile = (
        classify_training_equivalence(intervals, margins)
        if intervals is not None else {
            "decision": "NOT_ASSESSED",
            "reason": "PROFILE_DIRECTION_NOT_IDENTIFIABLE",
            "branches": {},
        }
    )
    envelope_inside = total_rms < total_rms_margin
    envelope_beyond = total_rms > total_rms_margin
    exact_identity = total_rms == 0.0 and bool(exact_identity_verified)

    failure_reasons = []
    if envelope_beyond:
        failure_reasons.append("FULL_UPDATE_RMS_EXCEEDS_ITS_MARGIN")
    elif not envelope_inside:
        failure_reasons.append("FULL_UPDATE_RMS_IS_ON_ITS_MARGIN")
    if profile["decision"] == "MATERIAL_EFFECT":
        failure_reasons.append(profile["reason"])
    elif profile["decision"] in {"INCONCLUSIVE", "NOT_ASSESSED"} and not exact_identity:
        failure_reasons.append(profile["reason"])
    if consequence_status == "FAILED":
        failure_reasons.append("PREDECLARED_TRAINING_CONSEQUENCE_FAILED")

    if consequence_status == "FAILED":
        decision = "MATERIAL_CONSEQUENCE"
        reason = "PREDECLARED_TRAINING_CONSEQUENCE_FAILED"
    elif profile["decision"] == "MATERIAL_EFFECT":
        decision = "MATERIAL_EFFECT"
        reason = profile["reason"]
    elif envelope_beyond:
        decision = "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"
        reason = "FULL_UPDATE_RMS_EXCEEDS_ITS_MARGIN"
    elif not envelope_inside:
        decision = "INCONCLUSIVE"
        reason = "FULL_UPDATE_RMS_IS_ON_ITS_MARGIN"
    elif exact_identity:
        decision = "EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE"
        reason = "ALL_CONFIRMATION_UPDATE_DIFFERENCES_ARE_ZERO"
    elif profile["decision"] in {"INCONCLUSIVE", "NOT_ASSESSED"}:
        decision = "INCONCLUSIVE"
        reason = profile["reason"]
    else:
        decision = "FIXED_SUITE_UPDATE_EQUIVALENT"
        reason = "FULL_UPDATE_RMS_AND_ALL_PROFILE_INTERVALS_ARE_INSIDE_MARGINS"

    return {
        "decision": decision,
        "reason": reason,
        "failure_reasons": failure_reasons,
        "claim_scope": "FIXED_SUITE_UPDATE",
        "material_consequence_status": consequence_status,
        "full_update_rms": total_rms,
        "full_update_rms_margin": total_rms_margin,
        "full_update_rms_inside_margin": envelope_inside,
        "exact_identity_verified": exact_identity,
        "profile_decision": profile["decision"],
        "branches": profile["branches"],
    }


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} interval must have two bounds")
    lower, upper = float(value[0]), float(value[1])
    if lower > upper:
        raise ValueError(f"{name} interval bounds are reversed")
    return lower, upper


def classify_training_equivalence(
    intervals: Mapping[str, Sequence[float]],
    margins: Mapping[str, float],
    *,
    material_consequence_failed: bool = False,
    valid_protocol: bool = True,
) -> dict:
    """Classify one declared model/repair/optimizer/state protocol."""

    if not valid_protocol:
        return {"decision": "ABSTAIN", "reason": "PROTOCOL_OR_REPAIR_INVALID"}
    if set(intervals) != set(BRANCHES) or set(margins) != set(BRANCHES):
        raise ValueError("all and only the three frozen update branches are required")
    checked = {name: _interval(intervals[name], name) for name in BRANCHES}
    limits = {name: float(margins[name]) for name in BRANCHES}
    if any(value <= 0.0 for value in limits.values()):
        raise ValueError("equivalence margins must be positive")

    details = {}
    all_inside = True
    any_material = False
    any_small = False
    any_crosses_margin = False
    for name in BRANCHES:
        lower, upper = checked[name]
        margin = limits[name]
        inside = lower > -margin and upper < margin
        material = lower > margin or upper < -margin
        excludes_zero = lower > 0.0 or upper < 0.0
        crosses_margin = not inside and not material
        small = excludes_zero and inside
        details[name] = {
            "simultaneous_confidence_interval": [lower, upper],
            "equivalence_margin": margin,
            "inside_equivalence_region": inside,
            "entirely_beyond_materiality_margin": material,
            "detectable_inside_margin": small,
            "crosses_margin": crosses_margin,
        }
        all_inside &= inside
        any_material |= material
        any_small |= small
        any_crosses_margin |= crosses_margin

    if material_consequence_failed:
        decision = "MATERIAL_CONSEQUENCE"
        reason = "PREDECLARED_TRAINING_CONSEQUENCE_FAILED"
    elif any_material:
        decision = "MATERIAL_EFFECT"
        reason = "AT_LEAST_ONE_UPDATE_EFFECT_INTERVAL_IS_BEYOND_ITS_MARGIN"
    elif any_crosses_margin:
        decision = "INCONCLUSIVE"
        reason = "AT_LEAST_ONE_INTERVAL_CROSSES_AN_EQUIVALENCE_MARGIN"
    elif all_inside and any_small:
        decision = "DETECTABLE_BUT_SMALL"
        reason = "ALL_INTERVALS_ARE_INSIDE_MARGINS_AND_AT_LEAST_ONE_EXCLUDES_ZERO"
    elif all_inside:
        decision = "EQUIVALENT_UNDER_PROTOCOL"
        reason = "ALL_SIMULTANEOUS_INTERVALS_ARE_INSIDE_THE_FROZEN_EQUIVALENCE_REGION"
    else:
        decision = "INCONCLUSIVE"
        reason = "AVAILABLE_INTERVALS_DO_NOT_SUPPORT_EQUIVALENCE_OR_MATERIALITY"
    return {
        "decision": decision,
        "reason": reason,
        "material_consequence_failed": bool(material_consequence_failed),
        "branches": details,
    }
