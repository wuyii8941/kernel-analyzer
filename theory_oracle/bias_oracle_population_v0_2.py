"""Trajectory-aware scalar B/H/N/U core for the bias Oracle calibration stage."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "forkcert.bias-oracle-population.v0.2"


@dataclass(frozen=True)
class EffectRecord:
    trajectory_id: str
    phase: str
    state_id: str
    repeat_id: int
    effect: float


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty collection")
    return sum(items) / len(items)


def _t_critical_two_sided(df: int, alpha: float) -> float:
    if df < 1:
        return math.inf
    from scipy.stats import t

    return float(t.ppf(1.0 - alpha / 2.0, df))


def estimate_scalar_population(
    records: list[EffectRecord],
    *,
    required_phases: tuple[str, ...],
    min_confirmation_trajectories: int = 8,
    measurement_floor: float = 0.0,
    practical_tolerance: float | None = None,
    tail_prevalence: float | None = None,
    tail_alpha: float = 0.05,
    interval_alpha: float = 0.05,
) -> dict[str, Any]:
    if not records:
        raise ValueError("no effect records")
    if min_confirmation_trajectories < 2:
        raise ValueError("min_confirmation_trajectories must be at least two")
    if measurement_floor < 0:
        raise ValueError("measurement_floor must be non-negative")
    if practical_tolerance is not None and practical_tolerance < measurement_floor:
        raise ValueError("practical tolerance cannot be below the measurement floor")
    if tail_prevalence is not None and not 0.0 < tail_prevalence < 1.0:
        raise ValueError("tail_prevalence must lie strictly between zero and one")
    if not 0.0 < tail_alpha < 1.0:
        raise ValueError("tail_alpha must lie strictly between zero and one")
    if not 0.0 < interval_alpha < 1.0:
        raise ValueError("interval_alpha must lie strictly between zero and one")
    if not required_phases or len(required_phases) != len(set(required_phases)):
        raise ValueError("required phases must be unique and non-empty")

    keys = [(r.trajectory_id, r.phase, r.state_id, r.repeat_id) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trajectory/phase/state/repeat record")
    if any(not math.isfinite(r.effect) for r in records):
        raise ValueError("all effects must be finite")

    by_state: dict[tuple[str, str, str], list[EffectRecord]] = defaultdict(list)
    for record in records:
        by_state[(record.trajectory_id, record.phase, record.state_id)].append(record)

    repeat_sets = {tuple(sorted(item.repeat_id for item in group)) for group in by_state.values()}
    if len(repeat_sets) != 1:
        raise ValueError("all states must share one balanced repeat-id set")
    repeat_ids = next(iter(repeat_sets))
    if len(repeat_ids) < 2:
        raise ValueError("at least two same-state repeats are required to identify N")

    trajectories = sorted({record.trajectory_id for record in records})
    observed_phases: dict[str, set[str]] = defaultdict(set)
    for record in records:
        observed_phases[record.trajectory_id].add(record.phase)
    required_phase_set = set(required_phases)
    invalid_phase_trajectories = [
        trajectory
        for trajectory in trajectories
        if observed_phases[trajectory] != required_phase_set
    ]
    if invalid_phase_trajectories:
        raise ValueError(f"trajectory phase coverage mismatch: {invalid_phase_trajectories}")

    state_rows: list[dict[str, Any]] = []
    for (trajectory, phase, state), group in sorted(by_state.items()):
        effects = [item.effect for item in sorted(group, key=lambda item: item.repeat_id)]
        state_rows.append(
            {
                "trajectory_id": trajectory,
                "phase": phase,
                "state_id": state,
                "mean_effect": _mean(effects),
                "runtime_variance": _sample_variance(effects),
                "repeat_effects": effects,
            }
        )

    by_trajectory_phase: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_trajectory_phase[(row["trajectory_id"], row["phase"])].append(row)

    phase_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for phase in required_phases:
            states = by_trajectory_phase[(trajectory, phase)]
            state_means = [row["mean_effect"] for row in states]
            state_runtime = [row["runtime_variance"] for row in states]
            observed_state_variance = _sample_variance(state_means)
            mean_runtime = _mean(state_runtime)
            state_heterogeneity_unconstrained = (
                observed_state_variance - mean_runtime / len(repeat_ids)
            )
            state_heterogeneity = max(0.0, state_heterogeneity_unconstrained)
            phase_rows.append(
                {
                    "trajectory_id": trajectory,
                    "phase": phase,
                    "states": len(states),
                    "mean_effect": _mean(state_means),
                    "observed_state_mean_variance": observed_state_variance,
                    "state_heterogeneity_repeat_corrected_unconstrained": state_heterogeneity_unconstrained,
                    "state_heterogeneity_repeat_corrected": state_heterogeneity,
                    "mean_runtime_variance": mean_runtime,
                    "phase_mean_estimation_variance": observed_state_variance
                    / len(states),
                }
            )

    by_trajectory_phase_row = {
        (row["trajectory_id"], row["phase"]): row for row in phase_rows
    }
    trajectory_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        phase_effects = [
            by_trajectory_phase_row[(trajectory, phase)]["mean_effect"] for phase in required_phases
        ]
        observed_phase_variance = _sample_variance(phase_effects)
        mean_phase_estimation_variance = _mean(
            by_trajectory_phase_row[(trajectory, phase)][
                "phase_mean_estimation_variance"
            ]
            for phase in required_phases
        )
        phase_heterogeneity_unconstrained = (
            observed_phase_variance - mean_phase_estimation_variance
        )
        trajectory_rows.append(
            {
                "trajectory_id": trajectory,
                "mean_effect": _mean(phase_effects),
                "observed_phase_mean_variance": observed_phase_variance,
                "mean_phase_estimation_variance": mean_phase_estimation_variance,
                "phase_heterogeneity_corrected_unconstrained": phase_heterogeneity_unconstrained,
                "phase_heterogeneity": max(
                    0.0, phase_heterogeneity_unconstrained
                ),
                "phase_effects": dict(zip(required_phases, phase_effects, strict=True)),
            }
        )

    trajectory_effects = [row["mean_effect"] for row in trajectory_rows]
    B = _mean(trajectory_effects)
    trajectory_count = len(trajectory_effects)
    if trajectory_count >= 2:
        between_trajectory_variance: float | None = _sample_variance(trajectory_effects)
        assert between_trajectory_variance is not None
        standard_error = math.sqrt(between_trajectory_variance / trajectory_count)
        critical = _t_critical_two_sided(trajectory_count - 1, interval_alpha)
        lower = B - critical * standard_error
        upper = B + critical * standard_error
    else:
        between_trajectory_variance = None
        standard_error = None
        critical = None
        lower = None
        upper = None

    phase_conditional_rows: list[dict[str, Any]] = []
    for phase in required_phases:
        effects = [
            float(by_trajectory_phase_row[(trajectory, phase)]["mean_effect"])
            for trajectory in trajectories
        ]
        conditional_mean = _mean(effects)
        if trajectory_count >= 2:
            conditional_variance: float | None = _sample_variance(effects)
            conditional_se: float | None = math.sqrt(
                conditional_variance / trajectory_count
            )
            conditional_critical: float | None = _t_critical_two_sided(
                trajectory_count - 1, interval_alpha
            )
            conditional_interval = [
                conditional_mean - conditional_critical * conditional_se,
                conditional_mean + conditional_critical * conditional_se,
            ]
        else:
            conditional_variance = None
            conditional_se = None
            conditional_critical = None
            conditional_interval = [None, None]
        phase_conditional_rows.append(
            {
                "phase": phase,
                "estimate": conditional_mean,
                "trajectory_t_interval": conditional_interval,
                "interval_alpha": interval_alpha,
                "between_trajectory_variance": conditional_variance,
                "standard_error": conditional_se,
                "critical_value": conditional_critical,
                "degrees_of_freedom": trajectory_count - 1,
                "trajectory_effects": effects,
                "trajectory_rows": [
                    {
                        "trajectory_id": trajectory,
                        "mean_effect": effect,
                    }
                    for trajectory, effect in zip(
                        trajectories, effects, strict=True
                    )
                ],
            }
        )

    weighted_runtime_variance = _mean(
        row["mean_runtime_variance"] for row in phase_rows
    )
    weighted_state_heterogeneity_unconstrained = _mean(
        row["state_heterogeneity_repeat_corrected_unconstrained"]
        for row in phase_rows
    )
    weighted_state_heterogeneity = _mean(
        row["state_heterogeneity_repeat_corrected"] for row in phase_rows
    )
    weighted_phase_heterogeneity = _mean(row["phase_heterogeneity"] for row in trajectory_rows)
    weighted_phase_heterogeneity_unconstrained = _mean(
        row["phase_heterogeneity_corrected_unconstrained"]
        for row in trajectory_rows
    )

    def component_profile(values: list[float], label: str) -> dict[str, Any]:
        estimate = _mean(values)
        if len(values) >= 2:
            variance = _sample_variance(values)
            se = math.sqrt(variance / len(values))
            component_critical = _t_critical_two_sided(
                len(values) - 1, interval_alpha
            )
            interval = [
                estimate - component_critical * se,
                estimate + component_critical * se,
            ]
        else:
            variance = None
            se = None
            component_critical = None
            interval = [None, None]
        return {
            "component": label,
            "estimate": estimate,
            "trajectory_t_interval": interval,
            "interval_alpha": interval_alpha,
            "between_trajectory_variance_of_component_estimates": variance,
            "standard_error": se,
            "critical_value": component_critical,
            "degrees_of_freedom": len(values) - 1,
            "trajectory_component_estimates": values,
            "status": "REGULARITY_CONDITIONAL_DESCRIPTION_NO_COMPONENT_VERDICT",
        }

    state_h_by_trajectory = [
        _mean(
            row["state_heterogeneity_repeat_corrected_unconstrained"]
            for row in phase_rows
            if row["trajectory_id"] == trajectory
        )
        for trajectory in trajectories
    ]
    phase_h_by_trajectory = [
        float(row["phase_heterogeneity_corrected_unconstrained"])
        for row in trajectory_rows
    ]
    runtime_n_by_trajectory = [
        _mean(
            row["mean_runtime_variance"]
            for row in phase_rows
            if row["trajectory_id"] == trajectory
        )
        for trajectory in trajectories
    ]

    if trajectory_count < min_confirmation_trajectories:
        shift_verdict = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif lower is not None and upper is not None and (
        lower > measurement_floor or upper < -measurement_floor
    ):
        shift_verdict = "REPRODUCIBLE_AVERAGE_SHIFT"
    else:
        shift_verdict = "NO_STABLE_AVERAGE_DETECTED"

    if practical_tolerance is None:
        materiality_verdict = "UNINSTANTIATED_MATERIALITY"
    elif lower is None or upper is None:
        materiality_verdict = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif lower > practical_tolerance or upper < -practical_tolerance:
        materiality_verdict = "MATERIAL_AVERAGE_SHIFT"
    elif lower >= -practical_tolerance and upper <= practical_tolerance:
        materiality_verdict = "PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT"
    elif shift_verdict == "REPRODUCIBLE_AVERAGE_SHIFT":
        materiality_verdict = "DETECTED_BUT_MATERIALITY_INDETERMINATE"
    else:
        materiality_verdict = "INDETERMINATE_MATERIALITY"

    if tail_prevalence is None:
        required_tail_trajectories = None
        tail_verdict = "UNINSTANTIATED_TAIL_TARGET"
    else:
        required_tail_trajectories = math.ceil(math.log(tail_alpha) / math.log(1.0 - tail_prevalence))
        tail_verdict = (
            "TAIL_COVERAGE_SUFFICIENT"
            if trajectory_count >= required_tail_trajectories
            else "TAIL_COVERAGE_INSUFFICIENT"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "construction": {
            "verdict": "VALID",
            "trajectories": trajectory_count,
            "required_phases": list(required_phases),
            "states": len(state_rows),
            "repeats_per_state": len(repeat_ids),
            "weighting": "equal trajectory; equal required phase within trajectory; equal state within phase",
        },
        "B": {
            "estimate": B,
            "trajectory_t_interval": [lower, upper],
            "interval_alpha": interval_alpha,
            "nominal_interval_coverage": 1.0 - interval_alpha,
            "trajectory_t_interval_95": (
                [lower, upper] if math.isclose(interval_alpha, 0.05) else None
            ),
            "trajectory_t_interval_95_status": (
                "PRIMARY_INTERVAL"
                if math.isclose(interval_alpha, 0.05)
                else "NOT_APPLICABLE_ADJUSTED_INTERVAL_USE_trajectory_t_interval"
            ),
            "standard_error": standard_error,
            "critical_value": critical,
            "degrees_of_freedom": trajectory_count - 1,
            "measurement_floor": measurement_floor,
        },
        "conditional_B": {
            "predeclared_phase_rows": phase_conditional_rows,
            "status": "DESCRIPTIVE_ONLY_PHASE_NOT_IN_FROZEN_MULTIPLICITY_FAMILY",
            "interpretation": "each row is B under P(state | declared phase), not runtime variance and not a global-B verdict",
            "operator_attribution_allowed": False,
        },
        "H": {
            "between_trajectory_variance": between_trajectory_variance,
            "mean_within_trajectory_phase_variance": weighted_phase_heterogeneity,
            "mean_within_trajectory_phase_variance_unconstrained": weighted_phase_heterogeneity_unconstrained,
            "mean_within_phase_state_variance_repeat_corrected": weighted_state_heterogeneity,
            "mean_within_phase_state_variance_repeat_corrected_unconstrained": weighted_state_heterogeneity_unconstrained,
            "uncertainty": {
                "phase_component": component_profile(
                    phase_h_by_trajectory, "phase_heterogeneity_corrected"
                ),
                "state_component": component_profile(
                    state_h_by_trajectory,
                    "within_phase_state_heterogeneity_repeat_corrected",
                ),
            },
            "nonnegative_values_are_truncated_descriptions": True,
            "components_are_not_additive_without_a_full_hierarchical_variance_model": True,
        },
        "N": {
            "mean_same_state_paired_effect_variance": weighted_runtime_variance,
            "uncertainty": component_profile(
                runtime_n_by_trajectory, "same_state_paired_effect_variance"
            ),
        },
        "U": {
            "primary_unit": "independent trajectory",
            "method": "trajectory-level Student t interval",
            "interval_alpha": interval_alpha,
            "states_tokens_coordinates_do_not_increase_top_level_df": True,
        },
        "identification_assumptions": {
            "paired_effect_definition": "candidate minus reference under the declared same-state randomness coupling",
            "repeat_exchangeability": "same-state paired-effect repeats are exchangeable with conditional mean-zero runtime disturbance",
            "paired_noise_covariance": "N is estimated from paired differences and therefore retains reference/candidate covariance; arm variances are not simply added",
            "state_effect_definition": "deterministic reduction tree, reassociation, cast placement and other fixed realization differences belong to m(state), not N",
            "trajectory_independence": "top-level intervals require independently generated trajectories under the frozen state-sampling design",
            "H_correction": "within-phase state H uses a balanced-repeat method-of-moments correction; unconstrained and truncated descriptions are both retained",
            "mechanism_nonclassification": "floating point, reduction and compiler optimization are not intrinsically labelled bias or variance mechanisms",
        },
        "tail_coverage": {
            "target_prevalence": tail_prevalence,
            "alpha": tail_alpha if tail_prevalence is not None else None,
            "required_independent_trajectories_for_at_least_one_observation": required_tail_trajectories,
            "verdict": tail_verdict,
            "scope": "design probability only; does not estimate tail effect or prevalence",
        },
        "verdicts": {
            "shift_existence": shift_verdict,
            "materiality": materiality_verdict,
            "tail_coverage": tail_verdict,
        },
        "trajectory_rows": trajectory_rows,
        "phase_rows": phase_rows,
        "state_rows": state_rows,
    }
