"""Trajectory-aware repair-contribution profiles for the Bias Oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from theory_oracle.bias_oracle_population_v0_2 import (
    EffectRecord,
    estimate_scalar_population,
)
from theory_oracle.bias_oracle_trajectory_signflip_v0_1 import (
    trajectory_signflip_test,
)


SCHEMA_VERSION = "forkcert.bias-oracle-contributor-profile.v0.1"
REQUIRED_ARMS = ("REFERENCE", "FULL_CANDIDATE", "CANDIDATE_REPAIR")
INJECTION_ARMS = ("REFERENCE", "FULL_CANDIDATE", "REFERENCE_INJECTION")


@dataclass(frozen=True)
class ContributorArmRecord:
    trajectory_id: str
    phase: str
    state_id: str
    repeat_id: int
    arm: str
    outcome: float


def _effect_records(
    records: list[ContributorArmRecord],
) -> tuple[list[EffectRecord], list[EffectRecord], list[EffectRecord]]:
    if not records:
        raise ValueError("no contributor arm records")
    cells: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for record in records:
        if record.arm not in REQUIRED_ARMS:
            raise ValueError(f"unsupported contributor arm: {record.arm}")
        if not math.isfinite(record.outcome):
            raise ValueError("all contributor outcomes must be finite")
        key = (record.trajectory_id, record.phase, record.state_id, record.repeat_id)
        arms = cells.setdefault(key, {})
        if record.arm in arms:
            raise ValueError(f"duplicate contributor arm cell: {key}/{record.arm}")
        arms[record.arm] = record.outcome
    incomplete = [key for key, arms in cells.items() if set(arms) != set(REQUIRED_ARMS)]
    if incomplete:
        raise ValueError(f"incomplete contributor arm cells: {incomplete[:5]}")

    baseline: list[EffectRecord] = []
    residual: list[EffectRecord] = []
    contribution: list[EffectRecord] = []
    for (trajectory, phase, state, repeat), arms in sorted(cells.items()):
        common = {
            "trajectory_id": trajectory,
            "phase": phase,
            "state_id": state,
            "repeat_id": repeat,
        }
        baseline.append(
            EffectRecord(
                **common,
                effect=arms["FULL_CANDIDATE"] - arms["REFERENCE"],
            )
        )
        residual.append(
            EffectRecord(
                **common,
                effect=arms["CANDIDATE_REPAIR"] - arms["REFERENCE"],
            )
        )
        contribution.append(
            EffectRecord(
                **common,
                effect=arms["FULL_CANDIDATE"] - arms["CANDIDATE_REPAIR"],
            )
        )
    return baseline, residual, contribution


def _directional_lower(interval: list[float | None], direction: int) -> float | None:
    lower, upper = interval
    if lower is None or upper is None:
        return None
    return float(lower) if direction == 1 else -float(upper)


def _directional_upper(interval: list[float | None], direction: int) -> float | None:
    lower, upper = interval
    if lower is None or upper is None:
        return None
    return float(upper) if direction == 1 else -float(lower)


def _absolute_bounds(interval: list[float | None]) -> tuple[float | None, float | None]:
    lower, upper = interval
    if lower is None or upper is None:
        return None, None
    lower_f, upper_f = float(lower), float(upper)
    absolute_upper = max(abs(lower_f), abs(upper_f))
    absolute_lower = 0.0 if lower_f <= 0.0 <= upper_f else min(abs(lower_f), abs(upper_f))
    return absolute_lower, absolute_upper


def _injection_effect_records(
    records: list[ContributorArmRecord],
) -> tuple[list[EffectRecord], list[EffectRecord]]:
    if not records:
        raise ValueError("no injection arm records")
    cells: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for record in records:
        if record.arm not in INJECTION_ARMS:
            raise ValueError(f"unsupported injection arm: {record.arm}")
        if not math.isfinite(record.outcome):
            raise ValueError("all injection outcomes must be finite")
        key = (record.trajectory_id, record.phase, record.state_id, record.repeat_id)
        arms = cells.setdefault(key, {})
        if record.arm in arms:
            raise ValueError(f"duplicate injection arm cell: {key}/{record.arm}")
        arms[record.arm] = record.outcome
    incomplete = [key for key, arms in cells.items() if set(arms) != set(INJECTION_ARMS)]
    if incomplete:
        raise ValueError(f"incomplete injection arm cells: {incomplete[:5]}")

    baseline: list[EffectRecord] = []
    injection: list[EffectRecord] = []
    for (trajectory, phase, state, repeat), arms in sorted(cells.items()):
        common = {
            "trajectory_id": trajectory,
            "phase": phase,
            "state_id": state,
            "repeat_id": repeat,
        }
        baseline.append(
            EffectRecord(
                **common,
                effect=arms["FULL_CANDIDATE"] - arms["REFERENCE"],
            )
        )
        injection.append(
            EffectRecord(
                **common,
                effect=arms["REFERENCE_INJECTION"] - arms["REFERENCE"],
            )
        )
    return baseline, injection


def estimate_injection_contribution(
    records: list[ContributorArmRecord],
    *,
    required_phases: tuple[str, ...],
    frozen_bias_direction: int,
    target_phase: str | None = None,
    min_confirmation_trajectories: int = 8,
    baseline_transport_floor: float = 0.0,
    directional_injection_floor: float = 0.0,
    primary_interval_alpha: float = 0.05,
    tail_prevalence: float | None = None,
    tail_alpha: float = 0.05,
) -> dict[str, Any]:
    if frozen_bias_direction not in (-1, 1):
        raise ValueError("frozen_bias_direction must be -1 or +1")
    if baseline_transport_floor < 0 or directional_injection_floor < 0:
        raise ValueError("directional floors must be non-negative")
    effective_required_phases = required_phases
    if target_phase is not None:
        if target_phase not in required_phases:
            raise ValueError("target_phase must belong to the declared required_phases")
        records = [record for record in records if record.phase == target_phase]
        if not records:
            raise ValueError("target_phase has no injection records")
        effective_required_phases = (target_phase,)

    baseline_records, injection_records = _injection_effect_records(records)
    common = {
        "required_phases": effective_required_phases,
        "min_confirmation_trajectories": min_confirmation_trajectories,
        "tail_prevalence": tail_prevalence,
        "tail_alpha": tail_alpha,
        "interval_alpha": primary_interval_alpha,
    }
    baseline = estimate_scalar_population(
        baseline_records,
        measurement_floor=baseline_transport_floor,
        **common,
    )
    injection = estimate_scalar_population(
        injection_records,
        measurement_floor=directional_injection_floor,
        **common,
    )
    baseline_lower = _directional_lower(
        baseline["B"]["trajectory_t_interval"], frozen_bias_direction
    )
    baseline_upper = _directional_upper(
        baseline["B"]["trajectory_t_interval"], frozen_bias_direction
    )
    injection_lower = _directional_lower(
        injection["B"]["trajectory_t_interval"], frozen_bias_direction
    )
    injection_upper = _directional_upper(
        injection["B"]["trajectory_t_interval"], frozen_bias_direction
    )
    if baseline_lower is None:
        baseline_transport = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif baseline_lower > baseline_transport_floor:
        baseline_transport = "BASELINE_BIAS_TRANSPORTED_IN_FROZEN_DIRECTION"
    elif baseline_upper is not None and baseline_upper < -baseline_transport_floor:
        baseline_transport = "BASELINE_TRANSPORT_FAILED_OPPOSITE_DIRECTION"
    else:
        baseline_transport = "BASELINE_TRANSPORT_INDETERMINATE"
    if injection_lower is None:
        injection_verdict = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif injection_lower > directional_injection_floor:
        injection_verdict = "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_INJECTION"
    elif injection_upper is not None and injection_upper < -directional_injection_floor:
        injection_verdict = "INJECTION_MOVES_AGAINST_FROZEN_BIAS_DIRECTION"
    else:
        injection_verdict = "NO_CONFIRMED_DIRECTIONAL_INJECTION"
    primary = (
        "INDETERMINATE_BASELINE_BIAS_DID_NOT_TRANSPORT"
        if baseline_transport != "BASELINE_BIAS_TRANSPORTED_IN_FROZEN_DIRECTION"
        else (
            "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_INJECTION_SENSITIVITY_PENDING"
            if injection_verdict == "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_INJECTION"
            else injection_verdict
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_type": "REFERENCE_CONTEXT_INJECTION",
        "construction": {
            "valid": True,
            "required_arms": list(INJECTION_ARMS),
            "paired_arm_cells": len(baseline_records),
            "target_phase": target_phase,
            "target_distribution": (
                f"P(state | phase={target_phase})"
                if target_phase is not None
                else "declared global state distribution"
            ),
        },
        "frozen_bias_direction": frozen_bias_direction,
        "profiles": {
            "baseline_candidate_minus_reference": baseline,
            "reference_context_injection_minus_reference": injection,
        },
        "directional_gate": {
            "baseline_transport_floor": baseline_transport_floor,
            "directional_injection_floor": directional_injection_floor,
            "baseline_transport_verdict": baseline_transport,
            "directional_injection_verdict": injection_verdict,
        },
        "primary_interval_verdict": primary,
        "final_claim_allowed": False,
        "remaining_gates": [
            "trajectory-level sensitivity",
            "candidate-specific intervention integrity",
            "separately frozen injection hypothesis family",
        ],
        "nonclaims": [
            "reference-context injection is not sufficiency",
            "repair and injection effects need not be symmetric",
            "injection attribution is intervention-dependent and not correctness",
        ],
    }


def estimate_repair_contribution(
    records: list[ContributorArmRecord],
    *,
    required_phases: tuple[str, ...],
    frozen_bias_direction: int,
    target_phase: str | None = None,
    min_confirmation_trajectories: int = 8,
    baseline_transport_floor: float = 0.0,
    directional_contribution_floor: float = 0.0,
    primary_interval_alpha: float = 0.05,
    tail_prevalence: float | None = None,
    tail_alpha: float = 0.05,
    absolute_reduction_enabled: bool = False,
    absolute_reduction_interval_alpha: float | None = None,
    minimum_absolute_reduction: float = 0.0,
) -> dict[str, Any]:
    if frozen_bias_direction not in (-1, 1):
        raise ValueError("frozen_bias_direction must be -1 or +1")
    if baseline_transport_floor < 0 or directional_contribution_floor < 0:
        raise ValueError("directional floors must be non-negative")
    if minimum_absolute_reduction < 0:
        raise ValueError("minimum_absolute_reduction must be non-negative")
    if absolute_reduction_enabled and absolute_reduction_interval_alpha is None:
        raise ValueError("absolute reduction claims require a separately frozen simultaneous alpha")

    effective_required_phases = required_phases
    if target_phase is not None:
        if target_phase not in required_phases:
            raise ValueError("target_phase must belong to the declared required_phases")
        records = [record for record in records if record.phase == target_phase]
        if not records:
            raise ValueError("target_phase has no contributor records")
        effective_required_phases = (target_phase,)

    baseline_records, residual_records, contribution_records = _effect_records(records)
    common = {
        "required_phases": effective_required_phases,
        "min_confirmation_trajectories": min_confirmation_trajectories,
        "tail_prevalence": tail_prevalence,
        "tail_alpha": tail_alpha,
    }
    baseline = estimate_scalar_population(
        baseline_records,
        measurement_floor=baseline_transport_floor,
        interval_alpha=(
            float(absolute_reduction_interval_alpha)
            if absolute_reduction_enabled
            else primary_interval_alpha
        ),
        **common,
    )
    residual = estimate_scalar_population(
        residual_records,
        measurement_floor=0.0,
        interval_alpha=(
            float(absolute_reduction_interval_alpha)
            if absolute_reduction_enabled
            else primary_interval_alpha
        ),
        **common,
    )
    contribution = estimate_scalar_population(
        contribution_records,
        measurement_floor=directional_contribution_floor,
        interval_alpha=primary_interval_alpha,
        **common,
    )

    baseline_interval = baseline["B"]["trajectory_t_interval"]
    residual_interval = residual["B"]["trajectory_t_interval"]
    contribution_interval = contribution["B"]["trajectory_t_interval"]
    baseline_lower = _directional_lower(baseline_interval, frozen_bias_direction)
    baseline_upper = _directional_upper(baseline_interval, frozen_bias_direction)
    contribution_lower = _directional_lower(contribution_interval, frozen_bias_direction)
    contribution_upper = _directional_upper(contribution_interval, frozen_bias_direction)
    residual_lower = _directional_lower(residual_interval, frozen_bias_direction)
    residual_upper = _directional_upper(residual_interval, frozen_bias_direction)

    if baseline_lower is None:
        baseline_transport = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif baseline_lower > baseline_transport_floor:
        baseline_transport = "BASELINE_BIAS_TRANSPORTED_IN_FROZEN_DIRECTION"
    elif baseline_upper is not None and baseline_upper < -baseline_transport_floor:
        baseline_transport = "BASELINE_TRANSPORT_FAILED_OPPOSITE_DIRECTION"
    else:
        baseline_transport = "BASELINE_TRANSPORT_INDETERMINATE"

    if contribution_lower is None:
        directional_verdict = "INDETERMINATE_TOO_FEW_TRAJECTORIES"
    elif contribution_lower > directional_contribution_floor:
        directional_verdict = "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTION"
    elif contribution_upper is not None and contribution_upper < -directional_contribution_floor:
        directional_verdict = "REPAIR_MOVES_AGAINST_FROZEN_BIAS_DIRECTION"
    else:
        directional_verdict = "NO_CONFIRMED_DIRECTIONAL_CONTRIBUTION"

    baseline_abs_lower, baseline_abs_upper = _absolute_bounds(baseline_interval)
    residual_abs_lower, residual_abs_upper = _absolute_bounds(residual_interval)
    absolute_reduction_lower_bound = None
    absolute_reduction_verdict = "UNINSTANTIATED_ABSOLUTE_REDUCTION_CLAIM"
    overshoot_established = False
    if absolute_reduction_enabled:
        assert baseline_abs_lower is not None and baseline_abs_upper is not None
        assert residual_abs_lower is not None and residual_abs_upper is not None
        absolute_reduction_lower_bound = baseline_abs_lower - residual_abs_upper
        absolute_reduction_verdict = (
            "SIMULTANEOUS_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCTION"
            if absolute_reduction_lower_bound > minimum_absolute_reduction
            else "ABSOLUTE_BIAS_REDUCTION_NOT_ESTABLISHED"
        )
        overshoot_established = (
            residual_upper is not None
            and residual_upper < 0.0
            and residual_abs_lower > baseline_abs_upper
        )

    if baseline_transport != "BASELINE_BIAS_TRANSPORTED_IN_FROZEN_DIRECTION":
        primary_verdict = "INDETERMINATE_BASELINE_BIAS_DID_NOT_TRANSPORT"
    elif directional_verdict != "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTION":
        primary_verdict = directional_verdict
    elif overshoot_established:
        primary_verdict = "DIRECTIONAL_CONTRIBUTOR_WITH_OVERSHOOT_SENSITIVITY_PENDING"
    elif absolute_reduction_verdict == "SIMULTANEOUS_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCTION":
        primary_verdict = "PRIMARY_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCER_SENSITIVITY_PENDING"
    else:
        primary_verdict = "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTOR_SENSITIVITY_PENDING"

    return {
        "schema_version": SCHEMA_VERSION,
        "construction": {
            "valid": True,
            "required_arms": list(REQUIRED_ARMS),
            "paired_arm_cells": len(baseline_records),
            "reference_cancels_algebraically_from_contribution": True,
            "same_state_arm_covariance_retained": True,
            "target_phase": target_phase,
            "target_distribution": (
                f"P(state | phase={target_phase})"
                if target_phase is not None
                else "declared global state distribution"
            ),
        },
        "frozen_bias_direction": frozen_bias_direction,
        "profiles": {
            "baseline_candidate_minus_reference": baseline,
            "residual_repair_minus_reference": residual,
            "repair_contribution_candidate_minus_repair": contribution,
        },
        "directional_gate": {
            "baseline_transport_floor": baseline_transport_floor,
            "directional_contribution_floor": directional_contribution_floor,
            "baseline_directional_interval_lower": baseline_lower,
            "contribution_directional_interval_lower": contribution_lower,
            "baseline_transport_verdict": baseline_transport,
            "directional_contribution_verdict": directional_verdict,
        },
        "absolute_reduction_gate": {
            "enabled": absolute_reduction_enabled,
            "simultaneous_interval_alpha": absolute_reduction_interval_alpha,
            "minimum_absolute_reduction": minimum_absolute_reduction,
            "conservative_reduction_lower_bound": absolute_reduction_lower_bound,
            "verdict": absolute_reduction_verdict,
            "overshoot_established": overshoot_established,
        },
        "primary_interval_verdict": primary_verdict,
        "final_claim_allowed": False,
        "remaining_gates": [
            "trajectory-level sensitivity",
            "candidate-specific intervention integrity",
            "tail/regularity scope",
            "complete frozen hypothesis family",
        ],
        "nonclaims": [
            "directional contribution is not root cause, necessity or sufficiency",
            "repair contribution is implementation-relative, not correctness",
            "a point-estimate distance reduction is not an absolute-Bias reduction claim",
        ],
    }


def apply_contribution_sensitivity(
    profile: dict[str, Any],
    *,
    decision_alpha: float,
    exact_max_trajectories: int = 16,
    monte_carlo_draws: int = 99999,
    monte_carlo_seed: int = 172904,
) -> dict[str, Any]:
    if not 0.0 < decision_alpha < 1.0:
        raise ValueError("decision_alpha must lie between zero and one")
    contribution = profile.get("profiles", {}).get(
        "repair_contribution_candidate_minus_repair", {}
    )
    interval_alpha = contribution.get("B", {}).get("interval_alpha")
    if not isinstance(interval_alpha, (int, float)) or not math.isclose(
        float(interval_alpha), decision_alpha, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("sensitivity alpha must equal the frozen primary interval alpha")
    primary = profile.get("primary_interval_verdict")
    supported_primary = {
        "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTOR_SENSITIVITY_PENDING",
        "PRIMARY_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCER_SENSITIVITY_PENDING",
        "DIRECTIONAL_CONTRIBUTOR_WITH_OVERSHOOT_SENSITIVITY_PENDING",
    }
    if primary not in supported_primary:
        return {
            "status": "NOT_RUN_PRIMARY_INTERVAL_DID_NOT_SUPPORT_CONTRIBUTION",
            "role": "sensitivity cannot promote a primary failure",
            "primary_interval_verdict": primary,
            "post_sensitivity_verdict": primary,
            "final_claim_allowed": False,
        }
    direction = profile.get("frozen_bias_direction")
    if direction not in (-1, 1):
        raise ValueError("profile lacks a frozen Bias direction")
    trajectory_rows = contribution.get("trajectory_rows", [])
    effects = [direction * float(row["mean_effect"]) for row in trajectory_rows]
    floor = float(profile["directional_gate"]["directional_contribution_floor"])
    sensitivity = trajectory_signflip_test(
        effects,
        null_center=floor,
        exact_max_trajectories=exact_max_trajectories,
        monte_carlo_draws=monte_carlo_draws,
        monte_carlo_seed=monte_carlo_seed,
    )
    sensitivity["decision_alpha"] = decision_alpha
    sensitivity["supports_primary_contribution"] = (
        sensitivity["two_sided_p_value"] <= decision_alpha
    )
    if sensitivity["supports_primary_contribution"]:
        post = primary.replace(
            "_SENSITIVITY_PENDING", "_SENSITIVITY_SUPPORTED_INTEGRITY_PENDING"
        )
    else:
        post = "INDETERMINATE_METHOD_SENSITIVITY"
    return {
        "status": "SENSITIVITY_EVALUATED",
        "role": "veto only; cannot promote the primary interval",
        "primary_interval_verdict": primary,
        "sensitivity": sensitivity,
        "post_sensitivity_verdict": post,
        "final_claim_allowed": False,
        "remaining_gate": "candidate-specific intervention integrity",
    }


def apply_injection_sensitivity(
    profile: dict[str, Any],
    *,
    decision_alpha: float,
    exact_max_trajectories: int = 16,
    monte_carlo_draws: int = 99999,
    monte_carlo_seed: int = 172904,
) -> dict[str, Any]:
    if not 0.0 < decision_alpha < 1.0:
        raise ValueError("decision_alpha must lie between zero and one")
    injection = profile.get("profiles", {}).get(
        "reference_context_injection_minus_reference", {}
    )
    interval_alpha = injection.get("B", {}).get("interval_alpha")
    if not isinstance(interval_alpha, (int, float)) or not math.isclose(
        float(interval_alpha), decision_alpha, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("sensitivity alpha must equal the frozen injection interval alpha")
    primary = profile.get("primary_interval_verdict")
    if primary != "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_INJECTION_SENSITIVITY_PENDING":
        return {
            "status": "NOT_RUN_PRIMARY_INTERVAL_DID_NOT_SUPPORT_INJECTION",
            "role": "sensitivity cannot promote a primary failure",
            "primary_interval_verdict": primary,
            "post_sensitivity_verdict": primary,
            "final_claim_allowed": False,
        }
    direction = profile.get("frozen_bias_direction")
    if direction not in (-1, 1):
        raise ValueError("profile lacks a frozen Bias direction")
    effects = [
        direction * float(row["mean_effect"])
        for row in injection.get("trajectory_rows", [])
    ]
    floor = float(profile["directional_gate"]["directional_injection_floor"])
    sensitivity = trajectory_signflip_test(
        effects,
        null_center=floor,
        exact_max_trajectories=exact_max_trajectories,
        monte_carlo_draws=monte_carlo_draws,
        monte_carlo_seed=monte_carlo_seed,
    )
    sensitivity["decision_alpha"] = decision_alpha
    sensitivity["supports_primary_injection"] = (
        sensitivity["two_sided_p_value"] <= decision_alpha
    )
    post = (
        "DIRECTIONAL_INJECTION_SENSITIVITY_SUPPORTED_INTEGRITY_PENDING"
        if sensitivity["supports_primary_injection"]
        else "INDETERMINATE_METHOD_SENSITIVITY"
    )
    return {
        "status": "SENSITIVITY_EVALUATED",
        "role": "veto only; cannot promote the primary interval",
        "primary_interval_verdict": primary,
        "sensitivity": sensitivity,
        "post_sensitivity_verdict": post,
        "final_claim_allowed": False,
        "remaining_gate": "candidate-specific injection integrity",
    }
