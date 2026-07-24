#!/usr/bin/env python
"""Prospectively size independent confirmation from four calibration trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "forkcert.bias-oracle-confirmation-precision.v0.1"
SPEC_VERSION = "forkcert.bias-oracle-confirmation-precision-spec.v0.1"
THRESHOLD_SOURCE_KINDS = {
    "desired_half_width": {
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
        "EXTERNAL_SCIENTIFIC_TOLERANCE",
    },
    "variance_floor_sd": {
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
        "EXTERNAL_CONSERVATIVE_VARIANCE_FLOOR",
    },
    "shift_existence_floor": {
        "EXACT_ZERO_NULL",
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
    },
}
REQUIRED_PHASES = ("early", "middle", "late")
U2_DIRECTION_VERSION = "forkcert.qwen3-u2-frozen-direction.v0.1"
U2_ENDPOINT = "U2_calibration_direction_shift"
PRECISION_TARGETED = "PRECISION_TARGETED"
FIXED_RESOURCE_EXISTENCE = "FIXED_RESOURCE_EXISTENCE"
FIXED_RESOURCE_SOURCE_KINDS = {
    "EXTERNAL_COMPUTE_BUDGET",
    "PREDECLARED_RESOURCE_ALLOCATION",
}
SIGNED_B_ENDPOINT_CLASSES = {
    "SIGNED_SCALAR_IMPLEMENTATION_SHIFT",
    "SIGNED_UPDATE_GEOMETRY_ENDPOINT",
    "SIGNED_UPDATE_ALIGNED_FORCING_ENDPOINT",
    "SIGNED_TASK_ENDPOINT",
    "SIGNED_STATE_ADAPTIVE_TASK_ENDPOINT",
    "SIGNED_FIXED_BANK_TASK_ENDPOINT",
    "SIGNED_EVENT_COUNT_SHIFT",
    "SIGNED_EVENT_PROBABILITY_SHIFT",
}
EXPECTED_ENDPOINT_ROLE_CATALOG = {
    "core_signed_bias_candidates": [
        "U1_reference_aligned_dot",
        "T1a_heldout_grpo_shift",
        "T1b_correct_answer_nll_shift",
        U2_ENDPOINT,
    ],
    "optional_signed_numerical_bias_candidates": [
        "training_loss_shift",
        "U1_reference_aligned_shift",
        "clip_count_shift",
    ],
    "optional_signed_semantic_bias_candidates": [
        "clip_directional_rate_shift",
        "gradient_clip_trigger_shift",
        "optimizer_skip_shift",
    ],
    "descriptive_nonnegative_profiles_not_bias": [
        "U2_paired_delta_l2",
        "clip_disagreement_rate",
        "clip_off_to_on_rate",
        "clip_on_to_off_rate",
        "clip_decision_exposure_count",
        "gradient_clip_trigger_disagreement",
        "optimizer_skip_disagreement",
    ],
    "freeze_rule": "after calibration availability/design review and before any confirmation outcome, freeze the exact signed endpoint family and multiplicity; descriptive profiles never enter signed-B confirmation",
}
EXPECTED_CALIBRATION_ANALYSIS_FILES = {
    "multi_trajectory_aggregator": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_calibration_multi_trajectory_v0_1.py",
    "record_loader": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_calibration_records_v0_1.py",
    "population_estimator": ROOT / "theory_oracle" / "bias_oracle_population_v0_2.py",
    "record_validator": ROOT / "theory_oracle" / "bias_oracle_record_v0_2.py",
    "task_semantics_validator": ROOT
    / "theory_oracle"
    / "evaluate_qwen3_calibration_state_endpoints_v0_1.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two independent calibration trajectories are required")
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def validate_threshold_sources(endpoint: dict[str, Any], name: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    sources = endpoint.get("threshold_sources")
    if not isinstance(sources, dict) or set(sources) != set(THRESHOLD_SOURCE_KINDS):
        return {}, [f"{name}: threshold_sources must separately cover desired width, variance floor and shift floor"]
    for threshold, allowed_kinds in THRESHOLD_SOURCE_KINDS.items():
        source = sources.get(threshold)
        if not isinstance(source, dict):
            errors.append(f"{name}: {threshold} source must be an auditable object")
            continue
        if source.get("kind") not in allowed_kinds:
            errors.append(f"{name}: unsupported {threshold} source kind")
        if not all(
            isinstance(source.get(field), str) and bool(source[field].strip())
            for field in ("description", "selection_rule")
        ):
            errors.append(f"{name}: {threshold} source requires description and selection_rule")
        if source.get("uses_calibration_candidate_mean_or_sign") is not False:
            errors.append(f"{name}: {threshold} source must exclude calibration candidate mean/sign")
    return sources, errors


def validate_existence_threshold_source(
    endpoint: dict[str, Any], name: str
) -> tuple[dict[str, Any], list[str]]:
    """Validate only the null/floor source needed by fixed-resource Mode E."""

    sources = endpoint.get("threshold_sources")
    if not isinstance(sources, dict):
        return {}, [f"{name}: threshold_sources must contain shift_existence_floor"]
    source = sources.get("shift_existence_floor")
    if not isinstance(source, dict):
        return {}, [f"{name}: shift_existence_floor source must be auditable"]
    errors: list[str] = []
    if source.get("kind") not in THRESHOLD_SOURCE_KINDS["shift_existence_floor"]:
        errors.append(f"{name}: unsupported shift_existence_floor source kind")
    if not all(
        isinstance(source.get(field), str) and bool(source[field].strip())
        for field in ("description", "selection_rule")
    ):
        errors.append(
            f"{name}: shift_existence_floor source requires description and selection_rule"
        )
    if source.get("uses_calibration_candidate_mean_or_sign") is not False:
        errors.append(
            f"{name}: shift_existence_floor source must exclude calibration candidate mean/sign"
        )
    return {"shift_existence_floor": source}, errors


def variance_upper_bound(
    sample_variance_value: float,
    calibration_trajectories: int,
    confidence: float,
) -> float:
    if sample_variance_value < 0:
        raise ValueError("sample variance must be non-negative")
    if calibration_trajectories < 2:
        raise ValueError("at least two calibration trajectories are required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("variance upper-bound confidence must lie between 0.5 and 1")
    from scipy.stats import chi2

    df = calibration_trajectories - 1
    lower_quantile = float(chi2.ppf(1.0 - confidence, df))
    if lower_quantile <= 0:
        raise ValueError("invalid chi-square quantile")
    return df * sample_variance_value / lower_quantile


def required_trajectories_for_half_width(
    variance_plan: float,
    desired_half_width: float,
    interval_alpha: float,
    minimum: int,
    resource_cap: int,
) -> int | None:
    if variance_plan <= 0:
        return None
    if desired_half_width <= 0:
        raise ValueError("desired half-width must be positive")
    if not 0 < interval_alpha < 1:
        raise ValueError("interval alpha must lie between zero and one")
    if minimum < 2 or resource_cap < minimum:
        raise ValueError("invalid trajectory minimum/resource cap")
    from scipy.stats import t

    for trajectories in range(minimum, resource_cap + 1):
        critical = float(t.ppf(1.0 - interval_alpha / 2.0, trajectories - 1))
        planned_half_width = critical * math.sqrt(variance_plan / trajectories)
        if planned_half_width <= desired_half_width:
            return trajectories
    return None


def tail_trajectory_requirement(tail: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    scope = tail.get("scope")
    if scope == "REGULARITY_CONDITIONAL_ONLY":
        return 0, {
            "scope": scope,
            "required_trajectories": 0,
            "claim": "no explicit rare-trajectory prevalence coverage",
        }
    if scope != "EXPLICIT_PREVALENCE_COVERAGE":
        raise ValueError("tail.scope must explicitly select regularity-only or prevalence coverage")
    prevalence = float(tail["minimum_prevalence"])
    alpha = float(tail["alpha"])
    if not 0 < prevalence < 1 or not 0 < alpha < 1:
        raise ValueError("tail prevalence and alpha must lie between zero and one")
    required = math.ceil(math.log(alpha) / math.log(1.0 - prevalence))
    return required, {
        "scope": scope,
        "minimum_prevalence": prevalence,
        "alpha": alpha,
        "required_trajectories": required,
        "claim": "design probability of observing at least one trajectory from a regime at or above the declared prevalence",
    }


def minimum_attainable_signflip_p(
    trajectories: int,
    *,
    exact_max_trajectories: int = 16,
    monte_carlo_draws: int = 99999,
) -> float:
    if trajectories < 2:
        raise ValueError("at least two trajectories are required")
    if exact_max_trajectories < 2 or monte_carlo_draws < 999:
        raise ValueError("invalid sign-flip sensitivity budget")
    if trajectories <= exact_max_trajectories:
        return 2.0 / (2**trajectories)
    return 1.0 / (monte_carlo_draws + 1.0)


def signflip_resolution_requirement(
    decision_alpha: float,
    minimum: int,
    resource_cap: int,
    *,
    exact_max_trajectories: int = 16,
    monte_carlo_draws: int = 99999,
) -> int | None:
    if not 0.0 < decision_alpha < 1.0:
        raise ValueError("sign-flip decision alpha must lie between zero and one")
    if minimum < 2 or resource_cap < minimum:
        raise ValueError("invalid sign-flip trajectory minimum/resource cap")
    for trajectories in range(minimum, resource_cap + 1):
        if minimum_attainable_signflip_p(
            trajectories,
            exact_max_trajectories=exact_max_trajectories,
            monte_carlo_draws=monte_carlo_draws,
        ) <= decision_alpha:
            return trajectories
    return None


def plan_confirmation(
    calibration: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []

    def numeric(name: str) -> float:
        try:
            value = float(spec.get(name))
        except (TypeError, ValueError):
            errors.append(f"{name} must be instantiated as a finite number")
            return math.nan
        if not math.isfinite(value):
            errors.append(f"{name} must be instantiated as a finite number")
            return math.nan
        return value

    def integer(name: str) -> int:
        value = spec.get(name)
        if isinstance(value, bool):
            errors.append(f"{name} must be instantiated as an integer")
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be instantiated as an integer")
            return 0
        if isinstance(value, float) and not value.is_integer():
            errors.append(f"{name} must be instantiated as an integer")
            return 0
        return parsed

    if spec.get("schema_version") != SPEC_VERSION:
        errors.append("unsupported precision spec schema")
    if spec.get("status") != "FROZEN_BEFORE_CONFIRMATION":
        errors.append("precision spec status is not FROZEN_BEFORE_CONFIRMATION")
    planning_mode = spec.get("planning_mode", PRECISION_TARGETED)
    if planning_mode not in {PRECISION_TARGETED, FIXED_RESOURCE_EXISTENCE}:
        errors.append("planning_mode must select precision-targeted or fixed-resource existence")
    construction = calibration.get("construction", {})
    if calibration.get("valid") is not True:
        errors.append("calibration aggregate is invalid")
    if construction.get("trajectories") != 4 or construction.get("top_level_df") != 3:
        errors.append("precision planning requires exactly four complete calibration trajectories")
    calibration_analysis = calibration.get("analysis_code")
    if not isinstance(calibration_analysis, dict):
        errors.append("calibration aggregate lacks analysis-code provenance")
    else:
        for name, expected_path in EXPECTED_CALIBRATION_ANALYSIS_FILES.items():
            link = calibration_analysis.get(name)
            if not isinstance(link, dict) or not isinstance(link.get("path"), str):
                errors.append(f"calibration analysis {name} link is missing")
                continue
            observed_path = Path(link["path"]).resolve()
            if observed_path != expected_path.resolve():
                errors.append(f"calibration analysis {name} path drifted")
            elif not observed_path.is_file() or sha256_file(observed_path) != link.get(
                "sha256"
            ):
                errors.append(f"calibration analysis {name} hash drifted")

    family = spec.get("endpoint_family")
    if not isinstance(family, list) or not family or len(family) != len(set(family)):
        errors.append("endpoint_family must be a unique non-empty list")
        family = []
    catalog = spec.get("endpoint_role_catalog")
    if catalog != EXPECTED_ENDPOINT_ROLE_CATALOG:
        errors.append("endpoint_role_catalog is missing or drifted")
    allowed_signed_endpoints = set(
        EXPECTED_ENDPOINT_ROLE_CATALOG["core_signed_bias_candidates"]
    ) | set(
        EXPECTED_ENDPOINT_ROLE_CATALOG[
            "optional_signed_numerical_bias_candidates"
        ]
    ) | set(
        EXPECTED_ENDPOINT_ROLE_CATALOG["optional_signed_semantic_bias_candidates"]
    )
    unsupported_role_endpoints = sorted(set(family) - allowed_signed_endpoints)
    if unsupported_role_endpoints:
        errors.append(
            f"endpoint_family contains non-catalog signed endpoints: {unsupported_role_endpoints}"
        )
    phase_family = spec.get("phase_conditioned_endpoint_family")
    if not isinstance(phase_family, list) or len(phase_family) != len(set(phase_family)):
        errors.append("phase_conditioned_endpoint_family must be a unique list")
        phase_family = []
    elif not set(phase_family).issubset(set(family)):
        errors.append("phase-conditioned endpoints must be a subset of endpoint_family")
    if U2_ENDPOINT in phase_family:
        errors.append("U2 phase-conditioned confirmation requires a separately frozen phase-direction design")
    confirmatory_comparisons = len(family) + len(phase_family) * len(REQUIRED_PHASES)
    family_alpha = numeric("family_alpha")
    if not 0 < family_alpha < 1:
        errors.append("family_alpha must lie between zero and one")
    multiplicity = spec.get("multiplicity")
    if multiplicity == "BONFERRONI_SIMULTANEOUS_TWO_SIDED":
        interval_alpha = (
            family_alpha / confirmatory_comparisons
            if confirmatory_comparisons
            else math.nan
        )
        joint_claim_allowed = True
    elif multiplicity == "NAMED_ENDPOINTS_NO_JOINT_CLAIM_TWO_SIDED":
        interval_alpha = family_alpha
        joint_claim_allowed = False
    else:
        errors.append("unsupported multiplicity declaration")
        interval_alpha = math.nan
        joint_claim_allowed = False

    if planning_mode == PRECISION_TARGETED:
        variance_confidence = numeric("variance_upper_confidence")
    else:
        variance_confidence_raw = spec.get("variance_upper_confidence")
        if variance_confidence_raw is None:
            variance_confidence = math.nan
        else:
            try:
                variance_confidence = float(variance_confidence_raw)
            except (TypeError, ValueError):
                errors.append("variance_upper_confidence must be finite when provided")
                variance_confidence = math.nan
    minimum = integer("minimum_confirmation_trajectories")
    resource_cap = integer("resource_cap")
    if planning_mode == PRECISION_TARGETED and not 0.5 < variance_confidence < 1.0:
        errors.append("variance_upper_confidence must lie between 0.5 and 1")
    if (
        planning_mode == FIXED_RESOURCE_EXISTENCE
        and math.isfinite(variance_confidence)
        and not 0.5 < variance_confidence < 1.0
    ):
        errors.append("provided variance_upper_confidence must lie between 0.5 and 1")
    if minimum < 8:
        errors.append("minimum_confirmation_trajectories must be at least eight")
    if resource_cap < minimum:
        errors.append("resource_cap must be at least the confirmation minimum")
    fixed_trajectories: int | None = None
    if planning_mode == FIXED_RESOURCE_EXISTENCE:
        fixed_trajectories = integer("fixed_confirmation_trajectories")
        if fixed_trajectories < minimum:
            errors.append(
                "fixed_confirmation_trajectories must meet the confirmation minimum"
            )
        if fixed_trajectories > resource_cap:
            errors.append("fixed_confirmation_trajectories exceeds resource_cap")
        fixed_source = spec.get("fixed_resource_source")
        if (
            not isinstance(fixed_source, dict)
            or fixed_source.get("kind") not in FIXED_RESOURCE_SOURCE_KINDS
            or not all(
                isinstance(fixed_source.get(field), str)
                and bool(fixed_source[field].strip())
                for field in ("description", "selection_rule")
            )
            or fixed_source.get("uses_calibration_candidate_mean_or_sign") is not False
        ):
            errors.append(
                "fixed_resource_source must auditably justify J without calibration candidate mean/sign"
            )
    else:
        fixed_source = None
    sensitivity_required: int | None = None
    if (
        math.isfinite(interval_alpha)
        and 0 < interval_alpha < 1
        and minimum >= 2
        and resource_cap >= minimum
    ):
        sensitivity_required = signflip_resolution_requirement(
            interval_alpha,
            minimum,
            resource_cap,
        )
        if sensitivity_required is None:
            errors.append("sign-flip sensitivity p-value resolution exceeds resource cap")
        elif (
            planning_mode == FIXED_RESOURCE_EXISTENCE
            and fixed_trajectories is not None
            and fixed_trajectories < sensitivity_required
        ):
            errors.append(
                "fixed_confirmation_trajectories cannot attain the adjusted sign-flip p-value resolution"
            )
    endpoint_specs = spec.get("endpoints", {})
    if not isinstance(endpoint_specs, dict) or set(endpoint_specs) != set(family):
        errors.append("endpoints spec keys must exactly match endpoint_family")
        endpoint_specs = {}
    try:
        tail_required, tail_result = tail_trajectory_requirement(spec.get("tail", {}))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        tail_required, tail_result = 0, {"scope": "INVALID"}
    if (
        planning_mode == FIXED_RESOURCE_EXISTENCE
        and fixed_trajectories is not None
        and fixed_trajectories < tail_required
    ):
        errors.append("fixed_confirmation_trajectories does not meet tail coverage")

    endpoint_results: dict[str, Any] = {}
    endpoint_required_counts: list[int] = []
    if not errors:
        for name in family:
            if planning_mode == PRECISION_TARGETED:
                threshold_sources, source_errors = validate_threshold_sources(
                    endpoint_specs[name], name
                )
            else:
                threshold_sources, source_errors = validate_existence_threshold_source(
                    endpoint_specs[name], name
                )
            if source_errors:
                errors.extend(source_errors)
                continue
            practical_tolerance_raw = endpoint_specs[name].get(
                "practical_tolerance"
            )
            practical_tolerance_source = endpoint_specs[name].get(
                "practical_tolerance_source"
            )
            practical_tolerance: float | None = None
            if practical_tolerance_raw is not None:
                try:
                    practical_tolerance = float(practical_tolerance_raw)
                except (TypeError, ValueError):
                    errors.append(f"{name}: practical_tolerance must be finite")
                    continue
                if not math.isfinite(practical_tolerance) or practical_tolerance <= 0:
                    errors.append(f"{name}: practical_tolerance must be positive")
                    continue
                if (
                    not isinstance(practical_tolerance_source, dict)
                    or practical_tolerance_source.get("kind")
                    != "EXTERNAL_SCIENTIFIC_TOLERANCE"
                    or not all(
                        isinstance(practical_tolerance_source.get(field), str)
                        and bool(practical_tolerance_source[field].strip())
                        for field in ("description", "selection_rule")
                    )
                    or practical_tolerance_source.get(
                        "uses_calibration_candidate_mean_or_sign"
                    )
                    is not False
                ):
                    errors.append(
                        f"{name}: practical_tolerance requires an independent external scientific tolerance source"
                    )
                    continue
            elif practical_tolerance_source is not None:
                errors.append(
                    f"{name}: practical_tolerance_source requires practical_tolerance"
                )
                continue
            try:
                shift_existence_floor = float(
                    endpoint_specs[name]["shift_existence_floor"]
                )
            except (KeyError, TypeError, ValueError):
                errors.append(f"{name}: shift-existence floor must be finite")
                continue
            if not math.isfinite(shift_existence_floor) or shift_existence_floor < 0:
                errors.append(f"{name}: invalid shift-existence floor")
                continue
            desired_half_width: float | None = None
            variance_floor_sd: float | None = None
            if planning_mode == PRECISION_TARGETED:
                try:
                    desired_half_width = float(
                        endpoint_specs[name]["desired_half_width"]
                    )
                    variance_floor_sd = float(endpoint_specs[name]["variance_floor_sd"])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"{name}: half-width and variance floor must be finite numbers in precision-targeted mode"
                    )
                    continue
                if (
                    not math.isfinite(desired_half_width)
                    or not math.isfinite(variance_floor_sd)
                    or desired_half_width <= 0
                    or variance_floor_sd < 0
                ):
                    errors.append(
                        f"{name}: invalid half-width or variance floor in precision-targeted mode"
                    )
                    continue
            if (
                practical_tolerance is not None
                and practical_tolerance < shift_existence_floor
            ):
                errors.append(
                    f"{name}: practical_tolerance cannot be below shift_existence_floor"
                )
                continue
            direction_link: dict[str, Any] | None = None
            if name == U2_ENDPOINT:
                direction_link = spec.get("U2_directional_replication", {})
                path_value = (
                    direction_link.get("direction_manifest_path")
                    if isinstance(direction_link, dict)
                    else None
                )
                if not isinstance(path_value, str):
                    errors.append(f"{name}: frozen direction manifest is missing")
                    continue
                path = Path(path_value).resolve()
                if not path.is_file() or sha256_file(path) != direction_link.get(
                    "direction_manifest_sha256"
                ):
                    errors.append(f"{name}: frozen direction identity failed")
                    continue
                direction = json.loads(path.read_text(encoding="utf-8"))
                if (
                    direction.get("schema_version") != U2_DIRECTION_VERSION
                    or direction.get("valid") is not True
                    or direction.get("verdict")
                    != "VALID_FROZEN_U2_CALIBRATION_DIRECTION"
                    or direction.get("status") != "FROZEN_BEFORE_CONFIRMATION"
                ):
                    errors.append(f"{name}: linked direction is not valid/frozen")
                    continue
                contract = direction.get("precision_contract", {})
                expected_contract = {"projection_shift_existence_floor": shift_existence_floor}
                if planning_mode == PRECISION_TARGETED:
                    expected_contract.update(
                        {
                            "desired_projection_half_width": desired_half_width,
                            "projection_variance_floor_sd": variance_floor_sd,
                        }
                    )
                if any(
                    not math.isclose(
                        float(contract.get(key, math.nan)),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    for key, expected in expected_contract.items()
                ):
                    errors.append(f"{name}: direction precision contract drifted")
                    continue
                effects = [
                    float(value)
                    for value in direction.get("stability", {}).get(
                        "crossfit_projections", []
                    )
                ]
                endpoint_class = "SIGNED_CALIBRATION_DIRECTION_UPDATE_ENDPOINT"
            else:
                calibration_endpoint = calibration.get("endpoints", {}).get(name, {})
                if calibration_endpoint.get("status") != "COMPLETE_FOUR_TRAJECTORY_CALIBRATION_DESCRIPTION":
                    errors.append(f"{name}: calibration endpoint is not complete")
                    continue
                endpoint_class = calibration_endpoint.get("endpoint_class")
                if endpoint_class not in SIGNED_B_ENDPOINT_CLASSES:
                    errors.append(
                        f"{name}: endpoint class {endpoint_class!r} cannot size a signed-bias confirmation"
                    )
                    continue
                trajectory_rows = calibration_endpoint.get("trajectory_rows", [])
                effects = [float(row["mean_effect"]) for row in trajectory_rows]
            if len(effects) != 4 or any(not math.isfinite(value) for value in effects):
                errors.append(f"{name}: expected four finite trajectory planning effects")
                continue
            observed_variance = sample_variance(effects)
            upper: float | None = None
            variance_plan: float | None = None
            if planning_mode == PRECISION_TARGETED:
                assert desired_half_width is not None
                assert variance_floor_sd is not None
                try:
                    upper = variance_upper_bound(
                        observed_variance, len(effects), variance_confidence
                    )
                except ValueError as error:
                    errors.append(f"{name}: {error}")
                    continue
                variance_plan = max(upper, variance_floor_sd**2)
                if variance_plan == 0:
                    endpoint_results[name] = {
                        "status": "UNINSTANTIATED_ZERO_SCALE",
                        "reason": "zero pilot variance without an independent positive variance floor",
                    }
                    errors.append(f"{name}: zero planning scale")
                    continue
                required = required_trajectories_for_half_width(
                    variance_plan,
                    desired_half_width,
                    interval_alpha,
                    minimum,
                    resource_cap,
                )
                if required is None:
                    endpoint_results[name] = {
                        "status": "INFEASIBLE_AT_DECLARED_PRECISION",
                        "desired_half_width": desired_half_width,
                        "variance_plan": variance_plan,
                        "resource_cap": resource_cap,
                    }
                    errors.append(f"{name}: precision target exceeds resource cap")
                    continue
            else:
                assert fixed_trajectories is not None
                required = fixed_trajectories
            endpoint_required_counts.append(required)
            phase_conditioned_plans: dict[str, Any] = {}
            if name in phase_family:
                calibration_endpoint = calibration.get("endpoints", {}).get(name, {})
                calibration_phase_rows = calibration_endpoint.get("phase_rows", [])
                for phase in REQUIRED_PHASES:
                    phase_effects = [
                        float(row["mean_effect"])
                        for row in calibration_phase_rows
                        if row.get("phase") == phase
                    ]
                    if len(phase_effects) != 4 or any(
                        not math.isfinite(value) for value in phase_effects
                    ):
                        errors.append(
                            f"{name}/{phase}: expected four finite phase trajectory effects"
                        )
                        continue
                    phase_observed_variance = sample_variance(phase_effects)
                    phase_upper: float | None = None
                    phase_variance_plan: float | None = None
                    if planning_mode == PRECISION_TARGETED:
                        assert desired_half_width is not None
                        assert variance_floor_sd is not None
                        try:
                            phase_upper = variance_upper_bound(
                                phase_observed_variance,
                                len(phase_effects),
                                variance_confidence,
                            )
                        except ValueError as error:
                            errors.append(f"{name}/{phase}: {error}")
                            continue
                        phase_variance_plan = max(
                            phase_upper, variance_floor_sd**2
                        )
                        phase_required = required_trajectories_for_half_width(
                            phase_variance_plan,
                            desired_half_width,
                            interval_alpha,
                            minimum,
                            resource_cap,
                        )
                        if phase_required is None:
                            errors.append(
                                f"{name}/{phase}: precision target exceeds resource cap"
                            )
                            continue
                    else:
                        phase_required = required
                    endpoint_required_counts.append(phase_required)
                    phase_conditioned_plans[phase] = {
                        "status": (
                            "PLANNED"
                            if planning_mode == PRECISION_TARGETED
                            else "PLANNED_FIXED_RESOURCE_EXISTENCE"
                        ),
                        "planning_mode": planning_mode,
                        "desired_half_width": desired_half_width,
                        "calibration_sample_variance": phase_observed_variance,
                        "variance_upper_bound": phase_upper,
                        "variance_plan": phase_variance_plan,
                        "required_confirmation_trajectories": phase_required,
                    }
            endpoint_results[name] = {
                "status": (
                    "PLANNED"
                    if planning_mode == PRECISION_TARGETED
                    else "PLANNED_FIXED_RESOURCE_EXISTENCE"
                ),
                "planning_mode": planning_mode,
                "desired_half_width": desired_half_width,
                "calibration_sample_variance": observed_variance,
                "variance_upper_bound": upper,
                "variance_floor_sd": variance_floor_sd,
                "shift_existence_floor": shift_existence_floor,
                "threshold_sources": threshold_sources,
                "practical_tolerance": practical_tolerance,
                "practical_tolerance_source": practical_tolerance_source,
                "variance_plan": variance_plan,
                "interval_alpha": interval_alpha,
                "required_confirmation_trajectories": required,
                "calibration_mean_or_sign_used_for_sizing": False,
                "endpoint_class": endpoint_class,
                "phase_conditioned_plans": phase_conditioned_plans,
            }
            if direction_link is not None:
                endpoint_results[name]["direction"] = {
                    "path": str(Path(direction_link["direction_manifest_path"]).resolve()),
                    "sha256": direction_link["direction_manifest_sha256"],
                    "planning_dispersion": "leave-one-trajectory-out cross-fitted projections",
                }

    if planning_mode == FIXED_RESOURCE_EXISTENCE:
        planned = (
            fixed_trajectories
            if endpoint_required_counts and not errors
            else None
        )
    else:
        planned = (
            max(
                [
                    minimum,
                    tail_required,
                    sensitivity_required or 0,
                    *endpoint_required_counts,
                ]
            )
            if endpoint_required_counts and not errors
            else None
        )
    if planned is not None and planned > resource_cap:
        errors.append("tail/precision trajectory requirement exceeds resource cap")
        planned = None
    valid = not errors and planned is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "planning_mode": planning_mode,
        "plan_role": (
            "FIXED_RESOURCE_SHIFT_EXISTENCE_CONFIRMATION"
            if planning_mode == FIXED_RESOURCE_EXISTENCE
            else "PRECISION_TARGETED_SHIFT_CONFIRMATION"
        ),
        "valid": valid,
        "verdict": "VALID_FROZEN_PRECISION_PLAN" if valid else "UNINSTANTIATED_OR_INFEASIBLE",
        "errors": errors,
        "multiplicity": {
            "method": multiplicity,
            "family_alpha": family_alpha if math.isfinite(family_alpha) else None,
            "per_interval_alpha": interval_alpha if math.isfinite(interval_alpha) else None,
            "joint_claim_allowed": joint_claim_allowed,
            "endpoint_family": family,
            "phase_conditioned_endpoint_family": phase_family,
            "confirmatory_comparisons": confirmatory_comparisons,
        },
        "variance_upper_confidence": (
            variance_confidence if math.isfinite(variance_confidence) else None
        ),
        "minimum_confirmation_trajectories": minimum,
        "resource_cap": resource_cap,
        "fixed_confirmation_trajectories": fixed_trajectories,
        "fixed_resource_source": fixed_source,
        "tail": tail_result,
        "sensitivity": {
            "method": "TRAJECTORY_RADEMACHER_SIGN_FLIP_STUDENTIZED",
            "role": "VETO_PRIMARY_SHIFT_ONLY",
            "exact_max_trajectories": 16,
            "monte_carlo_draws": 99999,
            "monte_carlo_seed": 172904,
            "minimum_trajectories_for_p_value_resolution": sensitivity_required,
            "minimum_attainable_p_at_planned_count": (
                minimum_attainable_signflip_p(planned) if planned is not None else None
            ),
        },
        "endpoints": endpoint_results,
        "planned_confirmation_trajectories": planned,
        "calibration_trajectories_reused_in_confirmation_df": False,
        "nonclaims": [
            (
                "fixed-resource existence mode does not promise target power or half-width"
                if planning_mode == FIXED_RESOURCE_EXISTENCE
                else "precision planning does not establish power for a particular nonzero effect"
            ),
            "regularity-conditional variance planning does not prove heavy-tail or rare-regime coverage",
            "a valid sample-size plan does not establish bias, impact, materiality or correctness",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    calibration_path = Path(args.calibration).resolve()
    spec_path = Path(args.spec).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = plan_confirmation(calibration, spec)
    result["inputs"] = {
        "calibration": {"path": str(calibration_path), "sha256": sha256_file(calibration_path)},
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "errors": result["errors"], "planned_confirmation_trajectories": result["planned_confirmation_trajectories"]}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
