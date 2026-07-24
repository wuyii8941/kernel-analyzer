#!/usr/bin/env python3
"""Evaluate a frozen boundary-conditional family on independent Qwen3 states."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    REQUIRED_PHASES,
    WEIGHTING_CONTRACT_ID,
    load_state,
    sha256_file,
)
from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    load_complete_state_bundles,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    FIXED_RESOURCE_EXISTENCE,
    minimum_attainable_signflip_p,
)
from theory_oracle.bias_oracle_population_v0_2 import (
    EffectRecord,
    estimate_scalar_population,
)
from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (
    EXPECTED_SENSITIVITY,
    apply_trajectory_sensitivity,
    classify_oracle_disposition,
    evaluate_operator_attribution_eligibility,
    evaluate_realized_precision,
    resolve_from,
    validate_confirmation_manifest,
    validate_confirmation_source_audit,
)
from theory_oracle.freeze_qwen3_boundary_condition_family_v0_1 import (
    SCHEMA_VERSION as FAMILY_SCHEMA_VERSION,
)
from theory_oracle.plan_qwen3_boundary_confirmation_resource_v0_1 import (
    SCHEMA_VERSION as RESOURCE_PLAN_SCHEMA,
    SCRIPT_PATH as RESOURCE_PLANNER_PATH,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-conditional-confirmation.v0.1"
SPEC_VERSION = "forkcert.qwen3-boundary-conditional-confirmation-spec.v0.1"
FAMILY_STATUS = "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY"
SIGNED_EFFECT_KINDS = {
    "boundary_margin_shift": "mean_margin_shift",
    "boundary_clip_directional_shift": "directional_event_shift",
}
SEMANTIC_IMPACT_KINDS = {
    "boundary_semantic_disagreement": "semantic_disagreement",
}
SUPPORTED_KINDS = {**SIGNED_EFFECT_KINDS, **SEMANTIC_IMPACT_KINDS}
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ANALYSIS = {
    "boundary_confirmation_evaluator": SCRIPT_PATH,
    "boundary_measurement": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_boundary_conditioned_calibration_v0_1.py",
    "population_estimator": ROOT / "theory_oracle" / "bias_oracle_population_v0_2.py",
    "trajectory_sensitivity": ROOT
    / "theory_oracle"
    / "bias_oracle_trajectory_signflip_v0_1.py",
}


def parse_endpoint(endpoint: str) -> tuple[str, float]:
    try:
        kind, tau_text = endpoint.split("::tau=", 1)
        tau = float(tau_text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid boundary endpoint identity: {endpoint!r}") from None
    if kind not in SUPPORTED_KINDS or not math.isfinite(tau) or tau <= 0:
        raise ValueError(f"unsupported boundary endpoint identity: {endpoint!r}")
    return kind, tau


def collect_boundary_effect_records(
    state_rows: list[dict[str, Any]], endpoint: str
) -> tuple[list[EffectRecord], dict[str, Any], list[str]]:
    kind, tau = parse_endpoint(endpoint)
    key = str(tau)
    metric = SUPPORTED_KINDS[kind]
    records: list[EffectRecord] = []
    support: Counter[tuple[str, str]] = Counter()
    exposure_counts: Counter[tuple[str, str]] = Counter()
    exposed_state_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    condition_masks: dict[str, str] = {}
    errors: list[str] = []
    for state in state_rows:
        anchor_stability = state.get("reference_anchor_stability", {})
        if (
            anchor_stability.get("reference_scorer_logps_exact_across_repeats")
            is not True
            or anchor_stability.get("formal_confirmation_allowed") is not True
        ):
            errors.append(
                f"{state.get('state_id')}: stochastic reference anchor is not "
                "identified by boundary confirmation v0.1"
            )
            continue
        repeats = state.get("repeat_profiles", [])
        current_values = []
        current_exposures = []
        current_mask_hashes = []
        for repeat in repeats:
            profile = repeat.get("tau_profiles", {}).get(key, {})
            value = profile.get(metric)
            exposures = profile.get("exposures")
            if value is None or not isinstance(exposures, int) or exposures <= 0:
                continue
            current_values.append((int(repeat["repeat_id"]), float(value)))
            current_exposures.append(exposures)
            current_mask_hashes.append(str(profile.get("condition_mask_sha256")))
        if not current_values:
            continue
        if len(current_values) != 2 or {item[0] for item in current_values} != {1, 2}:
            errors.append(f"{state.get('state_id')}: incomplete exposed repeat grid")
            continue
        if len(set(current_exposures)) != 1:
            errors.append(f"{state.get('state_id')}: anchor exposure changed across repeats")
            continue
        if len(set(current_mask_hashes)) != 1 or current_mask_hashes[0] in {"", "None"}:
            errors.append(f"{state.get('state_id')}: anchor condition-mask identity drifted")
            continue
        trajectory = str(state["trajectory_id"])
        phase = str(state["phase"])
        support[(trajectory, phase)] += 1
        exposure_counts[(trajectory, phase)] += current_exposures[0]
        exposed_state_ids[(trajectory, phase)].append(str(state["state_id"]))
        condition_masks[str(state["state_id"])] = current_mask_hashes[0]
        for repeat_id, value in current_values:
            if kind in SEMANTIC_IMPACT_KINDS and not 0.0 <= value <= 1.0:
                errors.append(
                    f"{state.get('state_id')}: semantic disagreement must lie in [0, 1]"
                )
                continue
            records.append(
                EffectRecord(
                    trajectory_id=trajectory,
                    phase=phase,
                    state_id=str(state["state_id"]),
                    repeat_id=repeat_id,
                    effect=value,
                )
            )
    trajectories = sorted({str(row.get("trajectory_id")) for row in state_rows})
    for trajectory in trajectories:
        for phase in REQUIRED_PHASES:
            if support[(trajectory, phase)] < 2:
                errors.append(
                    f"{trajectory}/{phase}: boundary endpoint requires at least two exposed states"
                )
    support_payload = {
        "tau": tau,
        "endpoint_kind": kind,
        "states_with_exposure_by_trajectory_phase": {
            f"{trajectory}::{phase}": support[(trajectory, phase)]
            for trajectory in trajectories
            for phase in REQUIRED_PHASES
        },
        "token_exposures_descriptive_only_by_trajectory_phase": {
            f"{trajectory}::{phase}": exposure_counts[(trajectory, phase)]
            for trajectory in trajectories
            for phase in REQUIRED_PHASES
        },
        "exposed_state_ids_by_trajectory_phase": {
            f"{trajectory}::{phase}": sorted(
                exposed_state_ids[(trajectory, phase)]
            )
            for trajectory in trajectories
            for phase in REQUIRED_PHASES
        },
        "reference_anchor_condition_mask_sha256_by_state": dict(
            sorted(condition_masks.items())
        ),
        "condition_anchor": (
            "REFERENCE_REPEAT_1_MARGIN_MASK_WITH_EXACT_REFERENCE_REPEATS;_"
            "STOCHASTIC_REFERENCE_REQUIRES_INDEPENDENT_ANCHOR_PROTOCOL"
        ),
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
    }
    return ([] if errors else records), support_payload, errors


def evaluate_endpoint_records(
    records: list[EffectRecord],
    *,
    endpoint_plan: dict[str, Any],
    interval_alpha: float,
    planned_trajectories: int,
    endpoint: str | None = None,
) -> dict[str, Any]:
    kind = parse_endpoint(endpoint)[0] if endpoint is not None else None
    if kind in SEMANTIC_IMPACT_KINDS:
        return evaluate_semantic_impact_records(
            records,
            endpoint_plan=endpoint_plan,
            interval_alpha=interval_alpha,
            planned_trajectories=planned_trajectories,
        )
    estimate = estimate_scalar_population(
        records,
        required_phases=REQUIRED_PHASES,
        min_confirmation_trajectories=planned_trajectories,
        measurement_floor=float(endpoint_plan["shift_existence_floor"]),
        interval_alpha=interval_alpha,
    )
    sensitivity, final = apply_trajectory_sensitivity(
        estimate,
        endpoint_plan,
        interval_alpha,
        EXPECTED_SENSITIVITY,
    )
    precision = evaluate_realized_precision(estimate, endpoint_plan)
    eligibility = evaluate_operator_attribution_eligibility(
        final, precision["verdict"]
    )
    eligibility["claim_scope"] = (
        "intervention-dependent contribution to this exact independently "
        "confirmed reference-anchored boundary-conditional estimand only; "
        "not contribution to global B"
    )
    return {
        "estimate": estimate,
        "sensitivity": sensitivity,
        "realized_precision": precision,
        "final_shift_verdict": final,
        "oracle_disposition": classify_oracle_disposition(
            final, precision["verdict"]
        ),
        "operator_attribution_eligibility": eligibility,
        "decision_axes": {
            "conditional_shift_existence": final,
            "realized_precision": precision["verdict"],
            "practical_materiality": "UNINSTANTIATED_MATERIALITY",
            "correctness": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
            "long_run_training_impact": "UNINSTANTIATED_ONE_STEP_ORACLE_ONLY",
        },
    }


def evaluate_semantic_impact_records(
    records: list[EffectRecord],
    *,
    endpoint_plan: dict[str, Any],
    interval_alpha: float,
    planned_trajectories: int,
) -> dict[str, Any]:
    """Confirm nonnegative disagreement without renaming it Bias.

    The same trajectory-aware mean/interval machinery is reused, but every
    externally visible field has semantic-impact names.  A positive lower
    confidence bound above the frozen floor is required; there is no signed
    direction claim and no global-B claim.
    """
    if any(not 0.0 <= float(record.effect) <= 1.0 for record in records):
        raise ValueError("semantic disagreement records must lie in [0, 1]")
    floor = endpoint_plan.get("semantic_impact_existence_floor")
    if not isinstance(floor, (int, float)):
        raise ValueError("semantic impact endpoint lacks an existence floor")
    normalized_plan = {
        **endpoint_plan,
        "shift_existence_floor": float(floor),
    }
    estimate = estimate_scalar_population(
        records,
        required_phases=REQUIRED_PHASES,
        min_confirmation_trajectories=planned_trajectories,
        measurement_floor=float(floor),
        interval_alpha=interval_alpha,
    )
    lower, upper = estimate["B"]["trajectory_t_interval"]
    primary = estimate["verdicts"]["shift_existence"]
    if (
        primary == "REPRODUCIBLE_AVERAGE_SHIFT"
        and (lower is None or float(lower) <= float(floor))
    ):
        raise ValueError(
            "nonnegative semantic impact cannot be confirmed through a negative-side interval"
        )
    sensitivity, final_base = apply_trajectory_sensitivity(
        estimate,
        normalized_plan,
        interval_alpha,
        EXPECTED_SENSITIVITY,
    )
    precision = evaluate_realized_precision(estimate, normalized_plan)
    verdict_names = {
        "REPRODUCIBLE_AVERAGE_SHIFT": "REPRODUCIBLE_SEMANTIC_DISAGREEMENT",
        "NO_STABLE_AVERAGE_DETECTED": "NO_REPRODUCIBLE_SEMANTIC_DISAGREEMENT_DETECTED",
        "INDETERMINATE_TOO_FEW_TRAJECTORIES": "INDETERMINATE_TOO_FEW_TRAJECTORIES",
        "INDETERMINATE_METHOD_SENSITIVITY": "INDETERMINATE_SEMANTIC_IMPACT_METHOD_SENSITIVITY",
    }
    final = verdict_names.get(final_base, final_base)
    blockers: list[str] = []
    if final_base != "REPRODUCIBLE_AVERAGE_SHIFT":
        blockers.append(final)
    if precision["verdict"] not in {
        "ADEQUATE_REALIZED_PRECISION",
        "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
    }:
        blockers.append(precision["verdict"])
    population_mean = dict(estimate["B"])
    population_mean["semantic_impact_existence_floor"] = population_mean.pop(
        "measurement_floor"
    )
    semantic_profile = {
        "schema_version": "forkcert.semantic-impact-population-view.v0.1",
        "construction": estimate["construction"],
        "population_mean_disagreement": population_mean,
        "predeclared_phase_mean_disagreement": estimate["conditional_B"][
            "predeclared_phase_rows"
        ],
        "state_and_phase_heterogeneity": estimate["H"],
        "same_state_runtime_variability": estimate["N"],
        "trajectory_sampling_uncertainty": estimate["U"],
        "trajectory_rows": estimate["trajectory_rows"],
        "phase_rows": estimate["phase_rows"],
        "state_rows": estimate["state_rows"],
        "identification_assumptions": estimate["identification_assumptions"],
        "semantic_impact_is_not_B": True,
    }
    return {
        "endpoint_role": "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B",
        "semantic_impact_estimate": semantic_profile,
        "sensitivity": sensitivity,
        "realized_precision": precision,
        "final_semantic_impact_verdict": final,
        "oracle_disposition": (
            "CONFIRMED_SEMANTIC_DISAGREEMENT"
            if final_base == "REPRODUCIBLE_AVERAGE_SHIFT" and not blockers
            else "SEMANTIC_DISAGREEMENT_NOT_CONFIRMED"
            if final_base == "NO_STABLE_AVERAGE_DETECTED"
            else "INDETERMINATE_SEMANTIC_IMPACT"
        ),
        "operator_attribution_eligibility": {
            "eligible": not blockers,
            "blockers": blockers,
            "claim_scope": (
                "intervention-dependent contribution to this exact independently "
                "confirmed boundary-conditioned semantic-disagreement estimand; "
                "not contribution to B"
            ),
        },
        "decision_axes": {
            "semantic_disagreement_existence": final,
            "realized_precision": precision["verdict"],
            "practical_materiality": "UNINSTANTIATED_MATERIALITY",
            "correctness": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
            "long_run_training_impact": "UNINSTANTIATED_ONE_STEP_ORACLE_ONLY",
        },
    }


def validate_boundary_spec(
    spec: dict[str, Any],
    family: dict[str, Any],
    family_path: Path,
    confirmation_manifest_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if spec.get("schema_version") != SPEC_VERSION:
        errors.append("unsupported boundary confirmation spec schema")
    if spec.get("status") != "FROZEN_BEFORE_CONFIRMATION":
        errors.append("boundary confirmation spec is not frozen")
    if (
        family.get("schema_version") != FAMILY_SCHEMA_VERSION
        or family.get("valid") is not True
        or family.get("status") != FAMILY_STATUS
    ):
        errors.append("boundary family manifest is not frozen/support-complete")
    if family.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("boundary family weighting contract drifted")
    link = spec.get("boundary_family", {})
    if (
        Path(link.get("path", "")).resolve() != family_path
        or link.get("sha256") != sha256_file(family_path)
    ):
        errors.append("boundary family identity drifted")
    endpoint_family = spec.get("endpoint_family")
    if endpoint_family != family.get("endpoint_family") or not endpoint_family:
        errors.append("spec endpoint family must exactly match frozen boundary family")
        endpoint_family = []
    if spec.get("planning_mode") != FIXED_RESOURCE_EXISTENCE:
        errors.append("boundary v0.1 supports fixed-resource existence mode only")
    anchor_protocol = spec.get("reference_anchor_protocol", {})
    if (
        anchor_protocol.get("mode")
        != "DETERMINISTIC_REFERENCE_WITH_REPEAT_EXACTNESS_GATE"
        or anchor_protocol.get("reference_conditional_output_is_point_mass_assumption")
        is not True
        or anchor_protocol.get("two_repeat_exactness_is_diagnostic_not_proof")
        is not True
        or anchor_protocol.get("fail_if_reference_repeats_differ") is not True
        or anchor_protocol.get("fallback_requires_independent_anchor_execution")
        is not True
    ):
        errors.append("boundary v0.1 reference-anchor identification protocol is not frozen")
    manifest_link = spec.get("confirmation_manifest", {})
    if (
        Path(manifest_link.get("path", "")).resolve()
        != confirmation_manifest_path
        or manifest_link.get("sha256") != sha256_file(confirmation_manifest_path)
    ):
        errors.append("boundary spec is not bound to the frozen confirmation manifest")
    resource_source = spec.get("fixed_resource_source", {})
    if (
        resource_source.get("kind") != "INHERITED_FROZEN_CONFIRMATION_BANK"
        or not all(
            isinstance(resource_source.get(field), str)
            and bool(resource_source[field].strip())
            for field in ("description", "selection_rule")
        )
        or resource_source.get("uses_boundary_confirmation_outcomes") is not False
    ):
        errors.append("fixed resource source must be the outcome-independent frozen confirmation bank")
    resource_link = spec.get("boundary_resource_plan", {})
    resource_path = Path(resource_link.get("path", "")).resolve()
    resource_plan: dict[str, Any] = {}
    if (
        not resource_path.is_file()
        or resource_link.get("sha256") != (
            sha256_file(resource_path) if resource_path.is_file() else None
        )
    ):
        errors.append("boundary resource-plan identity failed")
    else:
        resource_plan = json.loads(resource_path.read_text(encoding="utf-8"))
        if (
            resource_plan.get("schema_version") != RESOURCE_PLAN_SCHEMA
            or resource_plan.get("valid") is not True
            or resource_plan.get("status")
            != "VALID_FROZEN_BOUNDARY_RESOURCE_REQUIREMENT"
        ):
            errors.append("boundary resource plan is not valid/frozen")
        resource_family = resource_plan.get("boundary_family", {})
        if (
            Path(resource_family.get("path", "")).resolve() != family_path
            or resource_family.get("sha256") != sha256_file(family_path)
        ):
            errors.append("boundary resource plan uses a different family")
        if (
            resource_plan.get("multiplicity")
            != "BONFERRONI_SIMULTANEOUS_TWO_SIDED"
            or resource_plan.get("confirmatory_comparisons")
            != len(family.get("endpoint_family", []))
            or resource_plan.get("candidate_effect_mean_sign_or_variance_used")
            is not False
        ):
            errors.append("boundary resource plan family/multiplicity contract drifted")
        resource_analysis = resource_plan.get("analysis_code", {})
        if (
            Path(resource_analysis.get("path", "")).resolve()
            != RESOURCE_PLANNER_PATH
            or resource_analysis.get("sha256")
            != sha256_file(RESOURCE_PLANNER_PATH)
        ):
            errors.append("boundary resource planner provenance drifted")
    analysis = spec.get("analysis_code", {})
    for name, path in EXPECTED_ANALYSIS.items():
        link = analysis.get(name, {})
        if (
            Path(link.get("path", "")).resolve() != path
            or not path.is_file()
            or link.get("sha256") != sha256_file(path)
        ):
            errors.append(f"boundary confirmation analysis provenance failed: {name}")
    if spec.get("multiplicity") != "BONFERRONI_SIMULTANEOUS_TWO_SIDED":
        errors.append("boundary v0.1 requires Bonferroni simultaneous intervals")
    if spec.get("cross_family_joint_claim_allowed") is not False:
        errors.append("boundary v0.1 forbids an unadjusted joint claim with other endpoint families")
    family_alpha = spec.get("family_alpha")
    if not isinstance(family_alpha, (int, float)) or not 0 < float(family_alpha) < 1:
        errors.append("family_alpha must lie between zero and one")
        family_alpha = math.nan
    endpoint_plans = spec.get("endpoints", {})
    if set(endpoint_plans) != set(endpoint_family):
        errors.append("boundary endpoint plans must exactly match endpoint family")
    common = {
        "description",
        "selection_rule",
        "uses_calibration_candidate_mean_or_sign",
    }
    for endpoint in endpoint_family:
        try:
            kind, _ = parse_endpoint(endpoint)
        except ValueError as error:
            errors.append(str(error))
            continue
        plan = endpoint_plans.get(endpoint, {})
        semantic_impact = kind in SEMANTIC_IMPACT_KINDS
        floor_key = (
            "semantic_impact_existence_floor"
            if semantic_impact
            else "shift_existence_floor"
        )
        source_key = f"{floor_key}_source"
        floor = plan.get(floor_key)
        if not isinstance(floor, (int, float)) or not math.isfinite(float(floor)) or floor < 0:
            errors.append(f"{endpoint}: invalid {floor_key}")
        source = plan.get(source_key, {})
        if (
            source.get("kind") not in {"EXACT_ZERO_NULL", "NEGATIVE_CONTROL_ENVELOPE"}
            or not common.issubset(source)
            or source.get("uses_calibration_candidate_mean_or_sign") is not False
        ):
            errors.append(f"{endpoint}: invalid independent {source_key}")
    comparisons = len(endpoint_family)
    interval_alpha = float(family_alpha) / comparisons if comparisons and math.isfinite(float(family_alpha)) else math.nan
    if resource_plan and (
        not math.isclose(
            float(resource_plan.get("family_alpha", math.nan)),
            float(family_alpha),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(resource_plan.get("per_interval_alpha", math.nan)),
            interval_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        errors.append("boundary resource plan alpha differs from confirmation spec")
    return {
        "endpoint_family": endpoint_family,
        "endpoint_plans": endpoint_plans,
        "family_alpha": family_alpha,
        "interval_alpha": interval_alpha,
        "cross_family_joint_claim_allowed": False,
        "reference_anchor_protocol": anchor_protocol,
        "minimum_trajectories_for_signflip_resolution": resource_plan.get(
            "minimum_trajectories_for_signflip_resolution"
        ),
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-manifest", required=True)
    parser.add_argument("--boundary-family", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.confirmation_manifest).resolve()
    family_path = Path(args.boundary_family).resolve()
    spec_path = Path(args.spec).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family = json.loads(family_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    precision, inputs, errors = validate_confirmation_manifest(manifest, manifest_path)
    frozen, spec_errors = validate_boundary_spec(
        spec, family, family_path, manifest_path
    )
    errors.extend(spec_errors)
    planned = int(precision.get("planned_confirmation_trajectories", 0)) if precision else 0
    required_boundary_trajectories = frozen.get(
        "minimum_trajectories_for_signflip_resolution"
    )
    if (
        isinstance(required_boundary_trajectories, int)
        and planned < required_boundary_trajectories
    ):
        errors.append("frozen confirmation bank is smaller than boundary resource requirement")
    if math.isfinite(frozen["interval_alpha"]) and planned > 0:
        attainable = minimum_attainable_signflip_p(planned)
        if attainable > frozen["interval_alpha"]:
            errors.append("confirmation trajectory count cannot attain boundary-family adjusted sign-flip resolution")

    state_rows: list[dict[str, Any]] = []
    state_evidence: list[dict[str, Any]] = []
    taus = list(family.get("retained_taus", []))
    if not errors:
        for row in inputs:
            errors.extend(
                f"{row['trajectory_id']}: {error}"
                for error in validate_confirmation_source_audit(row)
            )
            plan_path = resolve_from(
                manifest_path.parent, str(row["capture_plan_path"])
            )
            results_root = resolve_from(
                manifest_path.parent, str(row["results_root"])
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            bundles, evidence, current_errors = load_complete_state_bundles(
                plan, results_root
            )
            errors.extend(
                f"{row['trajectory_id']}: {error}" for error in current_errors
            )
            for target, bundle in bundles:
                try:
                    state_rows.append(load_state(target, bundle, taus))
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    errors.append(f"{target.get('state_id')}: {error}")
            state_evidence.extend(
                {"trajectory_id": row["trajectory_id"], **item} for item in evidence
            )
    if len(state_rows) != planned * 24:
        errors.append(f"boundary confirmation state census is {len(state_rows)}/{planned * 24}")

    endpoints: dict[str, Any] = {}
    ready: list[str] = []
    blocked: dict[str, list[str]] = {}
    if not errors:
        for endpoint in frozen["endpoint_family"]:
            records, support, endpoint_errors = collect_boundary_effect_records(
                state_rows, endpoint
            )
            if endpoint_errors:
                endpoints[endpoint] = {
                    "status": "UNAVAILABLE_BOUNDARY_SUPPORT",
                    "support": support,
                    "errors": endpoint_errors,
                    "population_conditional_shift_inference_allowed": False,
                }
                blocked[endpoint] = endpoint_errors
                continue
            result = evaluate_endpoint_records(
                records,
                endpoint_plan={
                    "planning_mode": FIXED_RESOURCE_EXISTENCE,
                    **frozen["endpoint_plans"][endpoint],
                },
                interval_alpha=frozen["interval_alpha"],
                planned_trajectories=planned,
                endpoint=endpoint,
            )
            endpoints[endpoint] = {
                "status": (
                    "MEASURED_BOUNDARY_SEMANTIC_IMPACT_CONFIRMATION"
                    if parse_endpoint(endpoint)[0] in SEMANTIC_IMPACT_KINDS
                    else "MEASURED_BOUNDARY_CONDITIONAL_CONFIRMATION"
                ),
                "support": support,
                **result,
                "population_conditional_shift_inference_allowed": True,
                "correctness_claim_allowed": False,
            }
            eligibility = result["operator_attribution_eligibility"]
            if eligibility["eligible"]:
                ready.append(endpoint)
            else:
                blocked[endpoint] = eligibility["blockers"]
    valid = not errors
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION" if valid else "INVALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
        "construction": {
            "errors": errors,
            "trajectories": planned if valid else None,
            "states": len(state_rows),
            "interval_alpha": frozen["interval_alpha"],
            "cross_family_joint_claim_allowed": False,
            "weighting": "equal trajectory; equal phase; equal anchor-exposed state within phase; equal anchor-near-boundary token within state",
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        },
        "confirmation_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "boundary_family": {"path": str(family_path), "sha256": sha256_file(family_path)},
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "endpoints": endpoints,
        "operator_attribution_gate": {
            "ready_endpoints": ready,
            "blocked_endpoints": blocked,
            "automatic_operator_launch": False,
            "claim_scope": (
                "contribution to the exact confirmed reference-anchored "
                "boundary endpoint only; signed conditional effects and "
                "nonnegative semantic disagreement retain distinct claim names"
            ),
        },
        "state_evidence": state_evidence,
        "nonclaims": [
            "boundary conditional shift is not a global average B",
            "semantic disagreement is a nonnegative impact estimand and is not B",
            "reference anchoring does not make eager a correctness authority",
            "cross-family joint claims are not allowed",
            "one-step conditional shift does not establish long-run harm",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "operator_attribution_gate": payload["operator_attribution_gate"]}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
