#!/usr/bin/env python
"""Plan a frozen contributor-confirmation bank without using pilot mean/sign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    minimum_attainable_signflip_p,
    required_trajectories_for_half_width,
    sample_variance,
    signflip_resolution_requirement,
    tail_trajectory_requirement,
    variance_upper_bound,
)


SCHEMA_VERSION = "forkcert.qwen3-bias-contributor-precision-plan.v0.1"
SPEC_VERSION = "forkcert.qwen3-bias-contributor-precision-input-spec.v0.1"
CANDIDATE_VERSION = "forkcert.qwen3-bias-contributor-candidate-freeze.v0.1"
PILOT_VERSION = "forkcert.qwen3-bias-contributor-pilot-summary.v0.1"
CONTRIBUTOR_THRESHOLD_SOURCE_KINDS = {
    "desired_halfwidth": {
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
        "EXTERNAL_SCIENTIFIC_TOLERANCE",
    },
    "directional_contribution_floor": {
        "EXACT_ZERO_NULL",
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
        "EXTERNAL_SCIENTIFIC_ATTRIBUTION_TOLERANCE",
    },
    "variance_floor_sd": {
        "INDEPENDENT_MEASUREMENT_RESOLUTION",
        "NEGATIVE_CONTROL_ENVELOPE",
        "EXTERNAL_CONSERVATIVE_VARIANCE_FLOOR",
    },
}


def validate_contributor_threshold_sources(
    row: dict[str, Any], candidate_id: str
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    sources = row.get("threshold_sources")
    if not isinstance(sources, dict) or set(sources) != set(
        CONTRIBUTOR_THRESHOLD_SOURCE_KINDS
    ):
        return {}, [
            f"{candidate_id}: threshold_sources must separately cover desired width, contribution floor and variance floor"
        ]
    for threshold, allowed_kinds in CONTRIBUTOR_THRESHOLD_SOURCE_KINDS.items():
        source = sources.get(threshold)
        if not isinstance(source, dict):
            errors.append(f"{candidate_id}: {threshold} source is not auditable")
            continue
        if source.get("kind") not in allowed_kinds:
            errors.append(f"{candidate_id}: unsupported {threshold} source kind")
        if not all(
            isinstance(source.get(field), str) and bool(source[field].strip())
            for field in ("description", "selection_rule")
        ):
            errors.append(
                f"{candidate_id}: {threshold} source requires description and selection_rule"
            )
        if source.get("uses_pilot_contribution_mean_or_sign") is not False:
            errors.append(
                f"{candidate_id}: {threshold} source must exclude pilot contribution mean/sign"
            )
    return sources, errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_contributor_precision(
    candidate_freeze: dict[str, Any],
    spec: dict[str, Any],
    pilot_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if candidate_freeze.get("schema_version") != CANDIDATE_VERSION:
        errors.append("unsupported candidate-freeze schema")
    if candidate_freeze.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PILOT":
        errors.append("candidate set is not frozen before contributor pilot")
    candidates = candidate_freeze.get("primary_candidates", [])
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidate set is empty")
        candidate_ids: list[str] = []
    else:
        candidate_ids = [row.get("candidate_id") for row in candidates if isinstance(row, dict)]
        if len(candidate_ids) != len(candidates) or any(not item for item in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            errors.append("candidate IDs must be present and unique")

    if spec.get("schema_version") != SPEC_VERSION:
        errors.append("unsupported contributor precision input schema")
    if spec.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PRECISION_PLANNING":
        errors.append("contributor precision input spec is not frozen")
    endpoint = spec.get("target_endpoint")
    if endpoint != candidate_freeze.get("target_endpoint") or not endpoint:
        errors.append("precision endpoint disagrees with candidate freeze")

    direction = spec.get("frozen_bias_direction", {})
    if not isinstance(direction, dict) or direction.get("source") != "ENDPOINT_CONFIRMATION_ONLY":
        errors.append("Bias direction must come only from endpoint confirmation")
    elif direction.get("kind") == "SCALAR_SIGN":
        if direction.get("scalar_sign") not in (-1, 1):
            errors.append("scalar Bias direction must be -1 or +1")
    elif direction.get("kind") == "VECTOR_DIRECTION_ARTIFACT":
        if not direction.get("vector_direction_artifact_path") or not direction.get("vector_direction_artifact_sha256"):
            errors.append("vector Bias direction artifact is uninstantiated")
    else:
        errors.append("Bias direction kind is uninstantiated")

    multiplicity = spec.get("multiplicity", {})
    members = multiplicity.get("primary_repair_family_members") if isinstance(multiplicity, dict) else None
    if members != candidate_ids:
        errors.append("multiplicity family does not exactly equal frozen candidates")
    if not isinstance(multiplicity, dict) or multiplicity.get("method") != "BONFERRONI_SIMULTANEOUS_TWO_SIDED":
        errors.append("only Bonferroni simultaneous two-sided intervals are validated")
    try:
        family_alpha = float(multiplicity.get("family_alpha"))
    except (TypeError, ValueError):
        family_alpha = math.nan
    if not 0.0 < family_alpha < 1.0 or not candidate_ids:
        errors.append("family alpha is invalid")
        interval_alpha = math.nan
    else:
        interval_alpha = family_alpha / len(candidate_ids)
    if not isinstance(multiplicity, dict) or multiplicity.get("injection_is_separate_family") is not True:
        errors.append("injection must be a separate hypothesis family")

    try:
        variance_confidence = float(spec.get("variance_upper_confidence"))
    except (TypeError, ValueError):
        variance_confidence = math.nan
    if not 0.5 < variance_confidence < 1.0:
        errors.append("variance upper confidence must lie between 0.5 and 1")

    design = spec.get("global_design", {})
    minimum = design.get("minimum_confirmation_trajectories") if isinstance(design, dict) else None
    cap = design.get("resource_cap_trajectories") if isinstance(design, dict) else None
    if not isinstance(minimum, int) or minimum < 8:
        errors.append("minimum contributor confirmation trajectories must be at least 8")
    if not isinstance(cap, int) or not isinstance(minimum, int) or cap < minimum:
        errors.append("contributor resource cap is invalid")
    if not isinstance(design, dict) or design.get("no_optional_stopping_on_mean_or_sign") is not True:
        errors.append("optional stopping on pilot mean/sign must be prohibited")
    try:
        tail_required, tail_result = tail_trajectory_requirement(
            design.get("tail", {}) if isinstance(design, dict) else {}
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        tail_required, tail_result = 0, {"scope": "INVALID"}

    sensitivity = spec.get("sensitivity", {})
    expected_sensitivity = {
        "method": "TRAJECTORY_RADEMACHER_SIGN_FLIP_STUDENTIZED",
        "role": "VETO_PRIMARY_CONTRIBUTION_ONLY",
        "multiplicity_adjusted_alpha_required": True,
        "exact_max_trajectories": 16,
        "monte_carlo_draws": 99999,
        "monte_carlo_seed": 172904,
    }
    if not isinstance(sensitivity, dict) or any(
        sensitivity.get(key) != value for key, value in expected_sensitivity.items()
    ):
        errors.append("contributor sensitivity procedure drifted")

    basis = spec.get("precision_basis", {})
    mode = basis.get("mode") if isinstance(basis, dict) else None
    pilot_rows: list[dict[str, Any]] = []
    pilot_trajectories = 0
    if mode == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY":
        if pilot_summary is None:
            errors.append("pilot precision mode requires a pilot summary")
        else:
            if pilot_summary.get("schema_version") != PILOT_VERSION or pilot_summary.get("valid") is not True:
                errors.append("contributor pilot summary is invalid")
            if pilot_summary.get("target_endpoint") != endpoint:
                errors.append("contributor pilot endpoint mismatch")
            if pilot_summary.get("candidate_ids") != candidate_ids:
                errors.append("contributor pilot candidate IDs/order mismatch")
            construction = pilot_summary.get("construction", {})
            pilot_trajectories = construction.get("trajectories", 0) if isinstance(construction, dict) else 0
            if not isinstance(pilot_trajectories, int) or pilot_trajectories < 4:
                errors.append("contributor pilot requires at least four independent trajectories")
            if not isinstance(construction, dict) or construction.get("role") != "DISPERSION_ONLY_NO_MEAN_OR_SIGN_SELECTION":
                errors.append("contributor pilot role is not dispersion-only")
            raw_rows = pilot_summary.get("trajectory_specs", [])
            pilot_rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
            if len(pilot_rows) != pilot_trajectories:
                errors.append("contributor pilot trajectory identity count mismatch")
    elif mode == "EXTERNAL_FIXED_CONSERVATIVE_VARIANCE":
        if pilot_summary is not None:
            errors.append("external fixed variance mode must not consume a pilot summary")
        if not isinstance(basis, dict) or not basis.get("external_fixed_rationale"):
            errors.append("external fixed variance mode requires a rationale")
    else:
        errors.append("unsupported contributor precision basis")

    inputs = spec.get("candidate_precision_inputs", {})
    if not isinstance(inputs, dict) or set(inputs) != set(candidate_ids):
        errors.append("candidate precision inputs must exactly match frozen candidates")
        inputs = {}
    candidate_results: dict[str, Any] = {}
    candidate_required: list[int] = []
    if not errors:
        for candidate_id in candidate_ids:
            row = inputs[candidate_id]
            threshold_sources, source_errors = (
                validate_contributor_threshold_sources(row, candidate_id)
            )
            if source_errors:
                errors.extend(source_errors)
                continue
            try:
                desired = float(row["desired_halfwidth"])
                direction_floor = float(row["directional_contribution_floor"])
                variance_floor_sd = float(row["variance_floor_sd"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{candidate_id}: invalid candidate precision inputs")
                continue
            if not all(math.isfinite(value) for value in (desired, direction_floor, variance_floor_sd)) or desired <= 0 or direction_floor < 0 or variance_floor_sd < 0:
                errors.append(f"{candidate_id}: invalid half-width/directional/variance floor")
                continue
            observed_variance: float | None
            upper: float | None
            if mode == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY":
                pilot_candidate = pilot_summary.get("candidates", {}).get(candidate_id, {})
                if pilot_candidate.get("status") != "COMPLETE_CONTRIBUTOR_PILOT_DESCRIPTION":
                    errors.append(f"{candidate_id}: pilot contribution profile is incomplete")
                    continue
                trajectory_rows = pilot_candidate.get("trajectory_rows", [])
                effects = [float(item["mean_effect"]) for item in trajectory_rows]
                if len(effects) != pilot_trajectories or any(not math.isfinite(value) for value in effects):
                    errors.append(f"{candidate_id}: pilot trajectory effects are incomplete")
                    continue
                observed_variance = sample_variance(effects)
                upper = variance_upper_bound(
                    observed_variance, pilot_trajectories, variance_confidence
                )
                variance_plan = max(upper, variance_floor_sd**2)
            else:
                observed_variance = None
                upper = None
                try:
                    external_variance = float(row["external_variance_plan"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{candidate_id}: external variance plan is missing")
                    continue
                if not math.isfinite(external_variance) or external_variance <= 0:
                    errors.append(f"{candidate_id}: external variance plan must be positive")
                    continue
                variance_plan = max(external_variance, variance_floor_sd**2)
            if variance_plan <= 0:
                errors.append(f"{candidate_id}: zero planning scale")
                continue
            required = required_trajectories_for_half_width(
                variance_plan,
                desired,
                interval_alpha,
                minimum,
                cap,
            )
            if required is None:
                errors.append(f"{candidate_id}: precision target exceeds resource cap")
                continue
            candidate_required.append(required)
            candidate_results[candidate_id] = {
                "status": "PLANNED",
                "desired_halfwidth": desired,
                "directional_contribution_floor": direction_floor,
                "threshold_sources": threshold_sources,
                "trajectory_variance_floor": variance_floor_sd**2,
                "pilot_sample_variance": observed_variance,
                "trajectory_variance_upper_bound": upper,
                "variance_plan": variance_plan,
                "planned_trajectories": required,
                "pilot_mean_or_sign_used_for_sizing": False,
            }

    sensitivity_required: int | None = None
    if (
        math.isfinite(interval_alpha)
        and isinstance(minimum, int)
        and isinstance(cap, int)
        and minimum >= 2
        and cap >= minimum
        and isinstance(sensitivity, dict)
        and all(key in sensitivity for key in ("exact_max_trajectories", "monte_carlo_draws"))
    ):
        sensitivity_required = signflip_resolution_requirement(
            interval_alpha,
            minimum,
            cap,
            exact_max_trajectories=int(sensitivity["exact_max_trajectories"]),
            monte_carlo_draws=int(sensitivity["monte_carlo_draws"]),
        )
        if sensitivity_required is None:
            errors.append("sign-flip sensitivity resolution exceeds resource cap")

    planned = (
        max([minimum, tail_required, sensitivity_required or 0, *candidate_required])
        if isinstance(minimum, int) and candidate_required and not errors
        else None
    )
    if isinstance(cap, int) and planned is not None and planned > cap:
        errors.append("combined contributor trajectory requirement exceeds resource cap")
        planned = None
    valid = not errors and planned is not None
    pilot_basis = {
        "trajectory_specs": pilot_rows,
        "trajectory_level_repair_effect_dispersion_only": True,
        "mean_and_sign_excluded_from_sizing_and_candidate_selection": True,
        "excluded_from_final_confirmation_df": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": (
            "VALID_FROZEN_CONTRIBUTOR_PRECISION_PLAN"
            if valid
            else "UNINSTANTIATED_OR_INFEASIBLE_CONTRIBUTOR_PRECISION_PLAN"
        ),
        "errors": errors,
        "target_endpoint": endpoint,
        "frozen_bias_direction": direction,
        "candidate_freeze": {"candidate_ids": candidate_ids},
        "precision_basis": {
            "mode": mode,
            "contributor_pilot": pilot_basis,
            "external_fixed_rationale": basis.get("external_fixed_rationale") if isinstance(basis, dict) else None,
        },
        "multiplicity": {
            "primary_repair_family_members": candidate_ids,
            "method": multiplicity.get("method") if isinstance(multiplicity, dict) else None,
            "family_alpha": family_alpha if math.isfinite(family_alpha) else None,
            "per_interval_alpha": interval_alpha if math.isfinite(interval_alpha) else None,
            "injection_is_separate_family": True,
        },
        "variance_upper_confidence": variance_confidence if math.isfinite(variance_confidence) else None,
        "candidate_precision": candidate_results,
        "injection_precision": {
            "status": "UNINSTANTIATED_SEPARATE_PRECISION_PLAN_REQUIRED",
            "repair_alpha_variance_and_trajectory_count_reuse_prohibited": True,
        },
        "global_design": {
            "minimum_confirmation_trajectories": minimum,
            "planned_confirmation_trajectories": planned,
            "resource_cap_trajectories": cap,
            "tail_scope": tail_result.get("scope"),
            "tail": tail_result,
            "no_optional_stopping_on_mean_or_sign": True,
        },
        "sensitivity": {
            **expected_sensitivity,
            "minimum_trajectories_for_p_value_resolution": sensitivity_required,
            "minimum_attainable_p_at_planned_count": (
                minimum_attainable_signflip_p(
                    planned,
                    exact_max_trajectories=expected_sensitivity["exact_max_trajectories"],
                    monte_carlo_draws=expected_sensitivity["monte_carlo_draws"],
                )
                if planned is not None
                else None
            ),
        },
        "nonclaims": [
            "precision planning does not establish a contribution",
            "pilot mean/sign is excluded from sizing and candidate selection",
            "a valid plan does not establish intervention integrity, causality or correctness",
            "this plan covers the primary repair family only; injection requires a separately frozen precision plan",
        ],
    }


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    candidate_link = spec.get("candidate_freeze", {})
    candidate_path = _resolve(spec_path.parent, str(candidate_link.get("artifact_path", "")))
    link_errors = []
    if not candidate_path.is_file() or sha256_file(candidate_path) != candidate_link.get("artifact_sha256"):
        link_errors.append("candidate-freeze artifact missing/hash mismatch")
        candidate = {}
    else:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    pilot = None
    pilot_path = None
    basis = spec.get("precision_basis", {})
    if basis.get("mode") == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY":
        pilot_path = _resolve(spec_path.parent, str(basis.get("contributor_pilot_summary_path", "")))
        if not pilot_path.is_file() or sha256_file(pilot_path) != basis.get("contributor_pilot_summary_sha256"):
            link_errors.append("contributor pilot summary missing/hash mismatch")
        else:
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    result = plan_contributor_precision(candidate, spec, pilot)
    result["errors"] = link_errors + result["errors"]
    if link_errors:
        result["valid"] = False
        result["verdict"] = "UNINSTANTIATED_OR_INFEASIBLE_CONTRIBUTOR_PRECISION_PLAN"
    result["candidate_freeze"].update(
        {"artifact_path": str(candidate_path), "artifact_sha256": sha256_file(candidate_path) if candidate_path.is_file() else None}
    )
    if pilot_path is not None:
        result["precision_basis"]["contributor_pilot"].update(
            {"summary_path": str(pilot_path), "summary_sha256": sha256_file(pilot_path) if pilot_path.is_file() else None}
        )
    result["inputs"] = {
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "errors": result["errors"], "planned_confirmation_trajectories": result["global_design"]["planned_confirmation_trajectories"]}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
