#!/usr/bin/env python
"""Fail-closed design validator for a Qwen3 Bias contributor study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    minimum_attainable_signflip_p,
    signflip_resolution_requirement,
)
from theory_oracle.bias_oracle_contributor_precision_v0_1 import (
    SPEC_VERSION as CONTRIBUTOR_PRECISION_SPEC_VERSION,
    plan_contributor_precision,
)


MANIFEST_SCHEMA = "forkcert.qwen3-bias-contributor-confirmation-manifest.v0.1"
CANDIDATE_SCHEMA = "forkcert.qwen3-bias-contributor-candidate-freeze.v0.1"
UNIVERSE_SCHEMA = "forkcert.qwen3-bias-contributor-universe.v0.1"
COVERAGE_SCHEMA = "forkcert.qwen3-bias-contributor-universe-coverage.v0.1"
INTERVENTION_SCHEMA = "forkcert.qwen3-bias-contributor-intervention.v0.1"
REALIZATION_MAP_SCHEMA = "forkcert.qwen3-bias-contributor-state-realization-map.v0.1"
PRECISION_SCHEMA = "forkcert.qwen3-bias-contributor-precision-plan.v0.1"
CONFIRMATION_SCHEMA = "forkcert.qwen3-bias-oracle-confirmation.v0.1"
PLAN_SCHEMA = "forkcert.multi-transition-capture-plan.v0.1"
ALLOWED_UNITS = {
    "SOURCE_OPERATOR_INVOCATION",
    "GENERATED_KERNEL_INVOCATION",
    "GENERATED_REGION",
    "INTERVENTION_PACKAGE",
}
TRAJECTORY_FIELDS = (
    "trajectory_id",
    "trajectory_seed",
    "data_slice_id",
    "capture_plan_path",
    "capture_plan_sha256",
    "results_root",
)
CANDIDATE_FIELDS = (
    "candidate_id",
    "claim_unit_type",
    "intervention_version",
    "intervention_spec_path",
    "intervention_spec_sha256",
    "selection_source",
)
EXPECTED_ANALYSIS_FILES = {
    "design_validator": Path(__file__).resolve(),
    "precision_planner": ROOT / "theory_oracle" / "bias_oracle_contributor_precision_v0_1.py",
    "profile_estimator": ROOT / "theory_oracle" / "bias_oracle_contributor_v0_1.py",
    "trajectory_sensitivity": ROOT / "theory_oracle" / "bias_oracle_trajectory_signflip_v0_1.py",
}
EXPECTED_IDENTITY_SEMANTICS = {
    "SOURCE_OPERATOR_INVOCATION": "EXACT_SOURCE_GRAPH_NODE_WITH_STATE_REALIZATION_MAP",
    "GENERATED_KERNEL_INVOCATION": "EXACT_GENERATED_LAUNCH_WITH_STATE_REALIZATION_MAP",
    "GENERATED_REGION": "EXACT_GENERATED_REGION_WITH_STATE_REALIZATION_MAP",
    "INTERVENTION_PACKAGE": "DECLARED_INTERVENTION_PACKAGE_WITH_STATE_REALIZATION_MAP",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_confirmed_endpoint(
    confirmation: dict[str, Any], endpoint_name: str
) -> tuple[dict[str, Any], float | None]:
    marker = "::phase="
    if marker not in endpoint_name:
        result = confirmation.get("endpoints", {}).get(endpoint_name, {})
        value = result.get("estimate", {}).get("B", {}).get("estimate")
        return result, float(value) if isinstance(value, (int, float)) else None
    base_name, phase = endpoint_name.split(marker, 1)
    if phase not in {"early", "middle", "late"} or not base_name:
        return {}, None
    claim = (
        confirmation.get("endpoints", {})
        .get(base_name, {})
        .get("phase_conditioned_confirmation", {})
        .get("claims", {})
        .get(phase, {})
    )
    value = claim.get("estimate", {}).get("estimate")
    return claim, float(value) if isinstance(value, (int, float)) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_intervention_spec(
    spec: dict[str, Any],
    spec_path: Path,
    confirmation: dict[str, Any],
    confirmation_path: Path,
    candidate_id: str,
    unit_type: str,
    intervention_version: str,
) -> list[str]:
    errors: list[str] = []
    if spec.get("schema_version") != INTERVENTION_SCHEMA:
        errors.append(f"{candidate_id}: unsupported intervention schema")
    if spec.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PILOT":
        errors.append(f"{candidate_id}: intervention is not frozen before pilot")
    expected = (candidate_id, unit_type, intervention_version)
    observed = (
        spec.get("candidate_id"),
        spec.get("declared_intervention_unit"),
        spec.get("intervention_version"),
    )
    if observed != expected:
        errors.append(f"{candidate_id}: claim unit does not equal intervention spec")
    if spec.get("identity_semantics") != EXPECTED_IDENTITY_SEMANTICS.get(unit_type):
        errors.append(f"{candidate_id}: intervention identity semantics mismatch")
    structural = spec.get("structural_identity")
    if (
        not isinstance(structural, dict)
        or not isinstance(structural.get("identity_digest"), str)
        or not structural["identity_digest"]
        or not isinstance(structural.get("description"), str)
        or not structural["description"].strip()
    ):
        errors.append(f"{candidate_id}: structural intervention identity is incomplete")
    realization, realization_path = linked_json(
        spec.get("state_realization_map"),
        spec_path.parent,
        f"{candidate_id} state-realization map",
        "path",
        "sha256",
        errors,
    )
    if realization is None or realization_path is None:
        return errors
    if realization.get("schema_version") != REALIZATION_MAP_SCHEMA:
        errors.append(f"{candidate_id}: unsupported state-realization map schema")
    if realization.get("status") != "COMPLETE_VALID_BEFORE_CONTRIBUTOR_PILOT":
        errors.append(f"{candidate_id}: state-realization map is not complete/frozen")
    if (
        realization.get("candidate_id") != candidate_id
        or realization.get("claim_unit_type") != unit_type
        or realization.get("intervention_version") != intervention_version
    ):
        errors.append(f"{candidate_id}: state-realization map identity mismatch")
    bank = realization.get("state_bank_identity")
    bank_path = (
        resolve_from(realization_path.parent, str(bank.get("path", "")))
        if isinstance(bank, dict)
        else None
    )
    if (
        bank_path != confirmation_path
        or not isinstance(bank, dict)
        or bank.get("sha256") != sha256_file(confirmation_path)
    ):
        errors.append(f"{candidate_id}: state-realization map uses another state bank")
    expected_states = [
        row.get("state_id")
        for row in confirmation.get("state_evidence", [])
        if isinstance(row, dict)
    ]
    rows = realization.get("state_rows")
    observed_states = [
        row.get("state_id") for row in rows if isinstance(row, dict)
    ] if isinstance(rows, list) else []
    if not expected_states or observed_states != expected_states:
        errors.append(f"{candidate_id}: realization-map states/order mismatch")
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        realized = row.get("realized") if isinstance(row, dict) else None
        call_ids = row.get("exact_call_ids") if isinstance(row, dict) else None
        count = row.get("expected_call_count") if isinstance(row, dict) else None
        valid = (
            isinstance(realized, bool)
            and isinstance(call_ids, list)
            and len(call_ids) == len(set(call_ids))
            and isinstance(count, int)
            and count == len(call_ids)
            and ((realized and count > 0) or (not realized and count == 0))
        )
        if not valid:
            errors.append(f"{candidate_id}: realization-map row[{index}] is invalid")
        if realized and not isinstance(row.get("call_order_digest"), str):
            errors.append(f"{candidate_id}: realized row[{index}] lacks call-order digest")
        if realized is False and row.get("absence_verified") is not True:
            errors.append(f"{candidate_id}: absent row[{index}] lacks absence verification")
    return errors


def validate_candidate_universe(
    universe: dict[str, Any], endpoint_name: str, selected_candidate_ids: list[str]
) -> list[str]:
    errors: list[str] = []
    if universe.get("schema_version") != UNIVERSE_SCHEMA:
        errors.append("unsupported candidate-universe schema")
    if universe.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PILOT":
        errors.append("candidate universe is not frozen before contributor pilot")
    if universe.get("target_endpoint") != endpoint_name:
        errors.append("candidate universe endpoint mismatch")
    if not isinstance(universe.get("enumeration_scope"), str) or not universe[
        "enumeration_scope"
    ].strip():
        errors.append("candidate universe enumeration_scope is uninstantiated")
    scope_kind = universe.get("scope_kind")
    expected_unit_type = {
        "ALL_SOURCE_OPERATOR_INVOCATIONS_IN_FROZEN_STATE_BANK": "SOURCE_OPERATOR_INVOCATION",
        "ALL_GENERATED_KERNEL_INVOCATIONS_IN_FROZEN_STATE_BANK": "GENERATED_KERNEL_INVOCATION",
        "ALL_GENERATED_REGIONS_IN_FROZEN_STATE_BANK": "GENERATED_REGION",
        "DECLARED_INTERVENTION_SUBSPACE_IN_FROZEN_STATE_BANK": None,
    }.get(scope_kind)
    if scope_kind not in {
        "ALL_SOURCE_OPERATOR_INVOCATIONS_IN_FROZEN_STATE_BANK",
        "ALL_GENERATED_KERNEL_INVOCATIONS_IN_FROZEN_STATE_BANK",
        "ALL_GENERATED_REGIONS_IN_FROZEN_STATE_BANK",
        "DECLARED_INTERVENTION_SUBSPACE_IN_FROZEN_STATE_BANK",
    }:
        errors.append("candidate universe scope_kind is invalid")
    units = universe.get("units")
    if not isinstance(units, list) or not units:
        errors.append("candidate universe units are empty")
        units = []
    unit_ids: list[str] = []
    eligible_ids: list[str] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            errors.append(f"candidate universe unit[{index}] is not an object")
            continue
        candidate_id = unit.get("candidate_id")
        unit_type = unit.get("claim_unit_type")
        eligible = unit.get("selection_eligible")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"candidate universe unit[{index}] lacks candidate_id")
            continue
        unit_ids.append(candidate_id)
        if unit_type not in ALLOWED_UNITS:
            errors.append(f"candidate universe unit[{index}] has unsupported claim unit")
        elif expected_unit_type is not None and unit_type != expected_unit_type:
            errors.append(
                f"candidate universe unit[{index}] type disagrees with scope_kind"
            )
        if eligible is True:
            eligible_ids.append(candidate_id)
        elif eligible is False:
            if not isinstance(unit.get("exclusion_reason"), str) or not unit[
                "exclusion_reason"
            ].strip():
                errors.append(
                    f"candidate universe unit[{index}] exclusion reason is missing"
                )
        else:
            errors.append(
                f"candidate universe unit[{index}] selection_eligible is not boolean"
            )
    if len(unit_ids) != len(set(unit_ids)):
        errors.append("candidate universe IDs must be unique")
    if universe.get("selected_candidate_ids") != selected_candidate_ids:
        errors.append("candidate universe selected IDs/order mismatch")
    if not set(selected_candidate_ids).issubset(set(eligible_ids)):
        errors.append("selected candidates are not all eligible universe units")
    mode = universe.get("coverage_mode")
    scope = universe.get("claim_scope")
    all_covered = universe.get("all_eligible_units_covered_claim_allowed")
    if mode == "EXHAUSTIVE_ELIGIBLE_UNIVERSE":
        if set(selected_candidate_ids) != set(eligible_ids):
            errors.append("exhaustive universe does not select every eligible unit")
        if scope != "ALL_ELIGIBLE_UNITS_IN_FROZEN_UNIVERSE" or all_covered is not True:
            errors.append("exhaustive universe claim scope is inconsistent")
        coverage_link = universe.get("coverage_evidence")
        if not isinstance(coverage_link, dict) or not all(
            isinstance(coverage_link.get(field), str) and coverage_link[field]
            for field in ("path", "sha256")
        ):
            errors.append("exhaustive universe lacks linked coverage evidence")
    elif mode == "PREDECLARED_SUBSET_OF_UNIVERSE":
        if set(selected_candidate_ids) == set(eligible_ids):
            errors.append("subset mode must leave at least one eligible unit unselected")
        if scope != "SELECTED_CANDIDATES_ONLY" or all_covered is not False:
            errors.append("subset universe claim scope is inconsistent")
    else:
        errors.append("candidate universe coverage_mode is invalid")
    return errors


def validate_coverage_evidence(
    coverage: dict[str, Any],
    universe: dict[str, Any],
    confirmation: dict[str, Any],
    coverage_path: Path,
    confirmation_path: Path,
) -> list[str]:
    errors: list[str] = []
    if coverage.get("schema_version") != COVERAGE_SCHEMA:
        errors.append("unsupported candidate coverage schema")
    if coverage.get("status") != "COMPLETE_VALID_BEFORE_CONTRIBUTOR_PILOT":
        errors.append("candidate coverage evidence is not complete/frozen")
    if coverage.get("scope_kind") != universe.get("scope_kind"):
        errors.append("candidate coverage scope_kind mismatch")
    state_bank = coverage.get("state_bank_identity")
    state_bank_path = (
        resolve_from(coverage_path.parent, str(state_bank.get("path", "")))
        if isinstance(state_bank, dict)
        else None
    )
    if (
        state_bank_path != confirmation_path
        or not isinstance(state_bank, dict)
        or state_bank.get("sha256") != sha256_file(confirmation_path)
    ):
        errors.append("candidate coverage is not bound to endpoint-confirmation state bank")
    units = universe.get("units", [])
    unit_ids = [row.get("candidate_id") for row in units if isinstance(row, dict)]
    if coverage.get("enumerated_unit_ids") != unit_ids:
        errors.append("candidate coverage unit IDs/order do not equal universe")
    expected_state_ids = [
        row.get("state_id")
        for row in confirmation.get("state_evidence", [])
        if isinstance(row, dict)
    ]
    state_rows = coverage.get("state_rows")
    observed_state_ids = [
        row.get("state_id") for row in state_rows if isinstance(row, dict)
    ] if isinstance(state_rows, list) else []
    if (
        not expected_state_ids
        or len(expected_state_ids) != len(set(expected_state_ids))
        or observed_state_ids != expected_state_ids
    ):
        errors.append("candidate coverage state census does not equal confirmation states/order")
    observed_union: set[str] = set()
    for index, row in enumerate(state_rows if isinstance(state_rows, list) else []):
        ids = row.get("observed_unit_ids") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("census_complete") is not True
            or not isinstance(ids, list)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(set(unit_ids))
        ):
            errors.append(f"candidate coverage state row[{index}] is invalid")
            continue
        observed_union.update(ids)
    if observed_union != set(unit_ids):
        errors.append("candidate coverage observed union does not equal universe")
    artifacts = coverage.get("census_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("candidate coverage lacks raw census artifacts")
    else:
        for index, link in enumerate(artifacts):
            if not isinstance(link, dict) or not isinstance(link.get("path"), str):
                errors.append(f"candidate coverage artifact[{index}] link is invalid")
                continue
            path = resolve_from(coverage_path.parent, link["path"])
            if not path.is_file() or sha256_file(path) != link.get("sha256"):
                errors.append(f"candidate coverage artifact[{index}] hash mismatch")
    if coverage.get("census_claim_scope") != "ALL_OBSERVED_UNITS_IN_FROZEN_STATE_BANK":
        errors.append("candidate coverage claim scope exceeds or differs from observed census")
    return errors


def linked_json(
    link: Any,
    base: Path,
    label: str,
    path_key: str,
    hash_key: str,
    errors: list[str],
) -> tuple[dict[str, Any] | None, Path | None]:
    if not isinstance(link, dict) or not isinstance(link.get(path_key), str):
        errors.append(f"{label} path is uninstantiated")
        return None, None
    path = resolve_from(base, link[path_key])
    if not path.is_file():
        errors.append(f"{label} file is missing")
        return None, path
    if sha256_file(path) != link.get(hash_key):
        errors.append(f"{label} hash mismatch")
        return None, path
    try:
        return load_json(path), path
    except (OSError, json.JSONDecodeError):
        errors.append(f"{label} is not valid JSON")
        return None, path


def unique_present(rows: list[dict[str, Any]], field: str, label: str, errors: list[str]) -> None:
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        errors.append(f"{label} {field} values must be present")
    elif len(values) != len(set(values)):
        errors.append(f"{label} {field} values must be unique")


def validate_contributor_manifest(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    base = manifest_path.parent
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unsupported contributor manifest schema")
    if manifest.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_CONFIRMATION":
        errors.append("contributor manifest is not FROZEN_BEFORE_CONTRIBUTOR_CONFIRMATION")

    query = manifest.get("query", {})
    linked_json(
        {
            "path": query.get("query_manifest_path") if isinstance(query, dict) else None,
            "sha256": query.get("query_manifest_sha256") if isinstance(query, dict) else None,
        },
        base,
        "query manifest",
        "path",
        "sha256",
        errors,
    )
    if not isinstance(query, dict) or query.get("reference_implementation") != "eager_baseline_not_truth":
        errors.append("reference must remain eager_baseline_not_truth")
    for field in ("target_state_distribution", "candidate_implementation", "randomness_protocol"):
        if not isinstance(query, dict) or not query.get(field):
            errors.append(f"query.{field} is uninstantiated")

    endpoint = manifest.get("target_endpoint", {})
    endpoint_name = endpoint.get("name") if isinstance(endpoint, dict) else None
    if not isinstance(endpoint_name, str) or not endpoint_name:
        errors.append("target endpoint name is uninstantiated")
    if not isinstance(endpoint, dict) or not endpoint.get("signed_direction_definition"):
        errors.append("target endpoint signed direction is uninstantiated")
    if not isinstance(endpoint, dict) or endpoint.get("magnitude_only_endpoint_prohibited") is not True:
        errors.append("magnitude-only endpoints must be prohibited")
    target_phase = None
    if isinstance(endpoint_name, str) and "::phase=" in endpoint_name:
        _, target_phase = endpoint_name.split("::phase=", 1)
        if target_phase not in {"early", "middle", "late"}:
            errors.append("target endpoint phase condition is invalid")
    expected_state_condition = (
        {"kind": "PHASE", "value": target_phase}
        if target_phase is not None
        else {"kind": "GLOBAL"}
    )
    if not isinstance(query, dict) or query.get("target_state_condition") != expected_state_condition:
        errors.append("query target_state_condition does not match endpoint identity")

    confirmation_link = manifest.get("endpoint_confirmation_gate", {})
    confirmation, confirmation_path = linked_json(
        {
            "path": confirmation_link.get("evaluation_path") if isinstance(confirmation_link, dict) else None,
            "sha256": confirmation_link.get("evaluation_sha256") if isinstance(confirmation_link, dict) else None,
        },
        base,
        "endpoint confirmation evaluation",
        "path",
        "sha256",
        errors,
    )
    confirmed_endpoint_B: float | None = None
    if confirmation is not None:
        if confirmation.get("schema_version") != CONFIRMATION_SCHEMA or confirmation.get("valid") is not True:
            errors.append("linked endpoint confirmation is not valid")
        result, confirmed_endpoint_B = resolve_confirmed_endpoint(
            confirmation, endpoint_name or ""
        )
        if result.get("final_shift_verdict") != "REPRODUCIBLE_AVERAGE_SHIFT":
            errors.append("target endpoint lacks REPRODUCIBLE_AVERAGE_SHIFT")
        if endpoint_name not in confirmation.get("operator_attribution_gate", {}).get("ready_endpoints", []):
            errors.append("target endpoint is absent from confirmation operator-ready gate")
        if confirmation.get("operator_attribution_gate", {}).get("automatic_operator_launch") is not False:
            errors.append("endpoint confirmation must not automatically launch attribution")
    if not isinstance(confirmation_link, dict) or confirmation_link.get("observed_verdict") != "REPRODUCIBLE_AVERAGE_SHIFT":
        errors.append("manifest observed endpoint verdict is not frozen as reproducible shift")
    if not isinstance(confirmation_link, dict) or confirmation_link.get("method_sensitivity_conflict_prohibited") is not True:
        errors.append("method-sensitivity conflicts must be prohibited")

    candidate_freeze, candidate_path = linked_json(
        manifest.get("candidate_freeze"),
        base,
        "candidate-freeze artifact",
        "artifact_path",
        "artifact_sha256",
        errors,
    )
    candidates: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    if candidate_freeze is not None:
        if candidate_freeze.get("schema_version") != CANDIDATE_SCHEMA:
            errors.append("unsupported candidate-freeze schema")
        if candidate_freeze.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PILOT":
            errors.append("candidate artifact is not frozen before contributor pilot")
        if candidate_freeze.get("target_endpoint") != endpoint_name:
            errors.append("candidate artifact target endpoint mismatch")
        c_link = candidate_freeze.get("endpoint_confirmation_evaluation", {})
        if confirmation_path is not None:
            if not isinstance(c_link, dict) or resolve_from(candidate_path.parent, str(c_link.get("path", ""))) != confirmation_path:
                errors.append("candidate artifact confirmation path mismatch")
            elif c_link.get("sha256") != sha256_file(confirmation_path):
                errors.append("candidate artifact confirmation hash mismatch")
        if candidate_freeze.get("selection_data_role") != "DISCOVERY_ONLY":
            errors.append("candidate selection data role must be DISCOVERY_ONLY")
        if not candidate_freeze.get("selection_rule") or not candidate_freeze.get("freeze_timestamp_utc"):
            errors.append("candidate selection rule/timestamp is uninstantiated")
        rows = candidate_freeze.get("primary_candidates")
        if not isinstance(rows, list) or not rows:
            errors.append("primary candidate set is empty")
        else:
            candidates = [row for row in rows if isinstance(row, dict)]
            if len(candidates) != len(rows):
                errors.append("every primary candidate must be an object")
            for index, row in enumerate(candidates):
                missing = [field for field in CANDIDATE_FIELDS if not row.get(field)]
                if missing:
                    errors.append(f"candidate[{index}] missing {missing}")
                    continue
                if row["claim_unit_type"] not in ALLOWED_UNITS:
                    errors.append(f"candidate[{index}] has unsupported claim unit")
                spec_path = resolve_from(candidate_path.parent, row["intervention_spec_path"])
                if not spec_path.is_file() or sha256_file(spec_path) != row["intervention_spec_sha256"]:
                    errors.append(f"candidate[{index}] intervention spec missing/hash mismatch")
                else:
                    spec = load_json(spec_path)
                    if confirmation is None or confirmation_path is None:
                        errors.append(
                            f"candidate[{index}] cannot validate intervention without endpoint confirmation"
                        )
                    else:
                        errors.extend(
                            validate_intervention_spec(
                                spec,
                                spec_path,
                                confirmation,
                                confirmation_path,
                                row["candidate_id"],
                                row["claim_unit_type"],
                                row["intervention_version"],
                            )
                        )
            candidate_ids = [row.get("candidate_id") for row in candidates if row.get("candidate_id")]
            if len(candidate_ids) != len(set(candidate_ids)):
                errors.append("candidate IDs must be unique")
        if candidate_freeze.get("claim_unit_must_equal_intervention_unit") is not True:
            errors.append("claim unit equality gate must be true")
        universe, universe_path = linked_json(
            candidate_freeze.get("candidate_universe"),
            candidate_path.parent,
            "candidate-universe artifact",
            "path",
            "sha256",
            errors,
        )
        if universe is not None:
            errors.extend(
                validate_candidate_universe(
                    universe, endpoint_name or "", candidate_ids
                )
            )
            state_bank = universe.get("state_bank_identity", {})
            state_bank_path = (
                resolve_from(universe_path.parent, str(state_bank.get("path", "")))
                if universe_path is not None and isinstance(state_bank, dict)
                else None
            )
            if (
                confirmation_path is None
                or state_bank_path != confirmation_path
                or not isinstance(state_bank, dict)
                or state_bank.get("sha256") != sha256_file(confirmation_path)
            ):
                errors.append(
                    "candidate universe is not bound to the endpoint-confirmation state bank"
                )
            if universe.get("coverage_mode") == "EXHAUSTIVE_ELIGIBLE_UNIVERSE":
                coverage, coverage_path = linked_json(
                    universe.get("coverage_evidence"),
                    universe_path.parent,
                    "candidate coverage evidence",
                    "path",
                    "sha256",
                    errors,
                )
                if (
                    coverage is not None
                    and coverage_path is not None
                    and confirmation is not None
                    and confirmation_path is not None
                ):
                    errors.extend(
                        validate_coverage_evidence(
                            coverage,
                            universe,
                            confirmation,
                            coverage_path,
                            confirmation_path,
                        )
                    )

    precision, precision_path = linked_json(
        manifest.get("contributor_precision_plan"),
        base,
        "contributor precision plan",
        "path",
        "sha256",
        errors,
    )

    analysis_implementation = manifest.get("analysis_implementation", {})
    if not isinstance(analysis_implementation, dict):
        errors.append("analysis implementation links are uninstantiated")
    else:
        for name, expected_path in EXPECTED_ANALYSIS_FILES.items():
            path_value = analysis_implementation.get(f"{name}_path")
            hash_value = analysis_implementation.get(f"{name}_sha256")
            if not isinstance(path_value, str):
                errors.append(f"analysis {name} path is uninstantiated")
                continue
            observed_path = resolve_from(base, path_value)
            if observed_path != expected_path:
                errors.append(f"analysis {name} path is not the frozen implementation")
            elif not observed_path.is_file() or sha256_file(observed_path) != hash_value:
                errors.append(f"analysis {name} hash mismatch")
    planned: int | None = None
    pilot_rows: list[dict[str, Any]] = []
    if precision is not None:
        if precision.get("schema_version") != PRECISION_SCHEMA:
            errors.append("unsupported contributor precision schema")
        if precision.get("valid") is not True or precision.get("verdict") != "VALID_FROZEN_CONTRIBUTOR_PRECISION_PLAN":
            errors.append("contributor precision plan is not valid/frozen")
        if precision.get("target_endpoint") != endpoint_name:
            errors.append("precision target endpoint mismatch")
        injection_precision = precision.get("injection_precision", {})
        if (
            not isinstance(injection_precision, dict)
            or injection_precision.get("status")
            != "UNINSTANTIATED_SEPARATE_PRECISION_PLAN_REQUIRED"
            or injection_precision.get(
                "repair_alpha_variance_and_trajectory_count_reuse_prohibited"
            )
            is not True
        ):
            errors.append(
                "repair precision plan must leave injection as a separate uninstantiated plan"
            )
        direction = precision.get("frozen_bias_direction", {})
        if not isinstance(direction, dict) or direction.get("source") != "ENDPOINT_CONFIRMATION_ONLY":
            errors.append("Bias direction must come only from endpoint confirmation")
        elif direction.get("kind") == "SCALAR_SIGN":
            sign = direction.get("scalar_sign")
            if sign not in (-1, 1):
                errors.append("frozen scalar Bias sign must be -1 or +1")
            elif confirmation is not None:
                b_value = confirmed_endpoint_B
                if not isinstance(b_value, (int, float)) or b_value == 0 or (1 if b_value > 0 else -1) != sign:
                    errors.append("frozen scalar Bias sign disagrees with endpoint confirmation")
        elif direction.get("kind") == "VECTOR_DIRECTION_ARTIFACT":
            linked_json(
                {
                    "path": direction.get("vector_direction_artifact_path"),
                    "sha256": direction.get("vector_direction_artifact_sha256"),
                },
                precision_path.parent,
                "frozen vector Bias direction",
                "path",
                "sha256",
                errors,
            )
        else:
            errors.append("frozen Bias direction kind is uninstantiated")
        p_link = precision.get("candidate_freeze", {})
        if candidate_path is not None:
            if not isinstance(p_link, dict) or resolve_from(precision_path.parent, str(p_link.get("artifact_path", ""))) != candidate_path:
                errors.append("precision candidate artifact path mismatch")
            elif p_link.get("artifact_sha256") != sha256_file(candidate_path):
                errors.append("precision candidate artifact hash mismatch")
        if not isinstance(p_link, dict) or p_link.get("candidate_ids") != candidate_ids:
            errors.append("precision candidate IDs/order mismatch")
        basis = precision.get("precision_basis", {})
        mode = basis.get("mode") if isinstance(basis, dict) else None
        source_spec, source_spec_path = linked_json(
            precision.get("inputs", {}).get("spec") if isinstance(precision.get("inputs"), dict) else None,
            precision_path.parent,
            "contributor precision input spec",
            "path",
            "sha256",
            errors,
        )
        source_pilot: dict[str, Any] | None = None
        if source_spec is not None:
            if source_spec.get("schema_version") != CONTRIBUTOR_PRECISION_SPEC_VERSION or source_spec.get("status") != "FROZEN_BEFORE_CONTRIBUTOR_PRECISION_PLANNING":
                errors.append("linked contributor precision input spec is not frozen")
            source_candidate_link = source_spec.get("candidate_freeze", {})
            if candidate_path is not None:
                source_base = source_spec_path.parent if source_spec_path is not None else precision_path.parent
                if not isinstance(source_candidate_link, dict) or resolve_from(source_base, str(source_candidate_link.get("artifact_path", ""))) != candidate_path:
                    errors.append("precision input spec candidate artifact path mismatch")
                elif source_candidate_link.get("artifact_sha256") != sha256_file(candidate_path):
                    errors.append("precision input spec candidate artifact hash mismatch")
            if mode == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY":
                source_pilot, _ = linked_json(
                    basis.get("contributor_pilot") if isinstance(basis, dict) else None,
                    precision_path.parent,
                    "contributor pilot summary",
                    "summary_path",
                    "summary_sha256",
                    errors,
                )
            if candidate_freeze is not None:
                recomputed = plan_contributor_precision(candidate_freeze, source_spec, source_pilot)
                if not recomputed.get("valid"):
                    errors.append("contributor precision plan cannot be reproduced from frozen inputs")
                for key in (
                    "target_endpoint",
                    "frozen_bias_direction",
                    "multiplicity",
                    "variance_upper_confidence",
                    "candidate_precision",
                    "global_design",
                    "sensitivity",
                ):
                    if precision.get(key) != recomputed.get(key):
                        errors.append(f"contributor precision plan {key} is not reproducible")
        if mode not in {
            "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY",
            "EXTERNAL_FIXED_CONSERVATIVE_VARIANCE",
        }:
            errors.append("unsupported/uninstantiated contributor precision mode")
        pilot = basis.get("contributor_pilot", {}) if isinstance(basis, dict) else {}
        raw_pilot_rows = pilot.get("trajectory_specs", []) if isinstance(pilot, dict) else []
        pilot_rows = [row for row in raw_pilot_rows if isinstance(row, dict)]
        if mode == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY" and not pilot_rows:
            errors.append("independent contributor precision mode requires pilot trajectories")
        if mode == "EXTERNAL_FIXED_CONSERVATIVE_VARIANCE" and not basis.get("external_fixed_rationale"):
            errors.append("external fixed precision mode requires rationale")
        if pilot.get("trajectory_level_repair_effect_dispersion_only") is not True or pilot.get("mean_and_sign_excluded_from_sizing_and_candidate_selection") is not True or pilot.get("excluded_from_final_confirmation_df") is not True:
            errors.append("contributor pilot role restrictions are incomplete")
        for field in ("trajectory_id", "trajectory_seed", "data_slice_id"):
            unique_present(pilot_rows, field, "contributor pilot", errors)

        multiplicity = precision.get("multiplicity", {})
        members = multiplicity.get("primary_repair_family_members") if isinstance(multiplicity, dict) else None
        if members != candidate_ids:
            errors.append("precision multiplicity family does not equal frozen candidates")
        if multiplicity.get("method") != "BONFERRONI_SIMULTANEOUS_TWO_SIDED":
            errors.append("only frozen Bonferroni simultaneous intervals are currently validated")
        alpha = multiplicity.get("family_alpha")
        per_alpha = multiplicity.get("per_interval_alpha")
        if not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1 or not candidate_ids:
            errors.append("invalid contributor family alpha")
        elif not isinstance(per_alpha, (int, float)) or not math.isclose(float(per_alpha), float(alpha) / len(candidate_ids), rel_tol=0.0, abs_tol=1e-15):
            errors.append("contributor per-interval alpha is not Bonferroni-adjusted")
        candidate_precision = precision.get("candidate_precision", {})
        if set(candidate_precision) != set(candidate_ids):
            errors.append("candidate precision entries do not equal frozen candidates")
        required_numeric = (
            "desired_halfwidth",
            "directional_contribution_floor",
            "trajectory_variance_floor",
            "variance_plan",
            "planned_trajectories",
        )
        for candidate_id in candidate_ids:
            row = candidate_precision.get(candidate_id, {})
            for field in required_numeric:
                if not isinstance(row.get(field), (int, float)) or float(row[field]) < 0:
                    errors.append(f"candidate precision {candidate_id}.{field} is invalid")
        design = precision.get("global_design", {})
        planned = design.get("planned_confirmation_trajectories") if isinstance(design, dict) else None
        cap = design.get("resource_cap_trajectories") if isinstance(design, dict) else None
        declared_minimum = design.get("minimum_confirmation_trajectories") if isinstance(design, dict) else None
        if not isinstance(declared_minimum, int) or declared_minimum < 8:
            errors.append("contributor minimum confirmation trajectories must be at least 8")
        if not isinstance(planned, int) or planned < 8 or (
            isinstance(declared_minimum, int) and planned < declared_minimum
        ):
            errors.append("planned contributor trajectories must be at least 8")
        if not isinstance(cap, int) or not isinstance(planned, int) or cap < planned:
            errors.append("contributor resource cap is invalid")
        if design.get("no_optional_stopping_on_mean_or_sign") is not True or not design.get("tail_scope"):
            errors.append("contributor stopping/tail rules are uninstantiated")
        for candidate_id in candidate_ids:
            row = candidate_precision.get(candidate_id, {})
            if isinstance(row.get("desired_halfwidth"), (int, float)) and float(row["desired_halfwidth"]) <= 0:
                errors.append(f"candidate precision {candidate_id}.desired_halfwidth must be positive")
            candidate_planned = row.get("planned_trajectories")
            if not isinstance(candidate_planned, int) or candidate_planned < 8:
                errors.append(f"candidate precision {candidate_id}.planned_trajectories must be an integer >= 8")
            elif isinstance(planned, int) and candidate_planned > planned:
                errors.append(f"global contributor plan is below {candidate_id} requirement")
            variance_floor = row.get("trajectory_variance_floor")
            variance_upper = row.get("trajectory_variance_upper_bound")
            if mode == "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY" and not isinstance(variance_upper, (int, float)):
                errors.append(f"candidate precision {candidate_id} lacks pilot variance upper bound")
            if isinstance(variance_floor, (int, float)) and isinstance(variance_upper, (int, float)) and float(variance_upper) < float(variance_floor):
                errors.append(f"candidate precision {candidate_id} variance upper bound is below floor")
        sensitivity = precision.get("sensitivity", {})
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
        elif isinstance(per_alpha, (int, float)) and isinstance(planned, int) and isinstance(cap, int):
            resolution_required = signflip_resolution_requirement(
                float(per_alpha),
                8,
                cap,
                exact_max_trajectories=int(sensitivity["exact_max_trajectories"]),
                monte_carlo_draws=int(sensitivity["monte_carlo_draws"]),
            )
            if resolution_required is None or planned < resolution_required:
                errors.append("planned contributor trajectories cannot resolve adjusted sensitivity alpha")
            if sensitivity.get("minimum_trajectories_for_p_value_resolution") != resolution_required:
                errors.append("stored contributor sensitivity resolution requirement mismatch")
            expected_minimum_p = minimum_attainable_signflip_p(
                planned,
                exact_max_trajectories=int(sensitivity["exact_max_trajectories"]),
                monte_carlo_draws=int(sensitivity["monte_carlo_draws"]),
            )
            observed_minimum_p = sensitivity.get("minimum_attainable_p_at_planned_count")
            if not isinstance(observed_minimum_p, (int, float)) or not math.isclose(
                float(observed_minimum_p), expected_minimum_p, rel_tol=0.0, abs_tol=1e-15
            ):
                errors.append("stored contributor minimum attainable sensitivity p mismatch")

    bank = manifest.get("contributor_confirmation_bank", {})
    precision_link = manifest.get("contributor_precision_plan", {})
    if not isinstance(precision_link, dict) or precision_link.get("planned_confirmation_trajectories") != planned:
        errors.append("manifest contributor trajectory count disagrees with precision plan")
    if not isinstance(precision_link, dict) or precision_link.get("pilot_trajectories_excluded_from_confirmation_df") is not True:
        errors.append("contributor pilot trajectories must be excluded from confirmation df")
    for field in (
        "independent_of_calibration",
        "independent_of_endpoint_confirmation",
        "independent_of_contributor_precision_pilot",
    ):
        if not isinstance(bank, dict) or bank.get(field) is not True:
            errors.append(f"contributor bank {field} must be true")
    bank_rows_raw = bank.get("trajectory_specs", []) if isinstance(bank, dict) else []
    bank_rows = [row for row in bank_rows_raw if isinstance(row, dict)]
    if len(bank_rows) != len(bank_rows_raw):
        errors.append("every contributor trajectory spec must be an object")
    if not isinstance(planned, int) or bank.get("trajectory_count") != planned or len(bank_rows) != planned:
        errors.append("contributor bank trajectory count does not match precision plan")
    for field in TRAJECTORY_FIELDS:
        unique_present(bank_rows, field, "contributor confirmation", errors)

    forbidden: dict[str, set[Any]] = {field: set() for field in ("trajectory_id", "trajectory_seed", "data_slice_id")}
    if confirmation is not None:
        c_manifest_link = confirmation.get("manifest", {})
        c_manifest, _ = linked_json(c_manifest_link, confirmation_path.parent, "linked endpoint-confirmation manifest", "path", "sha256", errors)
        if c_manifest is not None:
            for row in c_manifest.get("trajectory_inputs", []):
                for field in forbidden:
                    forbidden[field].add(row.get(field))
            exclusion = c_manifest.get("calibration_exclusion", {})
            for field, plural in (("trajectory_id", "trajectory_ids"), ("trajectory_seed", "trajectory_seeds"), ("data_slice_id", "data_slice_ids")):
                forbidden[field].update(exclusion.get(plural, []))
    for row in pilot_rows:
        for field in forbidden:
            forbidden[field].add(row.get(field))
    for field in forbidden:
        overlap = forbidden[field].intersection(row.get(field) for row in bank_rows)
        if overlap:
            errors.append(f"contributor confirmation reuses prior-role {field}: {sorted(overlap, key=str)}")
    declared_exclusions = bank.get("all_prior_role_exclusions", []) if isinstance(bank, dict) else []
    declared: dict[str, set[Any]] = {field: set() for field in forbidden}
    for row in declared_exclusions if isinstance(declared_exclusions, list) else []:
        if isinstance(row, dict):
            for field in declared:
                if row.get(field) is not None:
                    declared[field].add(row[field])
    for field in forbidden:
        if not forbidden[field].issubset(declared[field]):
            errors.append(f"all_prior_role_exclusions does not cover {field}")

    for index, row in enumerate(bank_rows):
        if any(field not in row for field in TRAJECTORY_FIELDS):
            continue
        plan_path = resolve_from(base, row["capture_plan_path"])
        if not plan_path.is_file() or sha256_file(plan_path) != row["capture_plan_sha256"]:
            errors.append(f"contributor trajectory[{index}] plan missing/hash mismatch")
            continue
        plan = load_json(plan_path)
        identity = plan.get("identity", {})
        if plan.get("schema_version") != PLAN_SCHEMA:
            errors.append(f"contributor trajectory[{index}] plan schema invalid")
        for field in ("trajectory_id", "trajectory_seed", "data_slice_id"):
            if identity.get(field) != row[field]:
                errors.append(f"contributor trajectory[{index}] {field} disagrees with plan")
        targets = plan.get("targets", [])
        if len(targets) != 24 or Counter(item.get("phase") for item in targets) != Counter({"early": 8, "middle": 8, "late": 8}):
            errors.append(f"contributor trajectory[{index}] lacks frozen 8x3 states")

    family = manifest.get("hypothesis_family", {})
    if not isinstance(family, dict) or family.get("members") != candidate_ids:
        errors.append("manifest hypothesis family does not equal frozen candidates")
    if precision is not None:
        multiplicity = precision.get("multiplicity", {})
        for observed_key, expected_key in (("multiplicity_method", "method"), ("family_alpha", "family_alpha"), ("adjusted_interval_alpha", "per_interval_alpha")):
            if family.get(observed_key) != multiplicity.get(expected_key):
                errors.append(f"manifest hypothesis {observed_key} mismatches precision plan")
    if family.get("injection_is_separate_family") is not True:
        errors.append("injection must be a separate hypothesis family")

    arms = manifest.get("arms", {})
    if set(arms.get("required", [])) != {"REFERENCE", "FULL_CANDIDATE", "CANDIDATE_REPAIR"}:
        errors.append("required contributor arms drifted")
    if not isinstance(arms.get("paired_transition_repeats"), int) or arms["paired_transition_repeats"] < 2:
        errors.append("at least two paired transition repeats are required")
    estimand = manifest.get("primary_estimand", {})
    if estimand.get("name") != "C_REPAIR" or estimand.get("cluster_unit") != "independent_trajectory":
        errors.append("primary contributor estimand/cluster unit drifted")
    if estimand.get("confirmation_scale") != "PROJECTION_ON_ENDPOINT_CONFIRMATION_FROZEN_BIAS_DIRECTION":
        errors.append("contributor confirmation scale is not the frozen Bias direction")
    if estimand.get("target_phase") != target_phase:
        errors.append("primary estimand target_phase does not match endpoint identity")

    integrity = manifest.get("intervention_integrity_gates", {})
    required_integrity = (
        "pre_state_identity", "baseline_anchor", "exact_call_identity_and_count",
        "exactly_declared_replacements", "no_unexpected_recompile",
        "unaffected_realization_identity", "restoration_anchor", "sham_control",
        "complete_endpoint_validity",
    )
    if any(integrity.get(field) is not True for field in required_integrity) or integrity.get("on_failure") != "INVALID_INTERVENTION_NOT_NULL":
        errors.append("intervention-integrity gates are incomplete")
    interaction = manifest.get("interaction_policy", {})
    if interaction.get("additivity_assumed") is not False or interaction.get("unmeasured_interactions_label") != "INTERACTION_UNRESOLVED":
        errors.append("interaction policy must not assume additivity")
    claims = manifest.get("claim_boundary", {})
    if claims.get("correctness_authority") is not None or any(claims.get(field) is not False for field in ("root_cause_claim", "necessity_claim", "sufficiency_claim", "automatic_long_training_claim")):
        errors.append("claim boundary exceeds implementation-relative attribution")

    return candidate_freeze, precision, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    manifest = load_json(path)
    candidates, precision, errors = validate_contributor_manifest(manifest, path)
    result = {
        "schema_version": "forkcert.qwen3-bias-contributor-design-validation.v0.1",
        "valid": not errors,
        "verdict": "VALID_FROZEN_CONTRIBUTOR_DESIGN" if not errors else "INVALID_OR_UNINSTANTIATED_CONTRIBUTOR_DESIGN",
        "manifest": {"path": str(path), "sha256": sha256_file(path)},
        "candidate_count": len(candidates.get("primary_candidates", [])) if candidates else 0,
        "planned_confirmation_trajectories": precision.get("global_design", {}).get("planned_confirmation_trajectories") if precision else None,
        "errors": errors,
        "automatic_experiment_launch": False,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
