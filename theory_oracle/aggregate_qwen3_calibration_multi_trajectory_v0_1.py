#!/usr/bin/env python
"""Aggregate four complete calibration trajectories with trajectory-level df."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (  # noqa: E402
    collect_endpoint,
    endpoint_extractors,
    load_complete_state_bundles,
    load_json,
    sha256_file,
)
from theory_oracle.bias_oracle_population_v0_2 import (  # noqa: E402
    estimate_scalar_population,
)
from theory_oracle.evaluate_qwen3_calibration_state_endpoints_v0_1 import (  # noqa: E402
    SCHEMA_VERSION as TASK_ENDPOINT_SCHEMA,
    TASK_ENDPOINT_RANDOMNESS_SCOPE,
    endpoint_profile,
)


SCHEMA_VERSION = "forkcert.qwen3-calibration-multi-trajectory-aggregate.v0.1"
EXPECTED_TRAJECTORIES = {f"calibration-{index}" for index in range(4)}
REQUIRED_PHASES = ("early", "middle", "late")
ANALYSIS_CODE_PATHS = {
    "multi_trajectory_aggregator": Path(__file__).resolve(),
    "record_loader": ROOT / "theory_oracle" / "aggregate_qwen3_calibration_records_v0_1.py",
    "population_estimator": ROOT / "theory_oracle" / "bias_oracle_population_v0_2.py",
    "record_validator": ROOT / "theory_oracle" / "bias_oracle_record_v0_2.py",
    "task_semantics_validator": ROOT
    / "theory_oracle"
    / "evaluate_qwen3_calibration_state_endpoints_v0_1.py",
}


def _profile_signature(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_effect_signed_mean": profile.get("state_effect_signed_mean"),
        "N_transition_paired_effect_variance": profile.get(
            "N_transition_paired_effect_variance"
        ),
        "paired_transition_repeat_effects": [
            {
                key: row.get(key)
                for key in (
                    "transition_repeat",
                    "reference_evaluator_values",
                    "candidate_evaluator_values",
                    "reference_evaluator_mean",
                    "candidate_evaluator_mean",
                    "paired_effect",
                    "evaluator_paired_effect_variance",
                )
            }
            for row in profile.get("paired_transition_repeat_effects", [])
        ],
    }


def _numeric_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and not isinstance(right, bool) and math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-15
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _numeric_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(
            _numeric_tree_equal(left[key], right[key]) for key in left
        )
    return left == right


def validate_task_semantics(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    links = [
        row.get("provenance", {}).get("task_evaluation", {})
        for row in bundle.get("arm_records", [])
    ]
    identities = {
        (row.get("path"), row.get("sha256"))
        for row in links
        if isinstance(row, dict)
    }
    if len(links) != 4 or len(identities) != 1:
        return {}, ["all four arms must link one exact task evaluation"]
    path_value, expected_hash = next(iter(identities))
    path = Path(path_value).resolve() if isinstance(path_value, str) else None
    if path is None or not path.is_file() or sha256_file(path) != expected_hash:
        return {}, ["task evaluation artifact link/hash failed"]
    try:
        task = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}, ["task evaluation is unreadable"]
    if task.get("schema_version") != TASK_ENDPOINT_SCHEMA or task.get("valid") is not True:
        errors.append("task evaluation schema/validity failed")
    for endpoint_name, value_key in (("T1a", "loss"), ("T1b", "mean_nll")):
        endpoint = task.get(endpoint_name, {})
        try:
            recomputed = endpoint_profile(endpoint.get("arm_results", {}), value_key)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{endpoint_name} current semantics validation failed: {error}")
            continue
        if endpoint.get("status") == "MEASURED":
            stored = endpoint.get("profile")
            if not isinstance(stored, dict) or not _numeric_tree_equal(
                _profile_signature(recomputed), _profile_signature(stored)
            ):
                errors.append(f"{endpoint_name} stored profile differs from current semantics")
    artifacts = task.get("artifacts", {})
    bank_links = [artifacts.get("T1a_bank", {}), artifacts.get("T1a_bank_repeat", {})]
    bank_contents = set()
    for index, link in enumerate(bank_links, 1):
        bank_path_value = link.get("path") if isinstance(link, dict) else None
        bank_path = Path(bank_path_value).resolve() if isinstance(bank_path_value, str) else None
        if bank_path is None or not bank_path.is_file() or sha256_file(bank_path) != link.get("sha256"):
            errors.append(f"T1a bank {index} artifact link/hash failed")
        bank_contents.add(link.get("content_sha256") if isinstance(link, dict) else None)
    if None in bank_contents or len(bank_contents) != 1:
        errors.append("T1a bank fresh generations are not content-identical")
    randomness = task.get("randomness_decomposition")
    if randomness is not None and randomness != TASK_ENDPOINT_RANDOMNESS_SCOPE:
        errors.append("task randomness decomposition drifted")
    return {
        "path": str(path) if path is not None else None,
        "sha256": expected_hash,
        "recorded_evaluator_code_sha256": task.get("environment", {}).get(
            "evaluator_code_sha256"
        ),
        "randomness_scope": (
            "EXPLICIT_CURRENT_SCOPE"
            if randomness == TASK_ENDPOINT_RANDOMNESS_SCOPE
            else "LEGACY_IMPLICIT_SCOPE_NUMERICALLY_REVALIDATED"
        ),
        "numeric_profiles_match_current_semantics": not errors,
    }, errors


def complete_endpoint_payload(
    endpoint_class: str,
    availability: Counter[str],
    unavailable: list[dict[str, Any]],
    estimate: dict[str, Any],
) -> dict[str, Any]:
    """Name signed shifts and nonnegative profiles without semantic leakage."""

    common = {
        "status": "COMPLETE_FOUR_TRAJECTORY_CALIBRATION_DESCRIPTION",
        "endpoint_class": endpoint_class,
        "availability": dict(availability),
        "unavailable_records": unavailable,
        "H": estimate["H"],
        "N": estimate["N"],
        "U": estimate["U"],
        "trajectory_rows": estimate["trajectory_rows"],
        "phase_rows": estimate["phase_rows"],
        "state_rows": estimate["state_rows"],
        "confirmation_design_input_only": True,
        "population_B_claim_allowed": False,
    }
    if endpoint_class.startswith("SIGNED_"):
        common.update(
            {
                "central_estimand": "IMPLEMENTATION_RELATIVE_SIGNED_AVERAGE_SHIFT",
                "calibration_average_estimate": estimate["B"],
                "conditional_B": estimate["conditional_B"],
                "signed_B_candidate": True,
            }
        )
    else:
        common.update(
            {
                "central_estimand": "NONNEGATIVE_OR_ONE_SIDED_PROFILE_MEAN",
                "calibration_profile_mean": estimate["B"],
                "conditional_profile_means": estimate["conditional_B"],
                "signed_B_candidate": False,
            }
        )
    return common


def validate_calibration_layout(plans: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    identities = [plan.get("identity", {}) for plan in plans]
    trajectory_ids = [identity.get("trajectory_id") for identity in identities]
    if set(trajectory_ids) != EXPECTED_TRAJECTORIES or len(trajectory_ids) != 4:
        errors.append("calibration requires exactly calibration-0..calibration-3")
    for field in ("trajectory_seed", "data_slice_id"):
        values = [identity.get(field) for identity in identities]
        if None in values or len(values) != len(set(values)):
            errors.append(f"calibration {field} values must be present and distinct")
    for field, expected in (
        ("query_id", "Q-R"),
        ("trajectory_anchor", "EAGER_TRAJECTORY"),
    ):
        if any(identity.get(field) != expected for identity in identities):
            errors.append(f"all calibration plans require {field}={expected}")
    selection_rules = {
        identity.get("state_selection_prng_seed") for identity in identities
    }
    if len(selection_rules) != 1 or None in selection_rules:
        errors.append("all calibration plans require one shared state-selection rule")
    for plan in plans:
        identity = plan.get("identity", {})
        trajectory = identity.get("trajectory_id", "UNKNOWN")
        targets = plan.get("targets", [])
        phase_counts = Counter(target.get("phase") for target in targets)
        if len(targets) != 24 or phase_counts != Counter(
            {"early": 8, "middle": 8, "late": 8}
        ):
            errors.append(f"{trajectory} does not have the frozen 8x3 state layout")
        if len({target.get("state_id") for target in targets}) != len(targets):
            errors.append(f"{trajectory} has duplicate state IDs")
        if len({target.get("optimizer_step") for target in targets}) != len(targets):
            errors.append(f"{trajectory} has duplicate optimizer steps")
    return errors


def validate_capture_audit(
    plan_path: Path, plan: dict[str, Any], results_root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    audit_path = results_root / "capture_batch_audit.json"
    if not audit_path.is_file():
        return None, [f"missing capture audit: {audit_path}"]
    try:
        audit = load_json(audit_path)
    except (OSError, json.JSONDecodeError):
        return None, [f"invalid capture audit JSON: {audit_path}"]
    checks = audit.get("checks")
    source = audit.get("source_evidence")
    if audit.get("valid") is not True or audit.get("verdict") != "VALID":
        errors.append("capture audit is not valid")
    if audit.get("plan_sha256") != sha256_file(plan_path):
        errors.append("capture audit plan hash mismatch")
    if Path(audit.get("capture_root", "")).resolve() != Path(
        plan.get("capture_root", "")
    ).resolve():
        errors.append("capture audit root mismatch")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        errors.append("capture audit checks are incomplete")
    if not isinstance(source, dict) or not isinstance(source.get("checks"), dict) or not all(
        source["checks"].values()
    ):
        errors.append("capture audit source checks are incomplete")
        source = {}
    for label, path_key, hash_key in (
        ("source config", "config_path", "config_sha256"),
        ("source metadata", "metadata_path", "metadata_sha256"),
    ):
        value = source.get(path_key)
        linked = Path(value).resolve() if isinstance(value, str) else None
        if linked is None or not linked.is_file() or sha256_file(linked) != source.get(
            hash_key
        ):
            errors.append(f"capture audit {label} link failed")
    rows = audit.get("states")
    expected_ids = {row.get("state_id") for row in plan.get("targets", [])}
    observed_ids = {
        row.get("state_id") for row in rows if isinstance(row, dict)
    } if isinstance(rows, list) else set()
    if observed_ids != expected_ids or len(observed_ids) != 24:
        errors.append("capture audit state census does not equal frozen targets")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            errors.append("capture audit state row is invalid")
            continue
        linked_value = row.get("audit_path")
        linked = Path(linked_value).resolve() if isinstance(linked_value, str) else None
        if (
            row.get("snapshot_valid") is not True
            or row.get("history_exact") is not True
            or row.get("target_identity_exact") is not True
            or linked is None
            or not linked.is_file()
            or sha256_file(linked) != row.get("audit_sha256")
        ):
            errors.append(f"capture audit state link failed: {row.get('state_id')}")
    evidence = {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "source_config": {
            "path": source.get("config_path"),
            "sha256": source.get("config_sha256"),
        },
        "source_metadata": {
            "path": source.get("metadata_path"),
            "sha256": source.get("metadata_sha256"),
        },
        "state_audits": len(observed_ids),
    }
    return evidence, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        nargs=2,
        metavar=("PLAN", "RESULTS_ROOT"),
        required=True,
        help="Repeat exactly four times, once for each frozen calibration trajectory.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    entries = [
        (Path(plan).resolve(), Path(results_root).resolve())
        for plan, results_root in args.trajectory
    ]
    plans = [load_json(plan_path) for plan_path, _ in entries]
    errors = validate_calibration_layout(plans)
    all_bundles: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_evidence: list[dict[str, Any]] = []
    task_semantics_evidence: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    expected_effect_records = 0
    for (plan_path, results_root), plan in zip(entries, plans, strict=True):
        capture_evidence, capture_errors = validate_capture_audit(
            plan_path, plan, results_root
        )
        errors.extend(
            f"{plan.get('identity', {}).get('trajectory_id')}: {error}"
            for error in capture_errors
        )
        bundles, evidence, current_errors = load_complete_state_bundles(
            plan, results_root
        )
        trajectory_id = plan.get("identity", {}).get("trajectory_id")
        errors.extend(f"{trajectory_id}: {error}" for error in current_errors)
        if len(bundles) != len(plan.get("targets", [])):
            errors.append(
                f"{trajectory_id}: complete states {len(bundles)}/{len(plan.get('targets', []))}"
            )
        all_bundles.extend(bundles)
        for target, bundle in bundles:
            task_evidence, task_errors = validate_task_semantics(bundle)
            errors.extend(
                f"{trajectory_id}/{target.get('state_id')}: {error}"
                for error in task_errors
            )
            task_semantics_evidence.append(
                {
                    "trajectory_id": trajectory_id,
                    "state_id": target.get("state_id"),
                    **task_evidence,
                }
            )
        expected_effect_records += len(plan.get("targets", [])) * 2
        all_evidence.extend(
            {"trajectory_id": trajectory_id, **row} for row in evidence
        )
        trajectory_rows.append(
            {
                "trajectory_id": trajectory_id,
                "plan": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "results_root": str(results_root),
                "capture_audit": capture_evidence,
                "complete_states": len(bundles),
                "expected_states": len(plan.get("targets", [])),
            }
        )

    construction_valid = not errors and len(all_bundles) == 96
    endpoint_results: dict[str, Any] = {}
    for endpoint_name, (endpoint_class, extractor) in endpoint_extractors().items():
        records, availability, unavailable = collect_endpoint(
            all_bundles, extractor
        )
        endpoint_complete = construction_valid and len(records) == expected_effect_records
        if endpoint_complete:
            estimate = estimate_scalar_population(
                records,
                required_phases=REQUIRED_PHASES,
                min_confirmation_trajectories=8,
            )
            endpoint_results[endpoint_name] = complete_endpoint_payload(
                endpoint_class, availability, unavailable, estimate
            )
        else:
            endpoint_results[endpoint_name] = {
                "status": "UNAVAILABLE_FULL_FROZEN_CALIBRATION",
                "endpoint_class": endpoint_class,
                "availability": dict(availability),
                "unavailable_records": unavailable,
                "missing_effect_records": expected_effect_records - len(records),
                "confirmation_design_input_only": False,
                "population_B_claim_allowed": False,
                "reason": "no complete-case deletion; all four frozen trajectories and every endpoint state are required",
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": construction_valid,
        "verdict": (
            "VALID_COMPLETE_FOUR_TRAJECTORY_CALIBRATION"
            if construction_valid
            else "INVALID_INCOMPLETE_FROZEN_CALIBRATION"
        ),
        "construction": {
            "errors": errors,
            "trajectories": len({
                bundle["paired_effect_records"][0]["identity"]["trajectory_id"]
                for _, bundle in all_bundles
            }),
            "states": len(all_bundles),
            "expected_states": 96,
            "top_level_df": 3 if construction_valid else None,
            "weighting": "equal trajectory; equal phase within trajectory; equal state within phase",
            "clip_rate_within_state": "normalize by that state's recorded completion-token clip-decision positions before state averaging",
            "clip_rate_is_eligibility_conditioned": False,
            "eligibility_conditioned_clip_rate_location": "separate reference-anchored boundary-conditional family",
            "clip_rate_is_exposure_pooled": False,
            "runtime_variability_scope": {
                "conditioned_fixed_observed_realization_fields": [
                    "compiler_config_digest",
                    "graph_family_digest"
                ],
                "unobserved_realization_sources_not_separately_identified": [
                    "generated-kernel identity",
                    "autotuning variant identity"
                ]
            },
            "task_endpoint_randomness_scope": {
                "T1a_bank_sampling_variance": "UNIDENTIFIED_ONE_FROZEN_BANK_PER_STATE",
                "T1a_H_includes": "state variation plus state-adaptive frozen-bank content variation",
                "T1b_scope": "fixed correct-answer-bank functional"
            },
        },
        "trajectory_inputs": trajectory_rows,
        "analysis_code": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in ANALYSIS_CODE_PATHS.items()
        },
        "endpoints": endpoint_results,
        "state_evidence": all_evidence,
        "task_semantics_evidence": task_semantics_evidence,
        "purpose": "calibration scale/H/N and prospective confirmation precision only",
        "population_B_claim_allowed": False,
        "nonclaims": [
            "96 states do not create 96 top-level independent samples",
            "four calibration trajectories do not satisfy the frozen minimum-eight confirmation gate",
            "U2 delta L2 remains a magnitude profile, not the signed U2 mean field",
            "event disagreement and one-sided transition rates remain profiles, not signed B",
            "calibration intervals are design inputs, not confirmatory verdicts",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"verdict": payload["verdict"], "construction": payload["construction"]},
            indent=2,
        )
    )
    if not construction_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
