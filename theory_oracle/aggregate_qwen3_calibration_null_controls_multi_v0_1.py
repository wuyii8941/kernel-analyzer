#!/usr/bin/env python
"""Aggregate four complete trajectory-level measurement-null descriptions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_null_controls_v0_1 import (
    ANALYSIS_CODE_PATHS as SINGLE_ANALYSIS_CODE_PATHS,
    ARMS,
    SCALAR_PATHS,
    SCHEMA_VERSION as SINGLE_SCHEMA_VERSION,
    TASK_ENDPOINT_VALUE_FIELDS,
)
from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    load_json,
    sha256_file,
)


SCHEMA_VERSION = "forkcert.qwen3-calibration-null-controls-multi.v0.1"
EXPECTED_TRAJECTORIES = tuple(f"calibration-{index}" for index in range(4))
ANALYSIS_CODE_PATHS = {
    "multi_null_control_aggregator": Path(__file__).resolve(),
    **SINGLE_ANALYSIS_CODE_PATHS,
}


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two trajectory values are required")
    mean = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def metric_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signed = [float(row["trajectory_weighted_signed_contrast"]) for row in rows]
    absolute = [float(row["trajectory_weighted_absolute_contrast"]) for row in rows]
    state_maxima = [float(row["max_absolute_contrast"]) for row in rows]
    if any(
        not math.isfinite(value)
        for value in signed + absolute + state_maxima
    ):
        raise ValueError("null-control metric contains non-finite values")
    return {
        "trajectory_signed_contrasts": signed,
        "trajectory_absolute_contrasts": absolute,
        "mean_trajectory_signed_contrast": math.fsum(signed) / len(signed),
        "mean_trajectory_absolute_contrast": math.fsum(absolute) / len(absolute),
        "trajectory_signed_contrast_sd": sample_sd(signed),
        "maximum_observed_state_absolute_contrast": max(state_maxima),
        "nonzero_state_count": sum(int(row["nonzero_state_count"]) for row in rows),
        "observed_state_count": sum(int(row["state_count"]) for row in rows),
        "all_observed_contrasts_zero": all(value == 0.0 for value in state_maxima),
    }


def aggregate_controls(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    within_implementation: dict[str, Any] = {}
    for arm in ARMS:
        scalar_controls = {
            endpoint: metric_aggregate(
                [
                    summary["controls"]["within_implementation"][arm][
                        "scalar_controls"
                    ][endpoint]
                    for summary in summaries
                ]
            )
            for endpoint in SCALAR_PATHS
        }
        exact_controls: dict[str, Any] = {}
        fields = (
            "parameter_update_artifact_sha_equal",
            "next_state_digests_equal",
            "semantic_events_equal",
        )
        for field in fields:
            rows = [
                summary["controls"]["within_implementation"][arm][
                    "exact_artifact_event_controls"
                ][field]
                for summary in summaries
            ]
            exact_controls[field] = {
                "equal_states": sum(int(row["equal_states"]) for row in rows),
                "state_count": sum(int(row["state_count"]) for row in rows),
                "all_equal": all(row["all_equal"] is True for row in rows),
            }
        within_implementation[arm] = {
            "scalar_controls": scalar_controls,
            "exact_artifact_event_controls": exact_controls,
        }

    within_evaluator = {
        endpoint: {
            arm: metric_aggregate(
                [
                    summary["controls"]["within_evaluator"][endpoint][arm]
                    for summary in summaries
                ]
            )
            for arm in ARMS
        }
        for endpoint in TASK_ENDPOINT_VALUE_FIELDS
    }
    return {
        "within_implementation": within_implementation,
        "within_evaluator": within_evaluator,
    }


def validate_summary(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        row = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: unreadable summary: {exc}"]
    if row.get("schema_version") != SINGLE_SCHEMA_VERSION:
        errors.append(f"{path}: unsupported single-trajectory schema")
    if row.get("valid") is not True or row.get("verdict") != "VALID_COMPLETE_NULL_CONTROL_DESCRIPTION":
        errors.append(f"{path}: summary is not complete and valid")
    identity = row.get("trajectory_identity", {})
    trajectory_id = identity.get("trajectory_id")
    if trajectory_id not in EXPECTED_TRAJECTORIES:
        errors.append(f"{path}: unexpected trajectory identity")
    plan = row.get("plan", {})
    plan_path = Path(plan.get("path", "")).resolve()
    if not plan_path.is_file() or sha256_file(plan_path) != plan.get("sha256"):
        errors.append(f"{path}: plan artifact is missing or hash-mismatched")
    else:
        plan_identity = load_json(plan_path).get("identity", {})
        if plan_identity != identity:
            errors.append(f"{path}: trajectory identity does not match frozen plan")
    provenance = row.get("analysis_provenance", {})
    for role, expected_path in SINGLE_ANALYSIS_CODE_PATHS.items():
        observed = provenance.get(role, {})
        if (
            Path(observed.get("path", "")).resolve() != expected_path
            or not expected_path.is_file()
            or observed.get("sha256") != sha256_file(expected_path)
        ):
            errors.append(f"{path}: analysis provenance mismatch for {role}")
    if row.get("controls") is None:
        errors.append(f"{path}: controls are missing")
    return row, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-summary", action="append", required=True
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in args.trajectory_summary]
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    for path in paths:
        row, current_errors = validate_summary(path)
        errors.extend(current_errors)
        if row is not None:
            summaries.append(row)
            evidence.append({"path": str(path), "sha256": sha256_file(path)})
    identities = [
        summary.get("trajectory_identity", {}).get("trajectory_id")
        for summary in summaries
    ]
    if len(paths) != 4 or sorted(identities) != sorted(EXPECTED_TRAJECTORIES):
        errors.append("exactly calibration-0..3 summaries are required once each")

    valid = not errors and len(summaries) == 4
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_COMPLETE_FOUR_TRAJECTORY_NULL_CONTROL_DESCRIPTION"
        if valid
        else "INVALID_NULL_CONTROL_CONSTRUCTION",
        "trajectory_evidence": evidence,
        "trajectory_ids": identities,
        "analysis_provenance": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in ANALYSIS_CODE_PATHS.items()
        },
        "construction_errors": errors,
        "controls": aggregate_controls(summaries) if valid else None,
        "threshold_source_role": "MEASUREMENT_NULL_DESCRIPTION_ONLY",
        "threshold_instantiation_allowed": False,
        "reason": (
            "observed null envelopes may inform a separately frozen threshold rule, but zero observations do not create a positive half-width or variance floor"
            if valid
            else "invalid or incomplete trajectory summaries cannot define a null envelope"
        ),
        "nonclaims": [
            "four null-control trajectories do not prove runtime variability is absent",
            "null controls do not estimate candidate-reference B",
            "null controls do not define practical harm or correctness",
            "an analytic or external positive floor is still required when all observed contrasts are zero",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "errors": errors}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
