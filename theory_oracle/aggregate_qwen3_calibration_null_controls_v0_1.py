#!/usr/bin/env python
"""Build within-implementation/evaluator null controls from a complete trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    load_complete_state_bundles,
    load_json,
    sha256_file,
)


SCHEMA_VERSION = "forkcert.qwen3-calibration-null-controls.v0.1"
ANALYSIS_CODE_PATHS = {
    "null_control_aggregator": Path(__file__).resolve(),
    "record_loader": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_calibration_records_v0_1.py",
    "record_validator": ROOT / "theory_oracle" / "bias_oracle_record_v0_2.py",
}
ARMS = ("reference", "candidate")
SCALAR_PATHS = {
    "training_loss": ("propagation_ledgers", "training_loss"),
    "pre_clip_gradient_norm": (
        "propagation_ledgers",
        "pre_clip_gradient_norm",
    ),
    "parameter_update_l2": ("propagation_ledgers", "parameter_update_l2"),
    "T1a_arm_loss": ("T1a_arm_loss", "value"),
    "T1b_arm_nll": ("T1b_arm_nll", "value"),
}
TASK_ENDPOINT_VALUE_FIELDS = {"T1a": "loss", "T1b": "mean_nll"}


def nested_value(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row["outcomes"]
    for key in path:
        value = value[key]
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite scalar at {'.'.join(path)}")
    return parsed


def arm_rows_by_identity(bundle: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in bundle.get("arm_records", []):
        identity = row.get("identity", {})
        arm = identity.get("arm")
        repeat_id = identity.get("repeat_id")
        if arm not in ARMS or repeat_id not in (1, 2):
            raise ValueError("arm records must use reference/candidate and repeat IDs 1/2")
        if repeat_id in rows[arm]:
            raise ValueError(f"duplicate {arm} repeat {repeat_id}")
        rows[arm][repeat_id] = row
    if set(rows) != set(ARMS) or any(set(rows[arm]) != {1, 2} for arm in ARMS):
        raise ValueError("exactly two records are required for each arm")
    return dict(rows)


def task_evaluator_contrasts(task: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    if task.get("valid") is not True:
        raise ValueError("task evaluation is not valid")
    output: dict[str, dict[str, list[float]]] = {}
    for endpoint, value_field in TASK_ENDPOINT_VALUE_FIELDS.items():
        arm_results = task.get(endpoint, {}).get("arm_results", {})
        if set(arm_results) != set(ARMS):
            raise ValueError(f"{endpoint} arm_results must cover both arms")
        endpoint_output: dict[str, list[float]] = {}
        for arm in ARMS:
            by_transition: dict[int, dict[int, float]] = defaultdict(dict)
            for row in arm_results[arm]:
                transition_repeat = row.get("transition_repeat")
                evaluator_repeat = row.get("evaluator_repeat")
                if transition_repeat not in (1, 2) or evaluator_repeat not in (1, 2):
                    raise ValueError(f"{endpoint}/{arm} has invalid repeat identity")
                if evaluator_repeat in by_transition[transition_repeat]:
                    raise ValueError(f"{endpoint}/{arm} has duplicate evaluator repeat")
                value = float(row[value_field])
                if not math.isfinite(value):
                    raise ValueError(f"{endpoint}/{arm} has non-finite evaluator value")
                by_transition[transition_repeat][evaluator_repeat] = value
            if set(by_transition) != {1, 2} or any(
                set(by_transition[transition_repeat]) != {1, 2}
                for transition_repeat in (1, 2)
            ):
                raise ValueError(
                    f"{endpoint}/{arm} requires evaluator repeats 1/2 inside transition repeats 1/2"
                )
            endpoint_output[arm] = [
                by_transition[transition_repeat][2]
                - by_transition[transition_repeat][1]
                for transition_repeat in (1, 2)
            ]
        output[endpoint] = endpoint_output
    return output


def extract_state_controls(
    target: dict[str, Any],
    bundle: dict[str, Any],
    task: dict[str, Any],
    *,
    verify_update_artifacts: bool = True,
) -> dict[str, Any]:
    rows = arm_rows_by_identity(bundle)
    arm_controls: dict[str, Any] = {}
    for arm in ARMS:
        repeat_1 = rows[arm][1]
        repeat_2 = rows[arm][2]
        scalar_contrasts = {
            name: nested_value(repeat_2, path) - nested_value(repeat_1, path)
            for name, path in SCALAR_PATHS.items()
        }
        artifact_1 = repeat_1["outcomes"]["parameter_update_artifact"]
        artifact_2 = repeat_2["outcomes"]["parameter_update_artifact"]
        if verify_update_artifacts:
            for artifact in (artifact_1, artifact_2):
                path = Path(artifact["path"]).resolve()
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    raise ValueError(
                        f"{arm} parameter-update artifact is missing or hash-mismatched"
                    )
        arm_controls[arm] = {
            "scalar_repeat2_minus_repeat1": scalar_contrasts,
            "parameter_update_artifact_sha_equal": artifact_1["sha256"]
            == artifact_2["sha256"],
            "next_state_digests_equal": repeat_1["outcomes"]["next_state_digests"]
            == repeat_2["outcomes"]["next_state_digests"],
            "semantic_events_equal": repeat_1["outcomes"]["semantic_events"]
            == repeat_2["outcomes"]["semantic_events"],
        }
    return {
        "state_id": target["state_id"],
        "phase": target["phase"],
        "arm_controls": arm_controls,
        "evaluator_repeat2_minus_repeat1": task_evaluator_contrasts(task),
    }


def weighted_trajectory_mean(values: list[tuple[str, float]]) -> float:
    by_phase: dict[str, list[float]] = defaultdict(list)
    for phase, value in values:
        by_phase[phase].append(value)
    if set(by_phase) != {"early", "middle", "late"}:
        raise ValueError("null-control summary requires early/middle/late coverage")
    phase_means = [math.fsum(by_phase[phase]) / len(by_phase[phase]) for phase in sorted(by_phase)]
    return math.fsum(phase_means) / len(phase_means)


def scalar_summary(values: list[tuple[str, float]]) -> dict[str, Any]:
    return {
        "state_count": len(values),
        "nonzero_state_count": sum(value != 0.0 for _, value in values),
        "max_absolute_contrast": max(abs(value) for _, value in values),
        "trajectory_weighted_signed_contrast": weighted_trajectory_mean(values),
        "trajectory_weighted_absolute_contrast": weighted_trajectory_mean(
            [(phase, abs(value)) for phase, value in values]
        ),
    }


def summarize_state_controls(states: list[dict[str, Any]]) -> dict[str, Any]:
    arm_values: dict[str, dict[str, list[tuple[str, float]]]] = {
        arm: {name: [] for name in SCALAR_PATHS} for arm in ARMS
    }
    evaluator_values: dict[str, dict[str, list[tuple[str, float]]]] = {
        endpoint: {arm: [] for arm in ARMS}
        for endpoint in TASK_ENDPOINT_VALUE_FIELDS
    }
    exactness: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    for state in states:
        phase = state["phase"]
        for arm in ARMS:
            controls = state["arm_controls"][arm]
            for endpoint, value in controls["scalar_repeat2_minus_repeat1"].items():
                arm_values[arm][endpoint].append((phase, float(value)))
            for field in (
                "parameter_update_artifact_sha_equal",
                "next_state_digests_equal",
                "semantic_events_equal",
            ):
                exactness[arm][field] += int(controls[field] is True)
        for endpoint in TASK_ENDPOINT_VALUE_FIELDS:
            for arm in ARMS:
                evaluator_values[endpoint][arm].extend(
                    (phase, float(value))
                    for value in state["evaluator_repeat2_minus_repeat1"][endpoint][arm]
                )
    return {
        "within_implementation": {
            arm: {
                "scalar_controls": {
                    endpoint: scalar_summary(values)
                    for endpoint, values in arm_values[arm].items()
                },
                "exact_artifact_event_controls": {
                    field: {
                        "equal_states": exactness[arm][field],
                        "state_count": len(states),
                        "all_equal": exactness[arm][field] == len(states),
                    }
                    for field in exactness[arm]
                },
            }
            for arm in ARMS
        },
        "within_evaluator": {
            endpoint: {
                arm: scalar_summary(values)
                for arm, values in evaluator_values[endpoint].items()
            }
            for endpoint in TASK_ENDPOINT_VALUE_FIELDS
        },
    }


def load_verified_task(bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    links = {
        (
            arm["provenance"]["task_evaluation"]["path"],
            arm["provenance"]["task_evaluation"]["sha256"],
        )
        for arm in bundle["arm_records"]
    }
    if len(links) != 1:
        raise ValueError("all arms must link the same task evaluation artifact")
    path_text, expected_sha = next(iter(links))
    path = Path(path_text).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("task evaluation artifact is missing or hash-mismatched")
    return load_json(path), {"path": str(path), "sha256": expected_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    results_root = Path(args.results_root).resolve()
    plan = load_json(plan_path)
    bundles, state_evidence, errors = load_complete_state_bundles(plan, results_root)
    phase_counts = Counter(target["phase"] for target, _ in bundles)
    complete = (
        not errors
        and len(bundles) == len(plan["targets"])
        and phase_counts == Counter(target["phase"] for target in plan["targets"])
    )
    states: list[dict[str, Any]] = []
    task_evidence: list[dict[str, str]] = []
    if complete:
        for target, bundle in bundles:
            try:
                task, evidence = load_verified_task(bundle)
                states.append(
                    extract_state_controls(
                        target,
                        bundle,
                        task,
                        verify_update_artifacts=True,
                    )
                )
                task_evidence.append(evidence)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{target['state_id']}: {exc}")
        complete = not errors and len(states) == len(plan["targets"])

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": complete,
        "verdict": "VALID_COMPLETE_NULL_CONTROL_DESCRIPTION"
        if complete
        else "INVALID_INCOMPLETE_NULL_CONTROLS",
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "trajectory_identity": dict(plan.get("identity", {})),
        "results_root": str(results_root),
        "analysis_provenance": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in ANALYSIS_CODE_PATHS.items()
        },
        "construction": {
            "states": len(states),
            "expected_states": len(plan["targets"]),
            "phase_counts": dict(phase_counts),
            "errors": errors,
            "state_evidence": state_evidence,
            "task_evidence": task_evidence,
        },
        "controls": summarize_state_controls(states) if complete else None,
        "threshold_instantiation_allowed": False,
        "reason": (
            "one complete trajectory supplies measurement-null diagnostics only; threshold values require the frozen four-trajectory calibration/control rule"
            if complete
            else "partial states are not used to redefine the frozen control population"
        ),
        "nonclaims": [
            "two repeats do not establish absence of rare runtime variability",
            "zero observed null contrasts do not supply a positive variance floor",
            "measurement-null controls do not define practical harm or correctness",
            "within-implementation controls do not estimate candidate-reference B",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "construction": payload["construction"]}, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
