"""Failure-aware summaries for paired training experiments.

The statistical unit is one declared training stream. Retry records are
provenance for that unit; they never increase the sample size.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping


TERMINAL_NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


def _mean_finite_endpoint(record: Mapping[str, Any], endpoint: str,
                          expected_values: int) -> float:
    if record.get("status") != "COMPLETE":
        raise ValueError("a finite endpoint requires a COMPLETE run")
    values = record.get("evaluation_loss_by_step", {}).get(endpoint)
    if not isinstance(values, list) or len(values) != expected_values:
        raise ValueError("the final evaluation endpoint is incomplete")
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("the final evaluation endpoint is nonfinite")
    return math.fsum(values) / len(values)


def _interval(values: list[float], probability: float) -> list[float]:
    import scipy.stats

    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite paired units are required")
    center = math.fsum(values) / len(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        return [center, center]
    critical = float(scipy.stats.t.ppf((1.0 + probability) / 2.0, len(values) - 1))
    half = critical * math.sqrt(variance / len(values))
    return [center - half, center + half]


def _task_status(run: Mapping[str, Any] | None,
                 attempts: list[Mapping[str, Any]], endpoint: str,
                 expected_values: int) -> dict[str, Any]:
    attempt_counts = Counter(str(row.get("status", "UNKNOWN")) for row in attempts)
    if run is not None:
        try:
            final_loss = _mean_finite_endpoint(run, endpoint, expected_values)
        except ValueError as error:
            return {
                "status": "INVALID_COMPLETE_RECORD",
                "error": str(error),
                "attempt_counts": dict(attempt_counts),
            }
        return {
            "status": "COMPLETE_FINITE",
            "final_mean_evaluation_loss": final_loss,
            "attempt_counts": dict(attempt_counts),
        }
    numerical = [row for row in attempts if row.get("status") == "NONFINITE_TRAINING_FAILURE"]
    if numerical:
        steps = [int(row["failure_step"]) for row in numerical if row.get("failure_step") is not None]
        return {
            "status": TERMINAL_NUMERICAL_FAILURE,
            "failure_step": min(steps) if steps else None,
            "attempt_counts": dict(attempt_counts),
        }
    if attempts:
        return {
            "status": "EXECUTION_FAILURE",
            "attempt_counts": dict(attempt_counts),
            "latest_error": str(attempts[-1].get("error", "unknown execution failure")),
        }
    return {"status": "NOT_COMPLETED", "attempt_counts": {}}


def summarize_failure_aware_training(
    protocol: Mapping[str, Any],
    new_runs: Iterable[Mapping[str, Any]],
    failure_attempts: Iterable[Mapping[str, Any]],
    historical_runs: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize complete endpoints and training failures without survivor bias."""
    streams = int(protocol["stream_count"])
    endpoint = str(protocol["steps"])
    expected_values = int(protocol["evaluation_states"])
    new_conditions = tuple(protocol["new_conditions"])
    historical_conditions = tuple(protocol["historical_conditions"])
    conditions = (*historical_conditions, *new_conditions)

    run_index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in new_runs:
        key = (int(row["stream"]), str(row["condition"]))
        if key in run_index:
            raise ValueError(f"duplicate completed run for {key}")
        run_index[key] = row
    failures_by_task: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in failure_attempts:
        key = (int(row["stream"]), str(row["condition"]))
        failures_by_task.setdefault(key, []).append(row)

    tasks: dict[tuple[int, str], dict[str, Any]] = {}
    for stream in range(streams):
        for condition in conditions:
            if condition in historical_conditions:
                run = historical_runs.get((stream, condition))
                tasks[(stream, condition)] = _task_status(
                    run, [], endpoint, expected_values,
                )
            else:
                tasks[(stream, condition)] = _task_status(
                    run_index.get((stream, condition)),
                    failures_by_task.get((stream, condition), []), endpoint, expected_values,
                )

    condition_counts = {
        condition: dict(Counter(tasks[(stream, condition)]["status"]
                                for stream in range(streams)))
        for condition in conditions
    }
    contrast_definitions = {
        "KEY_ONLY_MINUS_ON": ("KEY_ONLY", "ON", "EQUIVALENCE"),
        "OFF_MINUS_REST_ONLY": ("OFF", "REST_ONLY", "EQUIVALENCE"),
        "COORDINATE_ROLL_MINUS_ON": ("COORDINATE_ROLL", "ON", "WORSENING"),
        "ONE_EXTRA_STEP_LAG_MINUS_ON": ("ONE_EXTRA_STEP_LAG", "ON", "WORSENING"),
    }
    margin = float(protocol["loss_margin"])
    probability = float(protocol["simultaneous_interval_level"])
    contrasts = {}
    for name, (left, right, objective) in contrast_definitions.items():
        values = []
        complete_streams = []
        unavailable = []
        for stream in range(streams):
            left_row, right_row = tasks[(stream, left)], tasks[(stream, right)]
            if left_row["status"] == right_row["status"] == "COMPLETE_FINITE":
                values.append(left_row["final_mean_evaluation_loss"] -
                              right_row["final_mean_evaluation_loss"])
                complete_streams.append(stream)
            else:
                unavailable.append({"stream": stream, "left_status": left_row["status"],
                                    "right_status": right_row["status"]})
        numerical_failure = any(
            row["left_status"] == TERMINAL_NUMERICAL_FAILURE or
            row["right_status"] == TERMINAL_NUMERICAL_FAILURE
            for row in unavailable
        )
        result: dict[str, Any] = {
            "left_condition": left,
            "right_condition": right,
            "objective": objective,
            "paired_finite_count": len(values),
            "paired_finite_streams": complete_streams,
            "unavailable_pairs": unavailable,
            "final_loss_defined_for_full_frozen_set": not unavailable,
        }
        if len(values) >= 2:
            center = math.fsum(values) / len(values)
            sample_variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
            result["complete_case_description"] = {
                "conditioning": "BOTH_CONDITIONS_COMPLETED_WITH_FINITE_ENDPOINT",
                "paired_values": values,
                "mean": center,
                "sample_variance": sample_variance,
                "zero_sample_variance_does_not_prove_zero_population_variance": True,
                "interval": _interval(values, probability),
                "interval_level": probability,
                "is_unconditional_primary_result": not unavailable,
            }
        if unavailable:
            result["decision"] = (
                "NOT_ASSESSED_DUE_TO_NUMERICAL_FAILURE" if numerical_failure
                else "NOT_ASSESSED_INCOMPLETE_OR_EXECUTION_FAILURE"
            )
        else:
            interval = result["complete_case_description"]["interval"]
            if objective == "EQUIVALENCE":
                result["decision"] = "EQUIVALENT" if (
                    interval[0] > -margin and interval[1] < margin
                ) else "NOT_ESTABLISHED"
            else:
                result["decision"] = "MATERIAL_WORSENING" if interval[0] > margin else "NOT_ESTABLISHED"
        contrasts[name] = result

    new_task_statuses = [tasks[(stream, condition)]["status"]
                         for stream in range(streams) for condition in new_conditions]
    unresolved = {"NOT_COMPLETED", "EXECUTION_FAILURE", "INVALID_COMPLETE_RECORD"}
    overall_status = (
        "PARTIAL" if any(status in unresolved for status in new_task_statuses)
        else "COMPLETE_WITH_NUMERICAL_FAILURES"
        if any(status == TERMINAL_NUMERICAL_FAILURE for status in new_task_statuses)
        else "COMPLETE"
    )
    return {
        "schema": "structured-residual-training-failure-aware-summary-v2",
        "status": overall_status,
        "stream_count": streams,
        "statistical_unit": "ONE_PREDECLARED_TRAINING_STREAM",
        "retry_attempts_increase_sample_size": False,
        "condition_outcomes": condition_counts,
        "tasks": [
            {"stream": stream, "condition": condition, **tasks[(stream, condition)]}
            for stream in range(streams) for condition in conditions
        ],
        "contrasts": contrasts,
        "failure_endpoint_inference": "DESCRIPTIVE_POST_HOC_FOR_THIS_PROTOCOL",
        "data_use": protocol["data_use"],
        "scope": protocol["not_claimed"],
    }
