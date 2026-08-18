import json

import pytest

from kernel_analyzer.bias import BiasStatus, BiasTrace, BiasTracePolicy
from scripts.run_bias_trace import build


def _trace() -> BiasTrace:
    return BiasTrace(
        "toy",
        ["c0", "c1"],
        ["e0", "e1"],
        BiasTracePolicy(min_states=2, centred_z_abs_max=2.0),
    )


def _add(trace, state_id, partition, local, gradient, update, drift):
    trace.add(
        state_id,
        partition,
        local_endpoint=local,
        parameter_gradient=gradient,
        effective_update=update,
        trajectory_drift=drift,
    )


def test_transition_certificate_reports_first_biased_layer_without_t4():
    trace = _trace()
    for state_id, value in (("c0", 1.0), ("c1", -1.0)):
        _add(trace, state_id, "calibration", value, value, value, value)
    _add(trace, "e0", "evaluation", -1.0, 1.0, 1.0, 1.0)
    _add(trace, "e1", "evaluation", 1.0, 2.0, 2.0, 2.0)
    result = trace.finalize()
    assert result["status"] == "COMPLETE"
    assert result["first_noncentered_layer"] == "PARAMETER_GRADIENT"
    assert result["layers"]["LOCAL_ENDPOINT"]["status"] == BiasStatus.CENTERED.value
    assert result["layers"]["PARAMETER_GRADIENT"]["status"] == BiasStatus.BIASED.value
    assert result["temporal"]["uses_t4_or_final_drift_label"] is False


def test_missing_layer_fails_closed_and_does_not_impute_zero():
    trace = _trace()
    for state_id, partition in (("c0", "calibration"), ("c1", "calibration"),
                                ("e0", "evaluation"), ("e1", "evaluation")):
        trace.add(
            state_id,
            partition,
            local_endpoint=1.0,
            parameter_gradient=None,
            effective_update=1.0,
            trajectory_drift=1.0,
        )
    result = trace.finalize()
    gradient = result["layers"]["PARAMETER_GRADIENT"]
    assert gradient["status"] == BiasStatus.UNRESOLVED.value
    assert gradient["missing_state_ids"] == ["e0", "e1"]
    assert result["status"] == "COMPLETE"


def test_trace_runner_rejects_candidate_or_verdict_leakage():
    payload = {
        "schema": "toy",
        "case_id": "toy",
        "oracle_verdict": "PASS",
        "state_split": {"calibration_state_ids": ["c0", "c1"],
                         "evaluation_state_ids": ["e0", "e1"]},
        "rows": [],
    }
    with pytest.raises(ValueError):
        build(payload)


def test_trace_runner_accepts_normalized_rows_and_is_json_serializable():
    rows = []
    for state_id, partition, value in (
        ("c0", "calibration", 1.0), ("c1", "calibration", 1.0),
        ("e0", "evaluation", 1.0), ("e1", "evaluation", 1.0),
    ):
        rows.append({
            "state_id": state_id,
            "partition": partition,
            "local_endpoint": {"signed_value": value, "norm": 1.0},
            "parameter_gradient": {"signed_value": value, "norm": 1.0},
            "effective_update": {"signed_value": value, "norm": 1.0},
            "trajectory_drift": {"signed_value": value, "norm": 1.0},
        })
    result = build({
        "schema": "kernel-analyzer-normalized-bias-trace-v1",
        "case_id": "toy",
        "state_split": {"calibration_state_ids": ["c0", "c1"],
                         "evaluation_state_ids": ["e0", "e1"]},
        "policy": {"min_states": 2},
        "rows": rows,
    })
    json.dumps(result)
    assert result["candidate_blind"] is True
    assert result["status"] == "COMPLETE"
