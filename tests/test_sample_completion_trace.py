from kernel_analyzer.sample_completion import predictor_view, validate_trace


def _trace():
    rows = [{"l2": 1.0, "vector_digest": f"d{step}"} for step in range(32)]
    return {
        "schema": "kernel-analyzer-sample-completion-trace-v1",
        "case_id": "toy",
        "state_ids": [f"s{step}" for step in range(32)],
        "common_state": {"all_components_equal": True},
        "candidate_repair_bound": True,
        "forward_backward_bound": True,
        "difference_reaches_parameter_gradient": True,
        "stages": {stage: rows for stage in ("operator_output", "parameter_gradient", "optimizer_update", "trajectory", "feedback")},
        "final_label": "SHOULD_NOT_ENTER_PREDICTOR",
        "final_drift": 123.0,
    }


def test_valid_trace_and_predictor_view_are_separate():
    payload = _trace()
    result = validate_trace(payload)
    assert result.valid
    view = predictor_view(payload)
    assert len(view["state_ids"]) == 16
    assert "final_drift" not in view
    assert "final_label" not in view
    assert "trajectory" not in view["stages"]


def test_missing_stage_fails_closed():
    payload = _trace()
    payload["stages"].pop("parameter_gradient")
    result = validate_trace(payload)
    assert not result.valid
    assert "MISSING_STAGE:parameter_gradient" in result.reasons

