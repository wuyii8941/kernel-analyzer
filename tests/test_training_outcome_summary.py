from kernel_analyzer.training_outcome_summary import summarize_failure_aware_training


def _run(stream, condition, value):
    return {
        "stream": stream,
        "condition": condition,
        "status": "COMPLETE",
        "evaluation_loss_by_step": {"10": [value, value]},
    }


def _protocol():
    return {
        "stream_count": 2,
        "steps": 10,
        "evaluation_states": 2,
        "historical_conditions": ["OFF", "ON"],
        "new_conditions": ["KEY_ONLY", "REST_ONLY", "COORDINATE_ROLL", "ONE_EXTRA_STEP_LAG"],
        "loss_margin": 0.01,
        "simultaneous_interval_level": 0.9875,
        "data_use": "SAME_STREAM_ATTRIBUTION",
        "not_claimed": ["UNSEEN_CONFIRMATION"],
    }


def test_retries_do_not_increase_n_and_numerical_failure_blocks_final_loss_decision():
    historical = {}
    runs = []
    for stream in range(2):
        historical[(stream, "OFF")] = _run(stream, "OFF", 2.0)
        historical[(stream, "ON")] = _run(stream, "ON", 1.9)
        runs.extend([
            _run(stream, "KEY_ONLY", 1.9),
            _run(stream, "REST_ONLY", 2.0),
            _run(stream, "ONE_EXTRA_STEP_LAG", 2.1),
        ])
    failures = [
        {"stream": 0, "condition": "COORDINATE_ROLL", "status": "EXECUTION_FAILURE",
         "error": "CUDA unavailable"},
        {"stream": 0, "condition": "COORDINATE_ROLL", "status": "NONFINITE_TRAINING_FAILURE",
         "failure_step": 4},
        {"stream": 1, "condition": "COORDINATE_ROLL", "status": "NONFINITE_TRAINING_FAILURE",
         "failure_step": 7},
    ]
    result = summarize_failure_aware_training(_protocol(), runs, failures, historical)
    assert result["status"] == "COMPLETE_WITH_NUMERICAL_FAILURES"
    assert result["condition_outcomes"]["COORDINATE_ROLL"] == {"NUMERICAL_FAILURE": 2}
    contrast = result["contrasts"]["COORDINATE_ROLL_MINUS_ON"]
    assert contrast["paired_finite_count"] == 0
    assert contrast["decision"] == "NOT_ASSESSED_DUE_TO_NUMERICAL_FAILURE"
    assert result["retry_attempts_increase_sample_size"] is False


def test_complete_run_supersedes_operational_retry_failure():
    historical = {}
    runs = []
    for stream in range(2):
        historical[(stream, "OFF")] = _run(stream, "OFF", 2.0)
        historical[(stream, "ON")] = _run(stream, "ON", 1.9)
        for condition, value in (
            ("KEY_ONLY", 1.9), ("REST_ONLY", 2.0),
            ("COORDINATE_ROLL", 2.1), ("ONE_EXTRA_STEP_LAG", 2.1),
        ):
            runs.append(_run(stream, condition, value))
    failures = [{"stream": 0, "condition": "COORDINATE_ROLL",
                 "status": "EXECUTION_FAILURE", "error": "CUDA unavailable"}]
    result = summarize_failure_aware_training(_protocol(), runs, failures, historical)
    assert result["status"] == "COMPLETE"
    task = next(row for row in result["tasks"]
                if row["stream"] == 0 and row["condition"] == "COORDINATE_ROLL")
    assert task["status"] == "COMPLETE_FINITE"
    assert task["attempt_counts"] == {"EXECUTION_FAILURE": 1}
    assert result["contrasts"]["COORDINATE_ROLL_MINUS_ON"]["paired_finite_count"] == 2
