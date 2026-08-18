import pytest

from scripts.build_bias_formation_certificate import build


def _common(value="same"):
    return {
        "candidate_weights_digest": value,
        "repair_weights_digest": value,
        "candidate_optimizer_digest": value,
        "repair_optimizer_digest": value,
        "candidate_input_digest": value,
        "repair_input_digest": value,
        "candidate_rng_digest": value,
        "repair_rng_digest": value,
        "candidate_scheduler_digest": value,
        "repair_scheduler_digest": value,
        "candidate_loss_scaler_digest": value,
        "repair_loss_scaler_digest": value,
    }


def _payload(with_update=True):
    calibration = [f"c{i}" for i in range(16)]
    confirmation = [f"e{i}" for i in range(16)]
    rows = []
    for state_id in calibration + confirmation:
        row = {
            "state_id": state_id,
            "partition": "calibration" if state_id.startswith("c") else "confirmation",
            "common_state_certificate": _common(),
            "local_endpoint": [1.0, 0.0],
            "parameter_gradient": [1.0, 0.0],
        }
        if with_update:
            row["effective_update"] = [1.0, 0.0]
        rows.append(row)
    return {
        "schema": "synthetic-v2_1-input",
        "case_id": "synthetic",
        "state_split": {
            "calibration_state_ids": calibration,
            "confirmation_state_ids": confirmation,
        },
        "policy": {"min_states": 16, "bootstrap_samples": 2000},
        "rows": rows,
    }


def test_builder_uses_v21_and_reports_first_confirmed_stage():
    result = build(_payload())
    assert result["schema"] == "kernel-analyzer-bias-formation-certificate-v2_1"
    assert result["status"] == "COMPLETE"
    assert result["first_confirmed_bias_stage"] == "LOCAL_ENDPOINT"
    assert result["formation_label_source"] == "v2_1_open_loop_population_certificate"
    assert result["trajectory_drift_in_formation"] is False


def test_builder_missing_layer_fails_closed():
    result = build(_payload(with_update=False))
    assert result["status"] == "UNRESOLVED_MISSING_LAYER"
    assert result["first_confirmed_bias_stage"] is None


def test_builder_rejects_old_single_digest_and_verdict_leakage():
    payload = _payload()
    payload["rows"][0]["common_state_digest"] = "same"
    with pytest.raises(ValueError, match="component-wise"):
        build(payload)
    payload = _payload()
    payload["historical_t4_verdict"] = "PASS"
    with pytest.raises(ValueError, match="historical/verdict"):
        build(payload)
