import json

import pytest

from scripts.build_bias_formation_certificate import build
from scripts.build_bias_transition_matrix import build as build_matrix


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


def test_transition_matrix_is_pending_without_certificates():
    rows = build_matrix()
    assert len(rows) == 4
    assert all(row["local"] == "PENDING" for row in rows)
    assert all(row["mechanism_candidate"] == "PENDING_MEASUREMENT" for row in rows)


def test_transition_matrix_uses_only_v21_certificate(tmp_path):
    payload = _payload()
    certificate_path = tmp_path / "formation.json"
    certificate_path.write_text(json.dumps(build(payload)), encoding="utf-8")
    rows = build_matrix({"liger_fused_ce_t128": certificate_path})
    liger = next(row for row in rows if row["case"] == "liger_fused_ce_t128")
    assert liger["local"] == "LOCAL_BIAS"
    assert liger["gradient"] == "GRADIENT_BIAS"
    assert liger["update"] == "UPDATE_BIAS"
    assert liger["mechanism_candidate"] == "SOURCE_CANDIDATE"
    assert liger["formation_label_source"] == "v2_1_open_loop_certificate"


def test_transition_matrix_rejects_old_or_trajectory_certificate(tmp_path):
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps({"schema": "kernel-analyzer-bias-formation-certificate-v2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="only v2.1"):
        build_matrix({"liger_fused_ce_t128": old_path})
    trajectory_path = tmp_path / "trajectory.json"
    certificate = build(_payload())
    certificate["trajectory_drift_in_formation"] = True
    trajectory_path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory drift"):
        build_matrix({"liger_fused_ce_t128": trajectory_path})
