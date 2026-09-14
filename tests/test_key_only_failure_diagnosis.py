import json

from scripts.summarize_key_only_failure_diagnosis import summarize


def test_negative_second_is_detected_before_parameter_nonfinite(tmp_path):
    good = {"nonfinite": 0, "negative": 0, "min": 0, "max_abs": 1}
    row = {"name": "weight", "compensated": True,
           **{k: dict(good) for k in ("parameter_before", "gradient", "effective_second", "next_second", "parameter_after")}}
    row["next_second"]["negative"] = 1
    row["parameter_after"]["nonfinite"] = 1
    (tmp_path / "step_0935.json").write_text(json.dumps({"step": 935, "parameters": [row]}))
    (tmp_path / "terminal.json").write_text(json.dumps({"error": "nonfinite loss at step 936"}))
    result = summarize(tmp_path)
    assert result["first_observed_anomaly_step"] == 935
    assert [e["field"] for e in result["first_observed_anomalies"]] == ["next_second", "parameter_after"]


def test_missing_observation_does_not_claim_clean_training(tmp_path):
    result = summarize(tmp_path)
    assert result["status"] == "PARTIAL"
    assert result["first_observed_anomaly_step"] is None
