import json

from kernel_analyzer.population_direction import population_positive_direction_prevalence
from scripts.verify_adamw8bit_error_compensation_probe import verify


def test_compensation_verifier_recomputes_result(tmp_path):
    (tmp_path / "units").mkdir()
    protocol = {
        "unit_count": 16, "null_improvement_probability": 0.5,
        "alpha": 0.05, "source_sha256": {},
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    values = []
    for index in range(16):
        row = {
            "default_minus_compensated_rms": 0.1,
            "default_write_rms": 0.2,
            "compensated_write_rms": 0.1,
            "compensated_state_storage_bytes": 10,
        }
        values.append(0.1)
        (tmp_path / "units" / f"unit-{index:03d}.json").write_text(json.dumps(row))
    result = {
        "primary_endpoint": population_positive_direction_prevalence(values),
        "prediction_result": "CONFIRMED",
        "mean_write_rms": {"default_block256": 0.2, "compensated_block256": 0.1},
        "state_storage_bytes": [10],
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert verify(tmp_path)["status"] == "VERIFIED"
