import json
import sys

from scripts import recompute_training_numerical_report as recompute


def test_fixed_suite_profile_without_population_intervals_uses_original_energy(
    tmp_path, monkeypatch,
):
    raw = tmp_path / "raw.json"
    output = tmp_path / "result.json"
    rows = [
        {
            "effect_energy": 0.04,
            "repair_energy": 1.0,
            "effect_repair_inner_product": 0.0,
            "nonzero_effect_coordinates": 1,
        }
        for _ in range(32)
    ]
    raw.write_text(json.dumps({
        "case_id": "fixed-suite",
        "primary_update_endpoint": "PARAMETER_WRITE",
        "original_coordinate_statistics": {"PARAMETER_WRITE": rows},
        "stages": {"PARAMETER_WRITE": {
            "COUNT_SKETCH_V3_FLOAT64_SEED_TEST": {
                "profile": {
                    "status": "DESCRIPTIVE_FIXED_SUITE_ONLY",
                    "population_inference": None,
                    "suite": {},
                }
            }
        }},
    }))
    monkeypatch.setattr(sys, "argv", [
        "recompute", str(raw), str(output), "--endpoint", "PARAMETER_WRITE",
    ])
    recompute.main()
    result = json.loads(output.read_text())
    assert result["measurement_status"] == "VALID"
    assert result["equivalence_decision"] == "NON_EQUIVALENT"
    assert result["bias_analysis"]["fixed_suite_total_rms"] == 0.2
    assert result["bias_analysis"]["direction_diagnostic_status"] == (
        "DESCRIPTIVE_FIXED_SUITE_ONLY_NO_POPULATION_INTERVALS"
    )
