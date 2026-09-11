import json

from kernel_analyzer.population_direction import population_positive_direction_prevalence
from scripts.verify_adamw8bit_population_direction import verify


def test_verifier_recomputes_direction_endpoint(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    protocol = {
        "unit_count": 16, "null_positive_probability": 0.5, "alpha": 0.05,
        "source_sha256": {},
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    values = []
    for index in range(16):
        row = {
            "effect_repair_inner_product": 1.0,
            "effect_energy": 1.0,
            "repair_energy": 1.0,
            "repeatability": "EXACT",
        }
        values.append(1.0)
        (units / f"unit-{index:03d}.json").write_text(json.dumps(row))
    result = {
        "primary_endpoint": population_positive_direction_prevalence(values),
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert verify(tmp_path)["status"] == "VERIFIED"


def test_verifier_reports_unseen_subset_after_cross_experiment_deduplication(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    histories = [[index] for index in range(16)]
    protocol = {
        "unit_count": 16, "null_positive_probability": 0.5, "alpha": 0.05,
        "source_sha256": {}, "unit_population_indices": histories,
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    values = []
    for index, history in enumerate(histories):
        row = {
            "effect_repair_inner_product": 1.0,
            "effect_energy": 1.0,
            "repair_energy": 1.0,
            "repeatability": "EXACT",
            "population_indices": history,
        }
        values.append(1.0)
        (units / f"unit-{index:03d}.json").write_text(json.dumps(row))
    (tmp_path / "result.json").write_text(json.dumps({
        "primary_endpoint": population_positive_direction_prevalence(values),
    }))
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps({"unit_population_indices": histories[:8]}))
    audit = verify(tmp_path, exclude_protocols=(earlier,))
    subset = audit["cross_experiment_independence_audit"]
    assert audit["status"] == "VERIFIED"
    assert subset["overlapping_unit_count"] == 8
    assert subset["unseen_unit_count"] == 8
    assert subset["unseen_subset_endpoint"]["positive_count"] == 8
