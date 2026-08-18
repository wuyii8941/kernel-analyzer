import json
import csv
from pathlib import Path

from scripts.run_bias_formation_synthetic_controls import run


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_controls_pass_without_natural_case_evidence():
    result = run()
    assert result["status"] == "PASS"
    assert result["scientific_natural_case_evidence"] is False
    assert all(row["pass"] for row in result["controls"])


def test_bias_formation_map_records_partial_natural_measurement():
    protocol = json.loads((ROOT / "results/property/bias_formation/protocol.json").read_text())
    matrix = json.loads((ROOT / "results/property/bias_formation/formation_matrix.json").read_text())
    assert protocol["protocol_id"] == "bias_formation_map_v1"
    assert protocol["gpu_campaign_started"] is False
    assert matrix["status"] == "PARTIAL_NATURAL_FORMATION_MEASUREMENT"
    phi = next(row for row in matrix["cases"] if row["case_id"] == "phi4_lm_head_dx_seq64")
    assert phi["local"] == "CENTERED"
    assert phi["parameter_gradient"] == "BIASED"


def test_population_keeps_full_denominator_without_formation_labels():
    path = ROOT / "results/property/bias_formation/bias_population.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1574
    assert sum(row["population_kind"] == "ENDPOINT_UNIT" for row in rows) == 1562
    assert sum(row["population_kind"] == "KNOWN_STRICT_CASE" for row in rows) == 12
    assert len({row["population_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["formation_local"] == "PENDING"
        assert row["formation_parameter_gradient"] == "PENDING"
        assert row["formation_effective_update"] == "PENDING"
        assert row["formation_label_source"] == "NOT_MEASURED"
