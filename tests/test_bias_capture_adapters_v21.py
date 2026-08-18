import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_property_feasibility_is_case_by_property_and_has_no_verdict():
    payload = json.loads((ROOT / "results/property/bias_formation_v2_1/property_feasibility.json").read_text())
    assert payload["schema"] == "kernel-analyzer-bias-property-feasibility-v2_1"
    assert payload["v2_gpu_measurements"] == 0
    assert len(payload["cases"]) == 9
    for case in payload["cases"]:
        assert set(case["properties"]) == set(payload["property_ids"])
        assert "verdict" not in json.dumps(case).lower()


def test_capture_preflights_never_start_gpu_or_emit_scientific_verdict():
    pilot = ROOT / "results/property/bias_formation_v2_1/pilot"
    files = list(pilot.glob("*.preflight.json"))
    assert len(files) == 4
    for path in files:
        payload = json.loads(path.read_text())
        assert payload["gpu_execution_started"] is False
        assert payload["scientific_verdict"] is False
        assert payload["dry_run_states"] == 2
