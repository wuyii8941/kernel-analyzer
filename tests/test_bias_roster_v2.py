import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bound_roster_has_no_placeholder_states_and_marks_ineligible_cases():
    roster = json.loads((ROOT / "results/property/bias_formation_v2/roster_bound.json").read_text())
    feasibility = json.loads((ROOT / "results/property/bias_formation_v2/feasibility_report.json").read_text())
    assert roster["schema"] == "kernel-analyzer-bias-property-roster-v2"
    assert feasibility["gpu_campaign_started"] is False
    for case in roster["cases"]:
        assert not any(str(state).startswith("calibration_") for state in case["state_ids"])
    silu = next(case for case in feasibility["cases"] if case["case_id"] == "qwen3vl_silu_seq160")
    assert silu["feasibility"] == "INELIGIBLE_WITH_REASON"
