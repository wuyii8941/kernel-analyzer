import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_is_not_a_scientific_result():
    report = json.loads(
        (ROOT / "results/property/sample_completion_v1/preflight_report.json").read_text()
    )
    assert report["schema"] == "kernel-analyzer-sample-completion-preflight-v1"
    assert report["scientific_results_written"] is False
    assert len(report["groups"]) == 4
    assert all(row["input_state_count"] == 32 for row in report["groups"])
