import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_base_audit_is_conservative_about_uniform_schema():
    result = json.loads((ROOT / "results/property/sample_completion_v1/base_case_audit.json").read_text())
    assert result["schema"] == "kernel-analyzer-sample-completion-base-case-audit-v1"
    assert len(result["rows"]) == 8
    assert result["uniform_trace_ready_count"] == 0
    assert all(row["scientific_label"] is None for row in result["rows"])
