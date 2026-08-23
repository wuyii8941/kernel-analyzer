import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_completion_audit_does_not_claim_campaign_completion():
    payload = json.loads(
        (ROOT / "results/property/sample_completion_v1/completion_audit.json").read_text()
    )
    assert payload["status"] == "INCOMPLETE_GPU_CAMPAIGN_NOT_RUN"
    assert payload["scientific_results_written"] is False
    assert "uniform_cases_20" in payload["pending_items"]
    assert "heldout_validation_complete" in payload["pending_items"]
