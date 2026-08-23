import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_heldout_split_is_frozen_and_has_both_expectations():
    payload = json.loads(
        (ROOT / "results/property/sample_completion_v1/heldout_roster.json").read_text()
    )
    assert payload["status"] == "FROZEN_BEFORE_MEASUREMENT"
    assert payload["scientific_labels_frozen"] is False
    assert len(payload["rows"]) >= 8
    assert len({row["model_group"] for row in payload["rows"]}) >= 2
    assert {row["expectation_bucket"] for row in payload["rows"]} == {
        "ESCALATE_CANDIDATE",
        "NO_ESCALATION_CANDIDATE",
    }
    assert all(row["scientific_label"] is None for row in payload["rows"])
