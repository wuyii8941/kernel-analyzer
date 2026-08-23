import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/sample_completion_v1"


def test_sample_completion_protocol_is_frozen_and_fail_closed():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["schema"] == "kernel-analyzer-sample-completion-protocol-v1"
    assert protocol["status"] == "FROZEN_BEFORE_NEW_MEASUREMENT"
    assert protocol["screening_schedule"]["final_label"] == 32
    assert protocol["oracle"]["safe_label_forbidden"] is True
    assert protocol["label_policy"]["unresolved_is_retained_in_denominator"] is True


def test_sample_completion_roster_has_eight_base_and_sixteen_search_units():
    roster = json.loads((OUT / "roster.json").read_text())
    assert roster["base_case_count"] == 8
    assert roster["search_unit_count"] == 16
    assert roster["case_count"] == 24
    ids = [row["case_id"] for row in roster["base_cases"] + roster["search_units"]]
    assert len(ids) == len(set(ids))
    assert all(row["scientific_label"] is None for row in roster["base_cases"])
    assert all(row["scientific_label"] is None for row in roster["search_units"])


def test_existing_snapshot_does_not_promote_old_evidence_to_new_uniform_sample():
    snapshot = json.loads((OUT / "existing_evidence_snapshot.json").read_text())
    assert snapshot["models_with_actual_artifacts"] == 10
    assert snapshot["systematic_census_cells"] == 12
    assert snapshot["current_sample_completion_uniform_cases"] == 0
    assert snapshot["current_sample_completion_uniform_controls"] == 0
