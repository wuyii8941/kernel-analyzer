from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_complete_coordinate_endpoint_audit_is_scoped_and_nonempty() -> None:
    report = json.loads(
        (
            ROOT
            / "results/property/root_cause_closure_v1/full_coordinate_endpoint_mean_bias_v1.json"
        ).read_text()
    )
    assert report["status"] == "COMPLETE_RETROSPECTIVE_ENDPOINT_AUDIT"
    assert report["natural_training_population_claim"] is False
    assert report["root_cause_claim"] is False
    assert report["familywise_error_controlled"] is False
    summary = report["summary"]
    assert summary["complete_coordinate_endpoint_rows"] >= 1000
    assert summary["finite_bank_mean_vector_nonzero"] == summary["complete_coordinate_endpoint_rows"]
    assert summary["finite_bank_mean_vector_nonzero_quality_valid"] == (
        summary["complete_coordinate_endpoint_rows"] - summary["quality_gate_failed"]
    )
    assert summary["projected_mean_supported"] > 0
    assert summary["quality_gate_failed"] > 0
    assert (
        summary["projected_mean_supported"]
        + summary["projected_mean_not_confirmed"]
        + summary["direction_not_identifiable"]
        + summary["quality_gate_failed"]
        == summary["complete_coordinate_endpoint_rows"]
    )
    assert len(report["rows"]) == summary["complete_coordinate_endpoint_rows"]
    assert all(
        row["scope"] == "RETROSPECTIVE_IID_WITH_REPLACEMENT_FROM_DECLARED_COVERAGE_BANK"
        for row in report["rows"]
    )
