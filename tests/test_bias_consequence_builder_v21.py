import pytest

from scripts.build_bias_consequence_certificate_v21 import build


def arm(value):
    return {"vector": [value, 0.0], "signed_value": value}


def test_builder_requires_four_arms_and_computes_recurrence():
    result = build({
        "case_id": "toy",
        "step_ids": ["0"],
        "rows": [{
            "step_id": "0",
            "candidate_at_candidate_state": arm(2.0),
            "repair_at_candidate_state": arm(1.0),
            "candidate_at_repair_state": arm(2.0),
            "repair_at_repair_state": arm(1.0),
            "drift_before": arm(0.0),
            "drift_after": arm(1.0),
        }],
    })
    assert result["status"] == "COMPLETE"
    assert result["four_counterfactual_arms_required"] is True
    assert result["global_recurrence_residual"]["norm"] == 0.0


def test_builder_rejects_caller_supplied_recurrence_and_formation_labels():
    row = {
        "step_id": "0",
        "candidate_at_candidate_state": arm(1.0),
        "repair_at_candidate_state": arm(0.0),
        "candidate_at_repair_state": arm(1.0),
        "repair_at_repair_state": arm(0.0),
        "drift_before": arm(0.0),
        "drift_after": arm(1.0),
        "recurrence_residual": 0.0,
    }
    with pytest.raises(ValueError):
        build({"case_id": "toy", "step_ids": ["0"], "rows": [row]})
    with pytest.raises(ValueError):
        build({
            "case_id": "toy", "step_ids": ["0"], "rows": [],
            "formation_point": "CENTERED",
        })
