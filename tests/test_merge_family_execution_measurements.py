import pytest

from scripts.merge_family_execution_measurements import merge


def inventory():
    return {"records": [{
        "release": "/data/releases/model_seq64_r1",
        "task_id": "backward:1:out_ptr0",
        "runtime_measurement_status": "NOT_ASSESSED_BY_THIS_INVENTORY",
        "reference_candidates": [],
    }]}


def audit(complete=True):
    record = {
        "task_id": "backward:1:out_ptr0",
        "status": "VERIFIED",
        "case_id": "case",
        "raw_artifact": "raw.json",
        "raw_sha256": "digest",
        "analysis": {"claim_scope": "FIXED_SUITE_UPDATE"},
    }
    return {
        "schema": "numerical-execution-manifest-audit-v1",
        "families": [{
            "name": "family",
            "eligible_measurement_complete": complete,
            "counts": {"VERIFIED": 1},
            "records": [record],
        }],
    }


def mapping():
    return [{
        "audit_family": "family",
        "release_name": "model_seq64_r1",
        "reference_family": "SILU_BACKWARD",
    }]


def test_merge_attaches_measurement_without_claiming_bias():
    result = merge(inventory(), audit(), mapping())
    row = result["records"][0]
    assert row["runtime_measurement_status"] == "VERIFIED"
    assert row["reference_candidates"][0]["family"] == "SILU_BACKWARD"
    assert result["selected_plan_measurement_merge"]["updated_positions"] == 1
    assert "bias" in result["selected_plan_measurement_merge"]["scope"]


def test_merge_fails_closed_for_incomplete_or_missing_position():
    with pytest.raises(ValueError, match="not complete"):
        merge(inventory(), audit(complete=False), mapping())
    bad = mapping()
    bad[0]["release_name"] = "other"
    with pytest.raises(ValueError, match="does not identify"):
        merge(inventory(), audit(), bad)
