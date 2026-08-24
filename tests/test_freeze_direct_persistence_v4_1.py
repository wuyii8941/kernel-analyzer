from scripts.freeze_direct_persistence_v4_1 import complete_identity


def test_v4_1_identity_rejects_missing_fields():
    ok, missing = complete_identity({"model": "m"})
    assert not ok
    assert "repair" in missing
    assert "moment_state" in missing


def test_v4_1_identity_accepts_complete_row():
    row = {
        "model": "m",
        "exact_endpoint": {"task_id": "t"},
        "sequence_length": 128,
        "state_order": {"formation": ["a"], "trajectory": ["b"]},
        "parameter_coordinate_digest": "digest",
        "optimizer": {"name": "AdamW"},
        "moment_state": "zero_then_evolved",
        "horizon": 32,
        "repair": {"kind": "observer"},
        "runner_source_digest": "digest",
    }
    assert complete_identity(row) == (True, [])
