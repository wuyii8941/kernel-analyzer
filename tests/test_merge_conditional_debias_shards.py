from scripts.merge_conditional_debias_shards import merge


def _shard(start):
    roles = {
        "repair_local_residual": {"status": "CONDITIONAL_CENTERED"},
        "candidate_gradient_effect_removed": {"status": "CONDITIONAL_BIAS"},
    }
    result = {
        "status": "COMPLETE_CONDITIONAL_DEBIAS_ENGINEERING",
        "candidate_id": "mamba",
        "architecture": "mamba",
        "carrier_parameter": "weight",
        "arms": ["JOINT"],
        "conditional_debias_enabled": True,
        "bindings": {
            "source_line_sha256": "source",
            "state_start": start,
            "state_count": 4,
        },
        "result_sha256": f"shard-{start}",
        "states": [],
    }
    for index in range(start, start + 4):
        result["states"].append({
            "state_id": f"state-{index:02d}",
            "arms": {"JOINT": {"conditional_debias": {"layers": roles}}},
        })
    return result


def test_merge_keeps_conditions_separate_and_never_builds_global_direction():
    result = merge([_shard(i) for i in (0, 4, 8, 12)])
    assert result["status"] == "COMPLETE_CONDITIONAL_DEBIAS_CONFIRMATION"
    assert len(result["states"]) == 16
    assert result["direction"] == "NOT_COMPUTED_NOT_A_CONDITIONAL_GATE"
    assert result["conditional_debias_summary"]["JOINT"]["global_direction_required"] is False
