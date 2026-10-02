from scripts.build_natural_candidate_frontier import build


def test_signature_candidate_frontier_is_triage_only_and_deduplicated():
    result = build()
    assert result["status"] == "TRIAGE_ONLY_NOT_A_ROOT_CAUSE_COUNT"
    assert result["record_count"] == 31
    assert result["candidate_family_count"] >= 6
    assert result["active_problem_group_count_unchanged"] == 52
    assert result["negative_control_count_unchanged"] == 2

    by_family = {row["candidate_family"]: row for row in result["groups"]}
    assert by_family["layout_transpose_materialization"]["promotion_status"] == (
        "CANDIDATE_ONLY_REGION_SUBSTITUTION"
    )
    embedding = by_family["embedding_dense_backward"]
    assert embedding["next_capture_priority"] == "HIGH"
    assert embedding["recommended_records"][0]["task_id"] == "backward:1885:out_ptr1"
    assert embedding["adapter_plan"] == "NEW_EMBEDDING_BOUNDARY_ADAPTER"
    assert by_family["recurrence_softplus"]["max_fixed_suite_write_rms"] > 0.1
    assert by_family["silu_gating"]["deduplication_note"].startswith("overlaps")
