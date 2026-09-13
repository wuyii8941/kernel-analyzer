from scripts.audit_triton_signature_campaign import audit


def test_valid_retry_wins_and_unmeasured_failure_is_preserved(tmp_path):
    release = str(tmp_path / "release")
    master = {"campaigns": [
        {"release": release, "task_id": "a", "case": {"case_id": "a"},
         "adapter": "GENERIC_COVERAGE", "operator_family": "REDUCTION"},
        {"release": release, "task_id": "b", "case": {"case_id": "b"},
         "adapter": "DECAYED_RECURRENCE", "operator_family": "RECURRENCE"},
    ]}
    first = {"schema": "triton-signature-runtime-batch-summary-v1", "rows": [
        {"source_release": release, "task_id": "a", "execution_status": "EXECUTION_FAILED",
         "measurement_status": "NOT_ASSESSED"},
    ]}
    retry = {"schema": "triton-signature-runtime-batch-summary-v1", "rows": [
        {"source_release": release, "task_id": "a", "execution_status": "VALID",
         "measurement_status": "VALID"},
    ]}
    special = {"schema": "family-first-campaign-summary-v1", "rows": [
        {"source_release": release, "task_id": "b", "execution_status": "INCOMPLETE_ATTEMPT",
         "measurement_status": "NOT_ASSESSED", "failure_reason": "ValueError: Live recurrence source differs"},
    ]}
    result = audit(master, [first, retry, special])
    assert result["final_status_counts"] == {
        "VALID": 1, "RUNTIME_PATH_MISMATCH_NOT_MEASURED": 1
    }
    assert result["all_frozen_tasks_accounted"] is True
