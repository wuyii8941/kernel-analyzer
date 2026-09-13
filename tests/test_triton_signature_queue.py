import hashlib
import json

from scripts.run_triton_signature_queue import campaign_status, pending_indices


def campaign(tmp_path, task_id="backward:1:out_ptr0"):
    return {
        "campaign_output": str(tmp_path / "output"),
        "adapter": "GENERIC_COVERAGE",
        "task_id": task_id,
        "operator_family": "FUSED_MIXED",
        "case": {"case_id": "case"},
    }


def test_valid_and_failed_campaigns_are_not_requeued(tmp_path):
    good = campaign(tmp_path / "good", "backward:1:out_ptr0")
    bad = campaign(tmp_path / "bad", "backward:2:out_ptr0")
    new = campaign(tmp_path / "new", "backward:3:out_ptr0")
    for item, status_value in ((good, "VALID"), (bad, "EXECUTION_FAILED")):
        run_id = hashlib.sha256(item["task_id"].encode()).hexdigest()[:20]
        status = __import__("pathlib").Path(item["campaign_output"]) / "runs" / run_id / "status.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({"status": status_value}))
    assert campaign_status(good) == "VALID"
    assert campaign_status(bad) == "TERMINAL_EXECUTION_FAILED"
    pending, inventory = pending_indices({"campaigns": [good, bad, new]})
    assert pending == [2]
    assert [row["status"] for row in inventory] == [
        "VALID", "TERMINAL_EXECUTION_FAILED", "NOT_STARTED"
    ]


def test_incomplete_campaign_is_not_silently_restarted(tmp_path):
    item = campaign(tmp_path)
    run_id = hashlib.sha256(item["task_id"].encode()).hexdigest()[:20]
    (__import__("pathlib").Path(item["campaign_output"]) / "runs" / run_id).mkdir(parents=True)
    assert campaign_status(item) == "INCOMPLETE_ATTEMPT"
    pending, _ = pending_indices({"campaigns": [item]})
    assert pending == []


def test_timeout_record_is_terminal_and_not_silently_restarted(tmp_path):
    item = campaign(tmp_path)
    output = __import__("pathlib").Path(item["campaign_output"])
    output.mkdir(parents=True)
    (output / "execution_timeout.json").write_text('{"timeout_seconds":1}')
    assert campaign_status(item) == "TERMINAL_TIMEOUT"
    pending, _ = pending_indices({"campaigns": [item]})
    assert pending == []


def test_adapter_filter_runs_only_explicit_specialized_path(tmp_path):
    generic = campaign(tmp_path / "generic", "backward:1:x")
    specialized = campaign(tmp_path / "special", "backward:2:x")
    specialized["adapter"] = "DECAYED_RECURRENCE"
    pending, inventory = pending_indices(
        {"campaigns": [generic, specialized]}, adapters={"DECAYED_RECURRENCE"}
    )
    assert pending == [1]
    assert [row["adapter"] for row in inventory] == [
        "GENERIC_COVERAGE", "DECAYED_RECURRENCE"
    ]
