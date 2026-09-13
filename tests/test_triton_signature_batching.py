import hashlib
import json
from pathlib import Path

from scripts.build_triton_signature_batch_manifest import build
from scripts.run_triton_signature_batches import group_status, run_group


def source_campaign(tmp_path, task_id, *, release="release", architecture="qwen",
                    adapter="GENERIC_COVERAGE"):
    output = tmp_path / ("source-" + task_id.replace(":", "-"))
    return {
        "adapter": adapter,
        "release": str(tmp_path / release),
        "task_id": task_id,
        "operator_family": "ELEMENTWISE",
        "signature_key": ["ELEMENTWISE", "TRITON", "BACKWARD", task_id, "AOT_REPLAY"],
        "campaign_output": str(output),
        "runtime": {
            "architecture": architecture,
            "model": str(tmp_path / "model"),
            "input_bank": str(tmp_path / "bank.json"),
            "allow_graph_breaks": False,
        },
        "case": {"case_id": "case-" + task_id, "reference_method": "AOT_REPLAY"},
    }


def test_groups_only_not_started_generic_cases_by_runtime(tmp_path):
    first = source_campaign(tmp_path, "backward:1:out")
    second = source_campaign(tmp_path, "backward:2:out")
    completed = source_campaign(tmp_path, "backward:3:out")
    run_id = hashlib.sha256(completed["task_id"].encode()).hexdigest()[:20]
    status = Path(completed["campaign_output"]) / "runs" / run_id / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"status":"VALID"}')
    specialized = source_campaign(
        tmp_path, "backward:4:out", adapter="DECAYED_RECURRENCE"
    )
    result = build(
        {"campaigns": [first, second, completed, specialized]},
        output_root=tmp_path / "batched", batch_size=4,
    )
    assert result["group_count"] == 1
    assert result["case_count"] == 2
    assert result["groups"][0]["task_ids"] == ["backward:1:out", "backward:2:out"]
    assert {row["reason"] for row in result["excluded"]} == {
        "SOURCE_CAMPAIGN_ALREADY_STARTED_OR_TERMINAL",
        "SPECIALIZED_ADAPTER_RETAINS_ITS_EXISTING_EXECUTION_PATH",
    }
    assert result["selection_uses_numerical_outcomes"] is False


def test_group_status_preserves_each_task_result(tmp_path):
    group = {
        "campaign_output": str(tmp_path / "output"),
        "task_ids": ["backward:1:out", "backward:2:out"],
    }
    output = Path(group["campaign_output"])
    output.mkdir()
    (output / "protocol.json").write_text("{}")
    assert group_status(group)["execution_status"] == "PARTIAL_OR_NOT_STARTED"
    for task_id, value in zip(group["task_ids"], ("VALID", "EXECUTION_FAILED")):
        run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
        status = output / "runs" / run_id / "status.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({"status": value}))
    result = group_status(group)
    assert result["execution_status"] == "TERMINAL_WITHOUT_VALID_MEASUREMENT"
    assert result["task_statuses"] == {
        "backward:1:out": "VALID", "backward:2:out": "EXECUTION_FAILED"
    }


def test_terminal_task_can_only_be_retried_with_explicit_new_release(tmp_path):
    item = source_campaign(tmp_path, "backward:1:out")
    run_id = hashlib.sha256(item["task_id"].encode()).hexdigest()[:20]
    status = Path(item["campaign_output"]) / "runs" / run_id / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"status":"EXECUTION_FAILED"}')
    repaired = tmp_path / "repaired"
    repaired.mkdir()
    result = build(
        {"campaigns": [item]}, output_root=tmp_path / "batched", batch_size=4,
        release_overrides={str((tmp_path / "release").resolve()): str(repaired.resolve())},
        retry_tasks={item["task_id"]},
        graph_break_releases={str((tmp_path / "release").resolve())},
    )
    assert result["case_count"] == 1
    assert result["groups"][0]["release"] == str(repaired.resolve())
    assert result["groups"][0]["source_campaigns"][0]["explicit_retry"] is True
    assert result["groups"][0]["runtime"]["allow_graph_breaks"] is True


def test_group_timeout_is_terminal(tmp_path):
    group = {"campaign_output": str(tmp_path), "task_ids": ["backward:1:out"]}
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "execution_timeout.json").write_text("{}")
    assert group_status(group)["execution_status"] == "TERMINAL_TIMEOUT"


def test_timeout_record_preserves_completed_task_semantics(tmp_path, monkeypatch):
    group = {
        "group_id": "runtime-test",
        "campaign_output": str(tmp_path),
        "task_ids": ["backward:1:out"],
        "release": str(tmp_path / "release"),
        "case_plan": str(tmp_path / "plan.json"),
        "runtime": {
            "architecture": "qwen", "model": "model", "input_bank": "bank",
        },
    }
    (tmp_path / "protocol.json").write_text("{}")
    monkeypatch.setattr(
        "scripts.run_triton_signature_batches._run_process",
        lambda command, timeout: (None, "", "", True),
    )
    result = run_group(
        group, action="run", device="cuda:0", batch_size=1, timeout_seconds=1,
    )
    assert result["timed_out"] is True
    record = json.loads((tmp_path / "execution_timeout.json").read_text())
    assert record["completed_task_statuses_remain_valid"] is True
    assert record["incomplete_task_outputs_are_not_valid_measurements"] is True
