import hashlib
from pathlib import Path

import pytest

from scripts.build_triton_signature_continuation_manifest import build_continuation


def _group(tmp_path: Path):
    return {
        "group_id": "runtime-mamba-old",
        "release": str(tmp_path / "release"),
        "runtime": {
            "architecture": "mamba",
            "model": str(tmp_path / "model"),
            "input_bank": str(tmp_path / "bank.json"),
            "allow_graph_breaks": False,
        },
        "campaign_output": str(tmp_path / "old-output"),
        "task_ids": ["backward:1:x", "backward:2:x", "backward:3:x", "backward:4:x"],
        "cases": [{"case_id": f"case-{i}"} for i in range(1, 5)],
        "source_campaigns": [
            {"task_id": f"backward:{i}:x", "operator_family": "REDUCTION"}
            for i in range(1, 5)
        ],
    }


def _write_status(group, task_id, status):
    run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
    path = Path(group["campaign_output"]) / "runs" / run_id / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status":"' + status + '"}')


def test_continuation_uses_only_execution_status_and_splits_groups(tmp_path):
    group = _group(tmp_path)
    _write_status(group, "backward:1:x", "VALID")
    _write_status(group, "backward:2:x", "EXECUTION_FAILED")
    incomplete_id = hashlib.sha256("backward:3:x".encode()).hexdigest()[:20]
    (Path(group["campaign_output"]) / "runs" / incomplete_id).mkdir(parents=True)
    result = build_continuation(
        {"batch_size": 4, "groups": [group]},
        output_root=tmp_path / "continuation", max_cases_per_group=1,
    )
    assert result["case_count"] == 2
    assert result["group_count"] == 2
    assert result["batch_size"] == 1
    assert [item["task_ids"] for item in result["groups"]] == [
        ["backward:3:x"], ["backward:4:x"]
    ]
    reasons = {row["task_id"]: row["reason"] for row in result["excluded"]}
    assert reasons == {
        "backward:1:x": "ALREADY_VALID",
        "backward:2:x": "TERMINAL_TASK_STATUS_NOT_RETRIED",
    }
    assert result["selection_uses_numerical_outcomes"] is False
    assert result["scientific_selection_unchanged"] is True


def test_terminal_retry_requires_explicit_task_and_new_release(tmp_path):
    group = _group(tmp_path)
    for task_id in group["task_ids"]:
        _write_status(group, task_id, "EXECUTION_FAILED")
    with pytest.raises(ValueError, match="different runtime release"):
        build_continuation(
            {"batch_size": 4, "groups": [group]},
            output_root=tmp_path / "bad", max_cases_per_group=4,
            terminal_retry_tasks={"backward:1:x"},
        )
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    old = str((tmp_path / "release").resolve())
    result = build_continuation(
        {"batch_size": 4, "groups": [group]},
        output_root=tmp_path / "retry", max_cases_per_group=4,
        terminal_retry_tasks={"backward:1:x"},
        release_overrides={old: str(replacement.resolve())},
    )
    assert result["case_count"] == 1
    assert result["groups"][0]["release"] == str(replacement.resolve())
    assert result["groups"][0]["source_campaigns"][0]["explicit_terminal_retry"] is True


def test_only_task_filter_does_not_pull_unrelated_not_started_work(tmp_path):
    group = _group(tmp_path)
    result = build_continuation(
        {"batch_size": 4, "groups": [group]}, output_root=tmp_path / "filtered",
        max_cases_per_group=4, only_tasks={"backward:4:x"},
    )
    assert result["case_count"] == 1
    assert result["groups"][0]["task_ids"] == ["backward:4:x"]
    assert sum(
        row["reason"] == "NOT_SELECTED_FOR_THIS_EXECUTION_CONTINUATION"
        for row in result["excluded"]
    ) == 3
