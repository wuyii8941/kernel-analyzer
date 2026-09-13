import json
from pathlib import Path

import pytest

from scripts.build_specialized_signature_continuation import build


def source(tmp_path: Path, *, valid: bool = False) -> dict:
    old = tmp_path / "old"
    old.mkdir()
    if valid:
        (old / "completion_verification.json").write_text(json.dumps({
            "records": [{"task_id": "backward:1:out_ptr0", "status": "VERIFIED"}]
        }))
    return {"campaigns": [{
        "task_id": "backward:1:out_ptr0", "adapter": "DECAYED_RECURRENCE",
        "campaign_output": str(old), "case": {"case_id": "frozen"},
    }]}


def test_preserves_task_and_changes_only_versioned_inputs(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{"task_id": "backward:1:out_ptr0"}]}))
    result = build(source(tmp_path), specialized_plan=plan,
                   output_root=tmp_path / "next")
    campaign = result["campaigns"][0]
    assert campaign["task_id"] == "backward:1:out_ptr0"
    assert campaign["case"]["case_id"] == "frozen-source-refresh"
    assert result["selection_uses_numerical_outcomes"] is False


def test_refuses_to_replace_valid_measurement(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{"task_id": "backward:1:out_ptr0"}]}))
    with pytest.raises(ValueError, match="valid measurement"):
        build(source(tmp_path, valid=True), specialized_plan=plan,
              output_root=tmp_path / "next")


def test_selects_exact_task_from_larger_manifest(tmp_path: Path) -> None:
    data = source(tmp_path)
    data["campaigns"].append({"task_id": "backward:2:out_ptr0"})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{"task_id": "backward:1:out_ptr0"}]}))
    result = build(data, specialized_plan=plan, output_root=tmp_path / "next",
                   task_id="backward:1:out_ptr0")
    assert result["campaigns"][0]["task_id"] == "backward:1:out_ptr0"
