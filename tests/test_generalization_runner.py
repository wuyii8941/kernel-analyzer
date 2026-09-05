from __future__ import annotations

import json

from scripts.run_generalization_benchmark_v1 import _complete_result


def test_skip_existing_requires_a_complete_valid_artifact(tmp_path) -> None:
    target = tmp_path / "case.json"
    target.write_text(json.dumps({"status": "ABSTAIN_EXECUTION_FAILURE"}))
    assert not _complete_result(target)
    target.write_text(json.dumps({
        "schema": "kernel-analyzer-training-bias-profile-v2-raw-case",
        "status": "COMPLETE",
        "runtime_boundary": {"task_id": "backward:1:out_ptr0"},
    }))
    assert _complete_result(target)
