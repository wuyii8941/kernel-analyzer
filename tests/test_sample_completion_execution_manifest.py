import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_execution_manifest_has_four_model_groups_and_no_claimed_results():
    result = json.loads((ROOT / "results/property/sample_completion_v1/execution_manifest.json").read_text())
    assert result["status"] == "READY_FOR_GPU_BUT_NOT_EXECUTED"
    assert set(result["groups"]) == {"qwen", "phi4", "deepseek8b", "mamba"}
    assert result["total_search_units_with_exact_command"] == 16
    for group in result["groups"].values():
        assert group["status"] in {"READY_FOR_GPU", "BLOCKED_MODEL_PATH"}
        assert "--states" in group["commands"]["formal"]
