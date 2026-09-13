import json
from pathlib import Path

from scripts.build_scoped_mainline_completion_audit import build


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def evidence(tmp_path: Path) -> dict[str, Path]:
    return {
        "campaign_manifest": dump(
            tmp_path / "manifest.json",
            {
                "selection_uses_numerical_outcomes": False,
                "campaigns": [{"implementation_kind": "TRITON"}] * 64,
            },
        ),
        "campaign": dump(
            tmp_path / "campaign.json",
            {
                "selection_uses_numerical_outcomes": False,
                "frozen_task_count": 64,
                "final_status_counts": {
                    "VALID": 31,
                    "EXECUTION_TIMEOUT_NOT_MEASURED": 28,
                    "EXECUTION_FAILED_NOT_MEASURED": 4,
                    "RUNTIME_PATH_MISMATCH_NOT_MEASURED": 1,
                },
                "all_frozen_tasks_accounted": True,
                "valid_measurements_are_not_root_causes": True,
            },
        ),
        "catalog": dump(
            tmp_path / "catalog.json",
            {
                "all_positions_classified_including_explicit_unresolved": True,
                "canonical_position_count": 146104,
                "operator_family_count": 19,
                "distinct_position_support_status_counts": {
                    "VALID_MEASUREMENT_COMPLETED": 551
                },
            },
        ),
        "deduplication": dump(
            tmp_path / "dedup.json",
            {
                "valid_position_count": 31,
                "structural_signature_count": 31,
                "exact_candidate_computation_count": 30,
                "broad_catalogue_family_count": 4,
                "independent_root_cause_count": None,
            },
        ),
        "bounded_q": dump(tmp_path / "q.json", {"status": "PASS"}),
        "exceedance": dump(tmp_path / "exceedance.json", {"status": "PASS"}),
        "mechanism": dump(tmp_path / "mechanism.json", {"status": "VERIFIED"}),
        "training": dump(
            tmp_path / "training.json",
            {
                "status": "VERIFIED",
                "stream_count": 8,
                "recomputed": {"decision": "MATERIAL_IMPROVEMENT"},
            },
        ),
    }


def test_complete_when_all_bounded_requirements_hold(tmp_path):
    result = build(evidence(tmp_path))
    assert result["status"] == "COMPLETE_SCOPED_MAINLINE"
    assert all(result["checks"].values())


def test_fails_closed_when_training_result_is_not_verified(tmp_path):
    paths = evidence(tmp_path)
    dump(
        paths["training"],
        {
            "status": "VERIFIED",
            "stream_count": 8,
            "recomputed": {"decision": "NO_MATERIAL_IMPROVEMENT"},
        },
    )
    result = build(paths)
    assert result["status"] == "INCOMPLETE"
    assert result["checks"]["adamw8bit_training_improvement_verified"] is False
