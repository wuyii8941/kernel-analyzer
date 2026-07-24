import json
from pathlib import Path


def test_qwen3_compiler_case_detects_semantic_disagreement_without_numeric_delta() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "results/operator_oracle/qwen3_compiler_grad_case_a_blind_locator.json").read_text()
    )
    assert report["oracle"]["numeric_equal"] is True
    assert report["oracle"]["semantic_disagreement"] is True
    assert "requires_grad" in report["oracle"]["semantic_fields_changed"]
    assert any("mm" in str(node["target"]) for node in report["candidate_operation_inventory"])


def test_qwen3_compiler_case_post_reveal_has_negative_control() -> None:
    root = Path(__file__).resolve().parents[1]
    score = json.loads(
        (root / "results/operator_oracle/qwen3_compiler_grad_case_b_post_reveal_score.json").read_text()
    )
    assert score["scoring"]["buggy_run_shows_silent_metadata_loss"] is True
    assert score["scoring"]["linear_negative_control_preserves_metadata"] is True
    assert score["scoring"]["stage_localization_supports_aot_autograd"] is True


def test_qwen3_checkpoint_weight_case_is_the_stronger_materialized_case() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/qwen_bug_sources/opaque_qwen3_grad_case_b/case_manifest.json").read_text()
    )
    report = json.loads(
        (root / "results/operator_oracle/qwen3_compiler_grad_case_b_blind_locator.json").read_text()
    )
    assert manifest["case_id"] == "opaque_qwen3_grad_case_b"
    assert manifest["subject"]["weight_source"].startswith("Qwen3-1.7B")
    assert report["oracle"]["numeric_equal"] is True
    assert report["oracle"]["semantic_disagreement"] is True


def test_qwen3_actual_boundary_case_uses_real_model_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/qwen_bug_sources/opaque_qwen3_grad_case_c/case_manifest.json").read_text()
    )
    report = json.loads(
        (root / "results/operator_oracle/qwen3_compiler_grad_case_c_blind_locator.json").read_text()
    )
    assert manifest["subject"]["input_source"].startswith("Qwen3-1.7B-layer0-actual")
    assert manifest["subject"]["weight_source"].startswith("Qwen3-1.7B")
    assert report["claim"]["level"] == "BACKEND_SPECIFIC_STAGE_CANDIDATE"
    assert report["claim"]["level"] == "BACKEND_SPECIFIC_STAGE_CANDIDATE"
    assert report["stage_control"]["candidate_differs_while_control_does_not"] is True
