import json
from pathlib import Path

import pytest

from scripts.build_rotary_family_evidence import build


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def analysis(case_id: str, rms: float) -> dict:
    return {
        "schema": "kernel-analyzer-analysis-result-v1",
        "case_id": case_id,
        "claim_scope": "FIXED_SUITE_UPDATE",
        "measurement_status": "VALID",
        "equivalence_decision": "NON_EQUIVALENT",
        "training_outcome": "NOT_MEASURED",
        "bias_analysis": {
            "decision_rule": "ORIGINAL_COORDINATE_Q_ONLY_FIXED_SUITE",
            "population_guarantee": False,
            "fixed_suite_total_rms": rms,
            "fixed_suite_aligned_ratio_of_sums": -0.01,
        },
    }


def plan(case_id: str, semantic_hash: str, carrier: str = "weight") -> dict:
    return {
        "cases": [
            {
                "case_id": case_id,
                "carrier": carrier,
                "implementation_kind": "TRITON",
                "reference_family": "ATTENTION_POSITION_SCALING",
                "reference_method": "REGISTERED_SAME_INPUT_REFERENCE",
                "reference_contract": {
                    "function_semantic_ast_sha256": semantic_hash,
                },
            }
        ]
    }


def optimizer_summary(case_id: str) -> dict:
    stage = {"PARAMETER_WRITE": {"confirmation_total_rms": 0.1}}
    return {
        "case_id": case_id,
        "status": "COMPLETE_FIXED_SUITE_DESCRIPTIVE_COMPARISON",
        "prediction_status": "POST_OUTCOME_DIAGNOSTIC_NOT_PREREGISTERED",
        "state_ids": [f"state-{index}" for index in range(32)],
        "conditions": {
            "COLD_ZERO_MOMENTS": stage,
            "WARM_MOMENTS": stage,
            "WARM_PARAMETERS_RESET_MOMENTS": stage,
        },
        "comparisons": {
            "warm_and_reset_gradient_statistics_exactly_equal": True,
        },
    }


def test_builds_bounded_rotary_evidence(tmp_path):
    case_id = "rotary-case"
    result = build(
        write(tmp_path / "high.json", analysis(case_id, 0.11)),
        write(tmp_path / "low.json", analysis(case_id, 0.08)),
        write(tmp_path / "high-plan.json", plan(case_id, "semantic")),
        write(tmp_path / "low-plan.json", plan(case_id, "semantic")),
        write(tmp_path / "optimizer.json", optimizer_summary(case_id)),
    )
    row = result["records"][0]
    assert row["family"] == "ROTARY"
    assert row["backend"] == "INDUCTOR_GENERATED_TRITON"
    assert row["high_position_result"]["parameter_write_total_rms"] == 0.11
    assert row["low_position_result"]["parameter_write_total_rms"] == 0.08
    assert row["position_scaling_is_unique_root_cause"] is False
    assert row["population_guarantee"] is False
    assert row["training_quality_claim"] is False


def test_rejects_semantically_different_source(tmp_path):
    case_id = "rotary-case"
    with pytest.raises(ValueError, match="same semantic source"):
        build(
            write(tmp_path / "high.json", analysis(case_id, 0.11)),
            write(tmp_path / "low.json", analysis(case_id, 0.08)),
            write(tmp_path / "high-plan.json", plan(case_id, "high")),
            write(tmp_path / "low-plan.json", plan(case_id, "low")),
            write(tmp_path / "optimizer.json", optimizer_summary(case_id)),
        )
