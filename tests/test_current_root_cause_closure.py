import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_ledger():
    return json.loads(
        (ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json")
        .read_text()
    )


def by_group(data, group_id):
    return next(row for row in data["rows"] if row["problem_group"] == group_id)


def test_current_root_cause_ledger_recomputes_and_keeps_open_branches():
    subprocess.run(
        [sys.executable, "scripts/build_current_root_cause_closure.py"],
        cwd=ROOT,
        check=True,
    )
    data = load_ledger()
    rows = data["rows"]
    assert len(rows) == len({row["problem_group"] for row in rows}) == 12
    assert data["summary"]["end_to_end_count"] == 1
    assert data["summary"]["negative_control_count"] == 2
    assert len(data["coverage_collections"]) == 2
    inventory = data["source_record_inventory"]
    assert inventory["record_count"] == 866
    assert inventory["source_kind_counts"] == {
        "HISTORICAL_MATRIX_ROW": 301,
        "LEGACY_CASE_REAUDIT": 8,
        "MAINLINE_ROLE_RECORD": 6,
        "MEASURED_POSITION": 551,
    }
    prior = json.loads(
        (ROOT / "results/property/case_causal_audit_v1/scientific_case_closure.json")
        .read_text()
    )
    accounted = {row["problem_group"] for row in rows}
    accounted.update(item["collection"] for item in data["coverage_collections"])
    assert accounted == {row["problem_group"] for row in prior["rows"]} | {
        "gemma_gelu_backward_evaluation",
        "gemma_rms_feature_reduction_order",
        "granite_router_topk_selection",
        "granite_moe_expert_contribution_order",
        "gemma_bound_square_sum",
    }
    assert by_group(data, "adamw8bit_moment_quantization")["closure"].startswith(
        "END_TO_END"
    )
    assert by_group(data, "softmax_saved_state_backward")["closure"].endswith(
        "NATURAL_BIAS_OPEN"
    )
    assert by_group(data, "fused_rope_position_scaling")["closure"].endswith(
        "SOURCE_ROOT_OPEN"
    )
    silu = by_group(data, "silu_backward_evaluation")
    assert silu["closure"].endswith("NATURAL_BIAS_OPEN")
    assert silu["derived"]["same_local_operands"] is True
    assert silu["derived"]["source_attribution"] == "CHECKED_FUNCTION_AST_AND_PRE_CALL_INPUTS"
    assert silu["derived"]["stages"]["PARAMETER_WRITE"]["total_effect_rms_range"][1] > 2e-4
    assert by_group(data, "gemma_bound_square_sum")["closure"].endswith(
        "SOURCE_MISMATCH_UNRESOLVED"
    )
    liger = by_group(data, "liger_fused_linear_ce_dw_accumulation")["derived"]
    assert liger["64"]["state_count"] == 32
    assert liger["64"]["confirmation_positive_count"] == 14
    assert liger["256"]["confirmation_positive_count"] == 11
    assert liger["64"]["update_branches_confirmed"]["additive"] is True
    assert liger["training_1024"]["validation_loss_difference"]["1024"] == 0.0


def test_gelu_source_evidence_is_recomputed_from_raw_records():
    data = load_ledger()
    evidence = by_group(data, "gemma_gelu_backward_evaluation")["derived"]["evidence"]
    assert evidence["natural_reference"]["state_count"] == 32
    assert evidence["natural_reference"]["confirmation_count"] == 16
    assert evidence["explicit_exponential_tanh"]["gradient_rms"] > 0.001
    assert evidence["explicit_exponential_tanh"]["update_rms"] > 0.04
    assert evidence["native_tanh_vs_fma_exact_in_recorded_profile"] is True
    for variant in (
        "natural_reference",
        "explicit_exponential_tanh",
        "native_tanh",
        "fused_multiply_add",
    ):
        for key in ("local_rms", "gradient_rms", "update_rms", "write_rms"):
            assert math.isfinite(evidence[variant][key])


def test_negative_controls_are_not_promoted_to_bias_cases():
    data = load_ledger()
    rms = by_group(data, "gemma_rms_feature_reduction_order")["derived"]
    endpoint_200 = rms["evidence"]["forward:200:out_ptr0"]
    assert endpoint_200["direct_endpoint_rms"] == 0.0
    assert endpoint_200["direct_gradient_rms"] == 0.0

    selection = by_group(data, "granite_router_topk_selection")["derived"]
    assert selection["all_scores_equal"] is True
    assert selection["all_gradients_zero_difference"] is True
    assert selection["all_writes_zero_difference"] is True
    assert selection["selected_set_unchanged"] is True

    granite = by_group(data, "granite_moe_expert_contribution_order")["derived"]
    assert by_group(data, "granite_moe_expert_contribution_order")["closure"].endswith(
        "NATURAL_BIAS_OPEN"
    )
    assert granite["state_count"] == 16
    assert 0.00002 < granite["fixed_suite_total_rms"] < 0.00003
    assert granite["equivalence_decision"] == "EQUIVALENT"


def test_mm_sources_remain_case_specific():
    data = load_ledger()
    mm = by_group(data, "mm_gemm_output_and_accumulation")["derived"][
        "case_specific_sources"
    ]
    assert mm["qwen_seq128_forward_8_output"]["coherent_sources"] == [
        "output_rounding"
    ]
    assert set(mm["qwen_seq64_forward_8_output"]["coherent_sources"]) == {
        "kernel",
        "output_rounding",
    }
    assert set(mm["mamba_seq64_forward_1_output"]["coherent_sources"]) == {
        "kernel",
        "output_rounding",
    }
    assert mm["phi4_seq64_backward_497_output"]["coherent_sources"] == ["kernel"]


def test_every_frozen_benchmark_and_catalog_family_has_a_root_cause_boundary():
    data = load_ledger()
    benchmark = data["generalization_benchmark_frontier"]
    assert benchmark["case_count"] == 16
    assert len(benchmark["cases"]) == 16
    assert all(
        row["root_cause_status"] == "MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION"
        and row["missing_observation"]
        for row in benchmark["cases"]
    )

    families = data["operator_family_frontier"]
    assert families["family_count"] == 17
    assert len(families["families"]) == 17
    assert {row["family_id"] for row in families["families"]} == {
        "LINEAR", "NORMALIZATION", "SOFTMAX", "CROSS_ENTROPY", "SILU_GATING",
        "SOFTPLUS", "RECURRENCE", "ROTARY", "REDUCTION", "INDEXED_ACCUMULATION",
        "GELU", "CONVOLUTION", "EMBEDDING", "SELECTION", "OPTIMIZER_UPDATE",
        "FUSED_ATTENTION", "ELEMENTWISE_BIAS",
    }
    assert all(row["root_cause_status"] and row["interpretation"] for row in families["families"])
