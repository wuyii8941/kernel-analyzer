from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_active_cases_have_scoped_mean_bias_evidence() -> None:
    subprocess.run(
        [sys.executable, "scripts/analyze_all_case_mean_bias.py"],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(
        (ROOT / "results/property/root_cause_closure_v1/all_case_mean_bias_audit_v1.json").read_text()
    )
    assert report["summary"]["audited_problem_groups"] == 54
    assert report["summary"]["active_problem_groups"] == 52
    assert report["summary"]["population_mean_supported"] == 7
    assert report["summary"]["scoped_vector_mean_nonzero_supported"] == 4
    assert report["summary"]["scoped_aligned_mean_supported"] == 3
    assert report["summary"]["benchmark_exact_mean_vector_rows"] == 17
    assert report["summary"]["benchmark_exact_mean_vector_nonzero_rows"] == 17
    assert report["summary"]["benchmark_empirical_bank_projected_mean_supported"] == 12
    benchmark_population = report["benchmark_empirical_bank_mean_tests"]
    assert len(benchmark_population) == 17
    assert all(item["status"] in {"PROJECTED_MEAN_SUPPORTED", "PROJECTED_MEAN_NOT_CONFIRMED", "DIRECTION_NOT_IDENTIFIABLE"} for item in benchmark_population)
    assert report["estimand_warning"].startswith("Only AdamW8bit, Liger, Granite, softmax")
    rows = {row["problem_group"]: row for row in report["rows"]}
    silu = rows["silu_backward_evaluation"]
    assert silu["mean_bias_status"] == "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK"
    assert silu["population_test"]["independent_unit_count"] == 128
    assert silu["population_test"]["two_sided_95_interval"][1] < 0.0
    assert silu["population_test"]["negative_mean_supported_one_sided"] is True
    gemma_gelu = rows["gemma_gelu_backward_evaluation"]
    assert gemma_gelu["mean_bias_status"] == "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK"
    assert gemma_gelu["population_test"]["independent_unit_count"] == 128
    assert gemma_gelu["population_test"]["two_sided_95_interval"][1] < 0.0
    assert gemma_gelu["population_test"]["negative_mean_supported_one_sided"] is True
    assert rows["granite_moe_expert_contribution_order"]["mean_bias_status"].startswith(
        "SUPPORTED_PROJECTED_MEAN"
    )
    assert rows["gemma4_audio_output_projection_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE"
    )
    assert rows["gemma4_audio_attention_softcap_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_TOTAL_EFFECT_SOURCE_CLOSED_DIRECTION_NOT_CONFIRMED"
    )
    assert rows["gemma4_audio_attention_softmax_probability_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE"
    )
    assert rows["gemma4_audio_attention_softmax_probability_materialization"]["fixed_suite_aligned_evidence"][0]["aligned_interval_normal_95"][1] < 0.0
    assert rows["deepseek_embedding_gradient_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["deepseek_embedding_gradient_materialization"]["fixed_suite_aligned_evidence"]["aligned_interval_normal_95"][1] < 0.0
    assert rows["deepseek_fused_embedding_nll_partial_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["deepseek_fused_embedding_nll_partial_materialization"]["fixed_suite_aligned_evidence"]["bf16_partial"]["aligned_write_interval_normal_95"][1] < 0.0
    assert rows["mamba_fused_selective_scan_reassociation"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    mamba_scan = rows["mamba_fused_selective_scan_reassociation"]
    assert mamba_scan["fixed_suite_evidence"][0]["source_cut_aligned_interval"][1] < 0.0
    assert "not identical" in mamba_scan["interpretation"]
    assert rows["deepseek_fused_embedding_nll_reduction_order"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["deepseek_fused_embedding_nll_reduction_order"]["fixed_suite_aligned_evidence"]["fp32_reverse_order"]["aligned_write_interval_normal_95"][0] > 0.0
    assert rows["gemma4_audio_lightconv_glu_product_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE"
    )
    assert rows["gemma4_audio_lightconv_depthwise_conv_backward_accumulation"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE_RATIO_OF_SUMS"
    )
    assert rows["gemma4_audio_subsampling_convolution_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE_RATIO_OF_SUMS"
    )
    assert rows["deberta_disentangled_relative_attention_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["bloom_alibi_attention_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["gptneo_gelu_native_fp32_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["rwkv_time_decay_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["rwkv_time_decay_materialization"]["fixed_suite_aligned_evidence"]["decay_fp32"]["write_aligned_interval_normal_95"][1] < 0.0
    assert rows["rwkv_receptance_sigmoid_materialization"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert all(
        bank["confirmation_aligned_write_interval_normal_95"][1] < 0.0
        for bank in rows["rwkv_receptance_sigmoid_materialization"]["fixed_suite_aligned_evidence"]["banks"]
    )
    assert rows["bert_nll_loss_evaluation"]["mean_bias_status"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert rows["bert_nll_loss_evaluation"]["fixed_suite_aligned_evidence"]["aligned_write_interval_normal_95"][0] > 0.0
    granite = rows["granite_moe_expert_contribution_order"]["fixed_suite_evidence"][0]
    assert granite["fixed_suite_cross_state_u_statistic"] == 0.0
    liger = rows["liger_fused_linear_ce_dw_accumulation"]["fixed_suite_evidence"]
    assert any(entry["fixed_suite_cross_state_u_positive"] for entry in liger)
    adam = rows["adamw8bit_moment_quantization"]["population_test"]
    assert adam["mean_positive_under_student_t"] is True
    granite_population = rows["granite_moe_expert_contribution_order"]["population_test"]
    assert rows["granite_moe_expert_contribution_order"]["scoped_vector_mean_nonzero_supported"] is True
    assert granite_population["one_sided_95_lower_bound"] > 0.0
    assert granite_population["two_sided_95_nonzero_supported"] is False
    assert granite_population["positive_mean_supported_one_sided"] is True
    assert granite_population["positive_count"] == 53
    granite_gradient = rows["granite_moe_expert_contribution_order"]["population_stage_tests"]["gradient"]
    assert granite_gradient["two_sided_95_interval"][0] > 0.0
    assert granite_gradient["two_sided_95_nonzero_supported"] is True
    assert granite_gradient["one_sided_95_lower_bound"] > 0.0
    assert granite_gradient["positive_count"] == 38
    liger_population = rows["liger_fused_linear_ce_dw_accumulation"]["population_test"]
    assert rows["liger_fused_linear_ce_dw_accumulation"]["scoped_vector_mean_nonzero_supported"] is True
    assert liger_population["one_sided_95_lower_bound"] > 0.0
    assert liger_population["two_sided_95_nonzero_supported"] is True
    assert liger_population["positive_count"] == 44
    liger_256 = rows["liger_fused_linear_ce_dw_accumulation"]["additional_population_tests"]["length256"]
    assert liger_256["one_sided_95_lower_bound"] > 0.0
    assert liger_256["positive_count"] == 50
    softmax_tests = rows["softmax_saved_state_backward"]["empirical_bank_population_tests"]
    assert rows["softmax_saved_state_backward"]["scoped_vector_mean_nonzero_supported"] is True
    assert any(item["component"] == "semantic_total" and item["confirmation_projection_one_sided_95_lower"] > 0 for item in softmax_tests)
    mm_tests = rows["mm_gemm_output_and_accumulation"]["empirical_bank_population_tests"]
    assert rows["mm_gemm_output_and_accumulation"]["scoped_vector_mean_nonzero_supported"] is True
    assert len([item for item in mm_tests if item["component"] == "total"]) == 4
    assert all(item["confirmation_projection_one_sided_95_lower"] > 0 for item in mm_tests if item["component"] == "total")
