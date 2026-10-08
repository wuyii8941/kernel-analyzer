#!/usr/bin/env python3
"""Record which root-cause conclusions are derivable from retained evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# KA_OUTPUT_DIR (tests): write the output there instead of the tracked file
_OUTPUT_DIR = __import__("os").environ.get("KA_OUTPUT_DIR")
OUT = (Path(_OUTPUT_DIR) / "exhaustion_audit.json" if _OUTPUT_DIR
       else ROOT / "results/property/root_cause_closure_v1/exhaustion_audit.json")


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    softmax = read("results/property/root_cause_closure_v1/softmax_saved_state.json")
    liger = read("results/property/root_cause_closure_v1/liger_chunk_order_reanalysis.json")
    silu = read("results/property/numerical_coverage_v1/deepseek128_silu_batch000/raw/mapped_backward_667_in_out_ptr0-silu-common-input.json")
    silu_factorial = read(
        "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/"
        "mapped_backward_667_in_out_ptr0-silu-common-input.json"
    )
    silu_factorial_comparison = read(
        "results/property/root_cause_closure_v1/silu_source_factorial_v1.json"
    )
    gelu_factorial = read(
        "results/property/root_cause_closure_v1/gelu_source_factorial_v1.json"
    )
    gelu_natural = read(
        "results/property/root_cause_closure_v1/gelu_natural_intermediates_v1.json"
    )
    gelu_componentwise = read(
        "results/property/root_cause_closure_v1/gelu_componentwise_intermediates_v1.json"
    )
    silu_intermediate = read(
        "results/property/root_cause_closure_v1/silu_intermediate_probe_v1.json"
    )
    silu_natural = read(
        "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json"
    )
    silu_intervention = read(
        "results/property/root_cause_closure_v1/silu_single_source_intervention_v2_4_20260920.json"
    )
    gelu_arithmetic_interventions = read(
        "results/property/root_cause_closure_v1/gelu_arithmetic_interventions_4_20260920.json"
    )
    gelu_mediation = read(
        "results/property/root_cause_closure_v1/gelu_single_source_mediation_v6_32_20260921.json"
    )
    rotary = read("results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json")
    rotary_probe = read("results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v4.json")
    rotary_natural = read("results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json")
    rotary_factorial = read(
        "results/property/root_cause_closure_v1/ministral_vision_rope_source_factorial_natural_8_20260920.json"
    )
    mm = read("results/property/case_causal_audit_v1/mm_conditional_sources.json")
    attention_input_probe = read(
        "results/property/root_cause_closure_v1/l23_softmax_kernel_input_probe_v1.json"
    )
    attention_scores_probe = read(
        "results/property/root_cause_closure_v1/l23_softmax_scores_probe_v1.json"
    )
    attention_statistics_probe = read(
        "results/property/root_cause_closure_v1/l23_softmax_statistics_probe_v1.json"
    )
    attention_carrier_probe = read(
        "results/property/root_cause_closure_v1/l23_softmax_carrier_right_probe_v1.json"
    )
    rmsnorm = read(
        "results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json"
    )
    gemma_conv = read(
        "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json"
    )
    mamba_conv = read(
        "results/property/new_problem_group_search_v1/mamba_causal_conv_materialization_natural_32_20260918.json"
    )
    mamba_state_output = read(
        "results/property/new_problem_group_search_v1/mamba_state_output_contraction_natural_32_20260918.json"
    )
    mamba_d_skip = read(
        "results/property/new_problem_group_search_v1/mamba_d_skip_materialization_natural_32_20260918.json"
    )
    embedding = read(
        "results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json"
    )
    gemma4_nll = read(
        "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json"
    )
    gemma4_softcap = read(
        "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json"
    )
    gemma4_softcap_logits = read(
        "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_logits_26_20260919.json"
    )
    gemma4_row_reduction = read(
        "results/property/new_problem_group_search_v1/gemma4_rms_row_reduction_natural_26_20260919.json"
    )
    gemma4_audio = read(
        "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_16_20260919.json"
    )
    gemma4_audio_attention_softcap = read(
        "results/property/new_problem_group_search_v1/gemma4_audio_attention_softcap_layer0_natural_16_20260920.json"
    )
    gemma4_audio_attention_softmax = read(
        "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_natural_16_20260920.json"
    )
    gemma4_audio_subsample = read(
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_raw_16_20260920.json"
    )
    deberta_relative = read(
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer0_seed0_32_20260919.json"
    )
    deberta_relative_last = read(
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer5_32_20260919.json"
    )
    bloom_alibi = read(
        "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_16_20260919.json"
    )
    bloom_alibi_last = read(
        "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_layer11_16_20260919.json"
    )
    qwen3_attention = read(
        "results/property/new_problem_group_search_v1/qwen3_attention_backend_isolated_layer13_qproj_16_20260918.json"
    )
    qwen3_attention_math = read(
        "results/property/new_problem_group_search_v1/qwen3_attention_backend_math_isolated_layer13_qproj_8_20260918.json"
    )
    mamba_softplus = read(
        "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_32_20260918.json"
    )
    mamba_transition = read(
        "results/property/new_problem_group_search_v1/mamba_discrete_transition_materialization_natural_cpu16_20260919.json"
    )
    mamba_scan_source_cut = read(
        "results/property/new_problem_group_search_v1/mamba_scan_source_cut_layer3_16_20260920.json"
    )
    bert_attention_bias = read(
        "results/property/new_problem_group_search_v1/bert_linear_bias_materialization_natural_cpu32_20260919.json"
    )
    bert_addmm = read(
        "results/property/case_causal_audit_v1/bert_fused_addmm_bias_materialization_boundary.json"
    )
    bert_score = read(
        "results/property/case_causal_audit_v1/bert_attention_score_materialization_boundary.json"
    )
    bert_value = read(
        "results/property/case_causal_audit_v1/bert_attention_value_materialization_boundary.json"
    )
    bert_nll = read(
        "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json"
    )
    qwen3vl_position = read(
        "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_16_20260918.json"
    )
    bert_pooler_tanh = read(
        "results/property/new_problem_group_search_v1/bert_pooler_tanh_natural_cpu32_20260919.json"
    )
    bert_pooler_tanh_dense = read(
        "results/property/new_problem_group_search_v1/bert_pooler_tanh_dense_natural_cpu32_20260919.json"
    )
    olmoe_router = read(
        "results/property/new_problem_group_search_v1/olmoe_router_accum_natural_16_20260919.json"
    )
    granite_router = read(
        "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json"
    )
    liger_jsd = read(
        "results/property/case_causal_audit_v1/liger_jsd_natural_training_boundary.json"
    )
    bert_softmax = read(
        "results/property/case_causal_audit_v1/bert_attention_softmax_materialization_boundary.json"
    )
    gptneo_gelu = read(
        "results/property/root_cause_closure_v1/gptneo_gelu_natural_32_20260920.json"
    )
    rwkv_time_decay = read(
        "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json"
    )
    rwkv_receptance_sigmoid = read(
        "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_natural_32_20260920.json"
    )
    rwkv_receptance_sigmoid_method = read(
        "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_method_32_20260920.json"
    )
    embedding_gradient_materialization = read(
        "results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json"
    )
    fused_embedding_partial = read(
        "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json"
    )
    assert softmax["status"] == "FIXED_CALL_LOCAL_ROOT_CONFIRMED" and softmax["row_count"] == 114688
    assert liger["status"] == "COMPLETE"
    assert liger["update_equivalence"]["decision"] == "FIXED_SUITE_UPDATE_EQUIVALENT"
    assert rwkv_receptance_sigmoid["status"] == "COMPLETE_NATURAL_RWKV_RECEPTANCE_SIGMOID_BOUNDARY"
    assert rwkv_receptance_sigmoid_method["status"] == "COMPLETE_NATURAL_RWKV_RECEPTANCE_SIGMOID_BOUNDARY"
    assert silu["status"] == "COMPLETE" and "original_coordinate_statistics" in silu
    assert silu_factorial["status"] == "COMPLETE" and silu_factorial["reference_comparison_scope"]["same_local_operands"]
    assert silu_factorial_comparison["effect_energy_equal_within_tolerance_all_stages"]
    assert gelu_factorial["comparisons"]["native_tanh_fused_multiply_add_profile_match"]
    assert gelu_natural["status"] == "REJECTED_FINAL_ALIGNMENT"
    assert gelu_componentwise["status"] == "COMPLETE_COMPONENTWISE_PROBE"
    assert all(gelu_componentwise["derivative_alignment_by_component"].values())
    assert silu_intermediate["first_difference_candidates"]["sigmoid"]
    assert not silu_intermediate["first_difference_candidates"]["exp"]
    assert silu_natural["status"] == "COMPLETE_NATURAL_INTERMEDIATE_PROBE"
    assert silu_natural["first_difference_candidates"]["sigmoid"]
    assert not silu_natural["first_difference_candidates"]["exp"]
    assert silu_intervention["status"] == "COMPLETE_PATH_PRESERVING_RECIPROCAL_INTERVENTION_V2"
    assert silu_intervention["summary"]["write_effect_rms_mean"] < 1e-4
    assert gelu_arithmetic_interventions["status"] == "COMPLETE_PATH_PRESERVING_GELU_ARITHMETIC_INTERVENTIONS"
    for variant in ("exp_tanh", "polynomial_fma", "derivative_factor_fma", "derivative_fma"):
        assert gelu_arithmetic_interventions["summary"][variant]["write_effect_rms_mean"] < 1e-2
    joint_gelu = gelu_mediation["mediation_summary"]["reference_derivative_product"]
    assert gelu_mediation["state_count"] == 32
    assert max(item["max_abs_product_vs_reference"] for item in gelu_mediation["override_checks"]) == 0.0
    assert joint_gelu["exact_write_count"] == 32
    assert joint_gelu["residual_norm_ratio_aggregate"] <= 1e-9
    assert "conditions" in rotary and mm["cases"]
    assert attention_input_probe["summary"]["all_input_replacements_observed"]
    assert all(
        row["full_qproj_grad_repair_l2"] > 0.0
        for row in attention_input_probe["rows"]
    )
    assert attention_scores_probe["summary"]["all_score_replacements_observed"]
    assert all(
        row["full_qproj_grad_repair_l2"] > 0.0
        for row in attention_scores_probe["rows"]
    )
    assert attention_statistics_probe["summary"]["all_statistics_replacements_observed"]
    assert all(
        row["full_qproj_grad_repair_l2"] > 0.0
        for row in attention_statistics_probe["rows"]
    )
    assert attention_carrier_probe["summary"]["all_carrier_right_replacements_observed"]
    assert all(
        row["full_qproj_grad_repair_l2"] > 0.0
        for row in attention_carrier_probe["rows"]
    )
    assert rotary_probe["status"] == "COMPLETE"
    assert all(row["tl_math_vs_libdevice"]["exact"] for row in rotary_probe["rows"])
    assert rotary_natural["status"] == "COMPLETE_NATURAL_INTERMEDIATE_PROBE"
    tolerance = rotary_natural.get("final_store_tolerance", {"max_abs": 0.0078125, "relative_l2": 2.0e-5})
    assert all(
        row["final_output0_alignment"]["max_abs"] <= tolerance["max_abs"]
        and row["final_output1_alignment"]["max_abs"] <= tolerance["max_abs"]
        and row["final_output0_alignment"]["relative_l2"] <= tolerance["relative_l2"]
        and row["final_output1_alignment"]["relative_l2"] <= tolerance["relative_l2"]
        for row in rotary_natural["rows"]
    )
    assert rotary_factorial["status"] == "COMPLETE_SOURCE_FACTORIAL_NATURAL_OPERANDS"
    assert rotary_factorial["summary"]["trig_fp64"]["write_effect_rms_mean"] == 0.0
    assert rotary_factorial["summary"]["muladd_fp32"]["aligned_write_mean"] < 0.0
    assert rotary_factorial["summary"]["full_fp32"]["aligned_write_mean"] < 0.0
    assert rmsnorm["status"] == "COMPLETE_SCOPED_NATURAL_TRAINING_SOURCE_CASE"
    assert [item["model"] for item in rmsnorm["models"]] == ["OLMoE-1B-7B-0125", "Qwen3-1.7B"]
    assert gemma_conv["status"] == "COMPLETE"
    assert gemma_conv["summary"]["state_count"] == 26
    assert gemma_conv["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_conv["status"] == "COMPLETE"
    assert mamba_conv["summary"]["state_count"] == 32
    assert mamba_conv["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_conv["summary"]["confirmation_projection_interval_normal_95"][1] < 0.0
    assert mamba_state_output["status"] == "COMPLETE"
    assert mamba_state_output["summary"]["state_count"] == 32
    assert mamba_state_output["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_d_skip["status"] == "COMPLETE"
    assert mamba_d_skip["summary"]["state_count"] == 32
    assert mamba_d_skip["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert embedding["status"] == "COMPLETE"
    assert embedding["summary"]["state_count"] == 24
    assert embedding["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert gemma4_nll["status"] == "COMPLETE"
    assert gemma4_nll["summary"]["state_count"] == 24
    assert gemma4_nll["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert gemma4_softcap["status"] == "COMPLETE"
    assert gemma4_softcap["summary"]["state_count"] == 26
    assert gemma4_softcap["summary"]["confirmation_aligned_interval_normal_95"][1] < 0.0
    assert gemma4_softcap_logits["summary"]["aligned_positive"] == 26
    assert gemma4_softcap_logits["summary"]["aligned_negative"] == 0
    assert gemma4_row_reduction["status"] == "COMPLETE"
    assert gemma4_row_reduction["summary"]["state_count"] == 26
    assert gemma4_row_reduction["summary"]["aligned_negative"] == 13
    assert gemma4_row_reduction["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert qwen3_attention["status"] == "COMPLETE"
    assert qwen3_attention["summary"]["state_count"] == 16
    assert qwen3_attention["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert qwen3_attention_math["status"] == "COMPLETE"
    assert qwen3_attention_math["summary"]["state_count"] == 8
    assert qwen3_attention_math["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert mamba_softplus["status"] == "COMPLETE"
    assert mamba_softplus["state_count"] == 32
    assert mamba_softplus["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_transition["status"] == "COMPLETE"
    assert mamba_transition["state_count"] == 16
    assert mamba_transition["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_scan_source_cut["status"] == "COMPLETE"
    assert mamba_scan_source_cut["state_count"] == 16
    assert mamba_scan_source_cut["summary"]["fused_aligned_interval_normal_95"][1] < 0.0
    assert mamba_scan_source_cut["summary"]["cut_aligned_interval_normal_95"][1] < 0.0
    assert bert_attention_bias["status"] == "COMPLETE"
    assert bert_attention_bias["state_count"] == 32
    assert bert_attention_bias["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert bert_addmm["status"] == "COMPLETE_FIXED_REAL_TEXT_CPU_BOUNDARY"
    assert bert_addmm["state_count"] == 32
    assert bert_addmm["aligned_write_interval_normal_95"][1] < 0.0
    assert bert_score["state_count"] == 32
    assert bert_score["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert bert_value["state_count"] == 32
    assert bert_value["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert bert_nll["status"] == "COMPLETE"
    assert bert_nll["summary"]["state_count"] == 128
    assert bert_nll["summary"]["aligned_write_interval_normal_95"][0] > 0.0
    assert qwen3vl_position["status"] == "COMPLETE_NATURAL_POSITION_INTERPOLATION_TRAINING_PROBE"
    assert qwen3vl_position["image_count"] == 16
    assert all(float(row["aligned_ratio"]) < 0.0 for row in qwen3vl_position["rows"])
    assert bert_pooler_tanh["status"] == "COMPLETE_NATURAL_TANH_POOLER_PROBE"
    assert bert_pooler_tanh["state_count"] == 32
    assert bert_pooler_tanh["summary"]["confirmation_projection_interval_normal_95"][0] > 0.0
    assert bert_pooler_tanh_dense["summary"]["confirmation_projection_interval_normal_95"][0] < 0.0 < bert_pooler_tanh_dense["summary"]["confirmation_projection_interval_normal_95"][1]
    assert olmoe_router["status"] == "COMPLETE"
    assert olmoe_router["state_count"] == 16
    assert olmoe_router["summaries"]["write"]["confirmation_aligned_interval_normal_95"][1] < 0.0
    assert granite_router["status"] == "COMPLETE"
    assert granite_router["summary"]["state_count"] == 32
    assert granite_router["summary"]["selected_set_changed_states"] == 0
    assert sum(value < 0.0 for value in granite_router["summary"]["confirmation_aligned_write_values"]) == 15
    assert gemma4_audio["status"] == "COMPLETE_NATURAL_AUDIO_LANGUAGE_BOUNDARY"
    assert gemma4_audio["summary"]["state_count"] == 16
    assert gemma4_audio["summary"]["write_effect_rms_mean"] > 0.5
    assert gemma4_audio_attention_softcap["status"] == "COMPLETE_NATURAL_AUDIO_ATTENTION_SOFTCAP_BOUNDARY"
    assert gemma4_audio_attention_softcap["summary"]["state_count"] == 16
    assert gemma4_audio_attention_softcap["summary"]["bf16_write_effect_rms_mean"] > 1.9
    assert gemma4_audio_attention_softmax["status"] == "COMPLETE_NATURAL_AUDIO_ATTENTION_SOFTMAX_BOUNDARY"
    assert gemma4_audio_attention_softmax["summary"]["state_count"] == 16
    assert gemma4_audio_attention_softmax["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert gemma4_audio_subsample["status"] == "COMPLETE_NATURAL_AUDIO_CONV_BOUNDARY"
    assert gemma4_audio_subsample["summary"]["state_count"] == 16
    assert gemma4_audio_subsample["summary"]["confirmation_aligned_write_ratio_of_sums"] < -0.3
    assert deberta_relative["status"] == "COMPLETE"
    assert deberta_relative["summary"]["state_count"] == 32
    assert deberta_relative["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert deberta_relative_last["status"] == "COMPLETE"
    assert deberta_relative_last["summary"]["state_count"] == 32
    assert deberta_relative_last["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert bloom_alibi["status"] == "COMPLETE"
    assert bloom_alibi["summary"]["state_count"] == 16
    assert bloom_alibi["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert bloom_alibi_last["status"] == "COMPLETE"
    assert bloom_alibi_last["summary"]["state_count"] == 16
    assert bloom_alibi_last["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert liger_jsd["state_count"] == 32
    assert liger_jsd["aligned_confirmation_interval_95_approx"][0] > 0.0
    assert bert_softmax["state_count"] == 32
    assert bert_softmax["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert gptneo_gelu["status"] == "COMPLETE"
    assert gptneo_gelu["summary"]["state_count"] == 32
    assert gptneo_gelu["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert rwkv_time_decay["status"] == "COMPLETE_NATURAL_RWKV_TIME_MIX_BOUNDARY"
    assert rwkv_time_decay["summary"]["state_count"] == 16
    assert rwkv_time_decay["summary"]["modes"]["decay_fp32"]["write_aligned_interval_normal_95"][1] < 0.0
    assert rwkv_time_decay["summary"]["modes"]["value_fp32"]["write_aligned_interval_normal_95"][0] < 0.0 < rwkv_time_decay["summary"]["modes"]["value_fp32"]["write_aligned_interval_normal_95"][1]
    assert all(
        record["summary"]["confirmation_aligned_write_interval_normal_95"][1] < 0.0
        for record in (rwkv_receptance_sigmoid, rwkv_receptance_sigmoid_method)
    )
    assert embedding_gradient_materialization["status"] == "COMPLETE"
    assert embedding_gradient_materialization["summary"]["state_count"] == 16
    assert embedding_gradient_materialization["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert fused_embedding_partial["status"] == "COMPLETE"
    assert fused_embedding_partial["summary"]["state_count"] == 16
    assert fused_embedding_partial["summary"]["variants"]["bf16_partial"]["aligned_write_interval_normal_95"][1] < 0.0
    assert fused_embedding_partial["summary"]["variants"]["fp64_reduction"]["aligned_write_interval_normal_95"][0] < 0.0 < fused_embedding_partial["summary"]["variants"]["fp64_reduction"]["aligned_write_interval_normal_95"][1]

    rows = [
        ("adamw8bit_moment_quantization", "end-to-end declared-protocol causal chain", "mean bias as the unique mediator of loss", "an intervention independently changing mean, variance, and coordinate/time structure while preserving the others"),
        ("liger_fused_linear_ce_dw_accumulation", "FP32 addition-order source is reproduced on disjoint length-64 and length-256 banks; the actual torch AdamW parameter write is measured in original coordinates, and independent 32-calibration/64-confirmation with-replacement empirical-bank runs at both lengths support a positive signed write projection mean under their declared iid assumptions", "generalization beyond the declared empirical banks, causal sufficiency for a long-run loss consequence, and any stronger claim about which natural input structure prevents cancellation", "a genuinely held-out or different-model state distribution plus a warm-moment or multi-step comparison; a stronger cancellation-condition claim requires dynamic-range or operand-conditioned inputs; Kahan is not a certified update repair"),
        ("liger_fused_linear_jsd_distillation", "real-text masked-student/unmasked-teacher JSD boundary has a positive aligned write interval for compiled Liger versus explicit FP32 JSD/CE", "independent teacher/student checkpoint generalization, a population mean certificate, and a multi-step distillation quality consequence", "an independent teacher/student pair and a predeclared multi-step distillation endpoint"),
        ("softmax_saved_state_backward", "saved-score/statistic inconsistency causes row-mass defect on every retained row; a 1024-step declared trajectory confirms the restoration reaches gradient/write and produces trajectory non-identity", "natural population mean update bias or material loss consequence", "independent state-bank sampling with a predeclared loss endpoint and consistent-saved-state restoration propagated through gradient and parameter write"),
        ("attention_state_to_q_projection_region", "the semantic S_bwd boundary is causally closed; eight-state path-preserving probes isolate the generated softmax-backward output feeding bmm_75, its upstream gradient input U, the pre-softmax score input and the consumed max/normalizer statistics, and each intervention changes q_proj gradient", "one unique source for the whole multi-source attention region", "separate path-preserving interventions for the upstream key and residual-stream producers, plus any interaction decomposition among the four softmax-call inputs; further aggregation cannot identify a unique whole-region source"),
        ("mm_gemm_output_and_accumulation", "case-specific same-operands decompositions plus retained conditional source-debiased ensembles: the named Qwen branches and Mamba local/zero-moment branches show centered repair residuals with candidate-minus-repair downstream bias", "one universal MM root, an absolute high-precision downstream reference, or a common natural-population mean bias", "an independently sampled natural-state confirmation or absolute downstream reference; a new shared factorial is needed only for a stronger cross-case MM mechanism claim"),
        ("silu_backward_evaluation", "the joint product/factor path-preserving intervention closes the reviewed source boundary, and a 128-draw with-replacement probe supports a negative aligned parameter-write mean under the declared DeepSeek trajectory bank", "an unrestricted natural-training population mean, a separate population-quality claim, and any finer instruction decomposition below the joint boundary", "a bank representing the unrestricted natural training population and a declared quality endpoint; no further offline inference can upgrade the empirical-bank result"),
        ("fused_rope_position_scaling", "natural Ministral replay plus a four-arm source factorial shows that FP64 trigonometric inputs with native BF16 rotation are exact identity, while FP32 multiply/add followed by BF16 write-back alone reproduces the native-to-FP32 signed write effect on all eight real vision states", "population mean-bias generalization, a loss consequence, and moments separately from the step counter", "an independent held-out image/state distribution and a fixed-step-counter optimizer-state comparison if either stronger claim is needed"),
        ("gemma_gelu_backward_evaluation", "same-bank source-choice-sensitive response for the bound GELU backward endpoint; native tanh and fused multiply-add agree through local/gradient/update/write while explicit exponential tanh changes the profile; the corrected path-preserving joint derivative/product-factor intervention reproduces the native/reference write effect exactly on all 32 retained states; a 128-draw with-replacement probe supports a negative aligned write mean under the declared Gemma-4 trajectory bank", "a finer instruction-level decomposition below the jointly sufficient boundary, an unrestricted natural-input mean bias, and any multi-step quality consequence", "a bank representing the unrestricted natural training population plus a declared quality endpoint; a finer instruction intervention is optional refinement and not required for the reviewed source closure"),
        ("gptneo_gelu_native_fp32_materialization", "native GPT-Neo NewGELUActivation versus the same-input explicit FP32 tanh-GELU with original-dtype write-back has a stable negative aligned parameter-write effect across 32 real text states; the native BF16 expression matches its explicit native formula exactly", "external generalization beyond the declared GPT-Neo checkpoint/text bank and a multi-step loss consequence", "an independent GPT-Neo state bank or checkpoint and a declared multi-step training endpoint"),
        ("gemma_rms_feature_reduction_order", "the tested FP32 feature-reduction order is a tiny or exact-identity source control with no confirmed direction", "a different RMS source being responsible for the earlier region effect", "a new predeclared RMS intervention; no further inference is available for this tested order"),
        ("granite_router_topk_selection", "the tested equal-score tie-order variant leaves selected experts, values, gradients, writes and loss unchanged", "all selection implementations being safe or every non-tie/NaN case being equivalent", "a separately declared changed-selection or non-tie intervention"),
        ("granite_moe_expert_contribution_order", "reversing FP32 expert-contribution accumulation changes local/gradient values and a new with-replacement empirical-bank run gives a positive signed parameter-write projection mean under its declared iid assumptions", "generalization beyond the declared empirical bank, a warm-state or multi-step effect, and any training-quality consequence", "a genuinely held-out or different-model state distribution plus a warm-moment or multi-step comparison"),
        ("rmsnorm_cast_materialization", "two real decoder training graphs show the same native RMSNorm cast-before-weight versus FP32-weight-multiply choice produces a nonzero aligned parameter-write effect across 26 states per model, while the cast-before-weight negative control is exact", "an additive vector mean, a population guarantee beyond the declared fixed suites, and any loss or long-run quality consequence", "an independent natural-state confirmation with a predeclared training endpoint, plus a path-preserving intervention if a lower-level source distinction is needed"),
        ("bert_layernorm_compiled_materialization", "one real BERT-tiny checkpoint shows compiled native LayerNorm agrees with the explicit FP32 formula while the eager native path produces an aligned parameter-write effect across 32 document-derived windows", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a unique compiler instruction inside the compiled/eager boundary", "an independently sampled or held-out checkpoint/state bank and a path-preserving compiler/materialisation intervention if a lower-level source distinction is needed"),
        ("bert_embedding_sum_materialization", "one real BERT-tiny checkpoint shows native word/position/token-type embedding addition versus an otherwise identical FP32 three-term sum produces a stable negative aligned parameter-write effect across 32 document-derived windows", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level instruction cause inside the embedding block", "an independently sampled or held-out checkpoint/state bank and a path-preserving embedding-sum intervention if a lower-level source distinction is needed"),
        ("granite_residual_addition_materialization", "one real Granite MoE checkpoint shows native attention residual addition versus an otherwise identical FP32 residual-plus-scaled-output addition produces a stable negative aligned parameter-write effect across 32 document-derived windows", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level instruction cause inside the residual-addition boundary", "an independently sampled or held-out checkpoint/state bank and a path-preserving residual-addition intervention if a lower-level source distinction is needed"),
        ("granite_moe_gate_product_materialization", "one real Granite MoE checkpoint shows native activation(gate)*up multiplication versus an otherwise identical FP32 product produces stable negative aligned scaling and a positive held-out write projection across 32 document-derived windows", "a population or cross-checkpoint claim, a loss-quality consequence, or a lower-level instruction cause inside the gate-times-up product", "an independently sampled or held-out checkpoint/state bank and a multi-step training endpoint; a lower-level claim requires a path-preserving product intervention"),
        ("granite_moe_output_gate_materialization", "one real Granite MoE checkpoint shows native expert-output*routing-gate multiplication versus an otherwise identical FP32 product produces stable negative aligned scaling across 32 document-derived windows", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level instruction cause inside the output-gate product", "an independently sampled or held-out checkpoint/state bank and a multi-step training endpoint; a lower-level claim requires a path-preserving product intervention"),
        ("gemma3_vision_patch_convolution", "one real Gemma-3 image/text bank shows native BF16 patch convolution versus same-input FP32 accumulation produces a large, consistently negative aligned parameter-write scaling; the additive held-out direction remains unconfirmed", "a population or cross-checkpoint claim, an additive vector mean, or a loss-quality consequence", "an independently sampled image/text bank or second checkpoint, followed by a declared training endpoint if quality impact is needed"),
        ("mamba_causal_conv_accumulation", "one real Mamba text bank shows native causal depthwise convolution versus same-input FP32 accumulation produces a negative held-out write direction and stable negative aligned scaling", "a population or cross-checkpoint claim, a multi-step loss consequence, and whether the native backend is a Triton kernel in the tested environment", "an independent Mamba bank or second checkpoint, backend identity capture, and a declared multi-step training endpoint"),
        ("mamba_state_output_contraction_materialization", "one real Mamba text bank shows native state-to-C contraction versus same-input FP32 contraction produces stable negative aligned write scaling", "a population or cross-checkpoint claim, an additive vector mean, a multi-step loss consequence, and whether the native backend is a Triton kernel in the tested environment", "an independent Mamba bank or second checkpoint, backend identity capture, and a declared multi-step training endpoint"),
        ("mamba_d_skip_materialization", "one real Mamba text bank shows native residual D*hidden_states product versus same-input FP32 product produces stable negative aligned write scaling", "a population or cross-checkpoint claim, an additive vector mean, a multi-step loss consequence, and whether the native backend is a Triton kernel in the tested environment", "an independent Mamba bank or second checkpoint, backend identity capture, and a declared multi-step training endpoint"),
        ("mamba_z_gate_materialization", "one real Mamba text bank shows native scan-output*SiLU(z) product versus same-input FP32 product produces stable negative aligned write scaling", "a population or cross-checkpoint claim, an additive vector mean, a multi-step loss consequence, and whether the native backend is a Triton kernel in the tested environment", "an independent Mamba bank or second checkpoint, backend identity capture, and a declared multi-step training endpoint"),
        ("deepseek_embedding_backward_accumulation", "one real DeepSeek/Qwen3 text bank shows native repeated-token embedding backward accumulation versus same-input FP32 index-add produces a stable negative aligned parameter-write scaling while forward loss is identical", "a population or cross-checkpoint claim, an additive vector mean, or a multi-step loss-quality consequence", "an independently sampled text bank or second checkpoint, followed by a declared multi-step training endpoint if quality impact is needed"),
        ("gemma4_causal_nll_loss_evaluation", "one real Gemma-4 E2B text bank holds logits fixed and compares native cross-entropy with explicit FP32 shifted log-softmax/gather; the NLL boundary produces a stable negative aligned parameter-write scaling", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level fused-cross-entropy instruction cause", "an independently sampled text bank or second checkpoint, followed by a declared multi-step training endpoint; a lower-level claim requires a path-preserving fused-loss intervention"),
        ("gemma4_final_logit_softcap_materialization", "one real Gemma-4 E2B text bank holds the hidden state fixed and compares native BF16 final-logit soft-cap arithmetic with an FP32 reference; the 13-state confirmation half has a negative aligned lm_head write interval", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level soft-cap instruction cause", "an independently sampled text bank or second checkpoint, followed by a declared multi-step training endpoint"),
        ("gemma4_audio_output_projection_materialization", "sixteen real LibriSpeech waveforms are passed through the same Gemma-4 audio-language graph while only the audio output projection changes from native BF16 to FP32 accumulation; all states have nonzero loss and projection-write differences, with mean write RMS about 88.8% and an entirely negative eight-state confirmation aligned interval", "a population or cross-checkpoint claim, a long-run audio-language quality consequence, or a Triton kernel identity claim", "an independently sampled real-speech/text bank and a declared multi-step audio-language endpoint; runtime identity capture is needed for a kernel-level claim"),
        ("gemma4_audio_attention_softcap_materialization", "sixteen real speech states pass through the same Gemma-4 audio-language graph while only the layer-0 attention-logit softcap arithmetic changes between native FP32, BF16-rounded, and FP64 variants; every state has a nonzero q_proj gradient/write effect and the BF16/FP64 write RMS averages about 193%/215%, while the held-out aligned intervals cross zero", "a population directional mean-bias claim or a multi-step audio-language quality consequence", "an independently sampled speech bank with a predeclared directional endpoint and a multi-step audio-language run"),
        ("gemma4_audio_attention_softmax_probability_materialization", "sixteen real speech states pass through the same Gemma-4 audio-language graph while only the layer-0 attention probabilities change from FP32 retention to BF16 materialization before value contraction; the q_proj write RMS averages about 1.77% and the confirmation aligned interval is strictly negative", "a population or audio-family-wide claim, a multi-step audio-language quality consequence, or a lower-level attention-kernel instruction claim", "an independently sampled speech bank or checkpoint, a declared multi-step endpoint, and runtime identity if a kernel-level claim is needed"),
        ("gemma4_audio_lightconv_glu_product_materialization", "sixteen real speech states pass through the same Gemma-4 audio-language graph while only the LightConv1d GLU gate-times-value product changes from native dtype to FP32 before the original-dtype write-back; mean write RMS is about 328.2% and the eight-state confirmation aligned interval is entirely negative", "a population or cross-checkpoint claim, a long-run audio-language quality consequence, or a Triton kernel identity claim", "an independently sampled real-speech/text bank and a declared multi-step audio-language endpoint; runtime identity capture is needed for a kernel-level claim"),
        ("gemma4_audio_lightconv_depthwise_conv_backward_accumulation", "sixteen real speech states pass through the same Gemma-4 audio-language graph while only the LightConv1d depthwise causal-convolution boundary changes from native Conv1d backward to explicit FP32 window multiply-accumulate; mean write RMS is about 31.4% and the eight-state confirmation ratio-of-sums aligned scaling is +1.975%, positive after deleting any one confirmation state", "a population or cross-checkpoint claim, a statewise universal sign, a long-run audio-language quality consequence, or a Triton kernel identity claim", "an independently sampled real-speech/text bank and a declared multi-step audio-language endpoint; runtime identity capture is needed for a kernel-level claim"),
        ("gemma4_audio_subsampling_convolution_materialization", "sixteen real speech states pass through the same Gemma-4 audio-language graph while only layer-1 audio subsampling Conv2d changes from native BF16 accumulation to explicit FP32 accumulation with BF16 writeback; mean write RMS is about 125.0% and the eight-state confirmation ratio-of-sums aligned scaling is -35.98%, negative after deleting any one confirmation state", "a population or cross-checkpoint claim, a statewise universal sign, a long-run audio-language quality consequence, or a Triton kernel identity claim", "an independently sampled real-speech/text bank and a declared multi-step audio-language endpoint; runtime identity capture is needed for a kernel-level claim"),
        ("deberta_disentangled_relative_attention_materialization", "two DeBERTa-v3-small layers on 32 real text windows each show native c2p/p2c relative-position score products versus FP32 score products with entirely negative confirmation aligned-write intervals", "a population or cross-checkpoint claim, a pretrained MLM quality consequence, or a unique lower-level instruction inside the relative-attention boundary", "an independently sampled state bank or task-trained head; runtime identity capture is needed for a lower-level kernel claim"),
        ("bloom_alibi_attention_materialization", "two Bloom-560m layers on 16 real text windows each show native ALiBi score materialization versus explicit FP32 QK-plus-ALiBi evaluation with entirely negative confirmation aligned-write intervals", "a population or cross-checkpoint claim, or a multi-step loss-quality consequence", "an independently sampled state bank or second checkpoint and a declared multi-step training endpoint"),
        ("qwen3_attention_sdpa_eager_backend", "same-input path-preserving eager versus SDPA attention boundary on a real Qwen3 text bank produces a stable negative aligned q_proj write scaling; a component probe shows FP32 score contraction has nearly the same aligned magnitude but does not reproduce the full SDPA effect vector", "which lower-level fused accumulation/materialization combination is sufficient, a population claim, or a loss-quality consequence", "a path-preserving intervention on the score accumulation/fusion boundary judged by per-state effect-vector overlap, followed by an independent state bank if a population claim is needed"),
        ("mamba_softplus_materialization", "same sequential Mamba recurrence with only delta-softplus precision changed produces a stable negative aligned dt_proj write scaling on two real text banks", "a population claim, full additive vector mean, or a multi-step loss consequence", "an independent state bank or second checkpoint and a declared multi-step training endpoint"),
        ("mamba_discrete_transition_materialization", "same sequential Mamba recurrence with only the exp(A*delta) exponent-argument materialisation changed produces stable negative aligned A_log write scaling on a real text bank", "a population claim, full additive vector mean, or a multi-step loss consequence", "an independent state bank or second checkpoint and a declared multi-step training endpoint"),
        ("mamba_fused_selective_scan_reassociation", "actual mamba_ssm fused layer-3 selective scan versus a same-layer explicit recurrence source cut, with convolution, projections, and downstream path held fixed, reproduces a negative signed aligned parameter-write component across 16 real text states; the full effect vector is not identical", "a full multi-layer Mamba claim, a natural population guarantee, an additive vector mean, or a multi-step loss consequence", "an independently sampled Mamba bank or checkpoint and a declared multi-step endpoint; a broader claim requires source cuts for the other recurrence boundaries"),
        ("rwkv_time_decay_materialization", "same RWKV sequential WKV recurrence with only the time_decay exponential promoted to FP32 reproduces the all-FP32 aligned write effect, while a value-only FP32 control does not", "a population claim, a multi-step loss consequence, or a claim about custom RWKV CUDA kernels (the tested model uses the sequential implementation)", "an independent RWKV text bank or checkpoint and a declared multi-step training endpoint; runtime identity is needed for a custom-kernel claim"),
        ("rwkv_receptance_sigmoid_materialization", "same RWKV attention block with only receptance sigmoid evaluation promoted to FP32 has a strictly negative confirmation aligned-write interval on two independent real-text banks", "a population claim, a multi-step loss consequence, or a claim about custom RWKV CUDA kernels", "an independent RWKV checkpoint or held-out bank and a declared multi-step training endpoint; runtime identity is needed for a custom-kernel claim"),
        ("deepseek_embedding_gradient_materialization", "same captured real embedding cotangent with repeated-index accumulation held fixed shows a strictly negative aligned write effect when final gradient retention changes from FP32 to BF16; the effect is only about 3e-8 RMS", "a practical training consequence, population generalization, or a claim that this dominates repeated-token accumulation", "an independent text bank if reproducibility of this micro-effect is important; a quality endpoint is not warranted without a larger effect"),
        ("deepseek_fused_embedding_nll_partial_materialization", "same generated fused embedding/NLL/normalization backward call boundary with only partial terms rounded to BF16 gives a negative aligned write interval, while the FP64 reduction control is near identity", "a claim that this explains all sources of the fused region, a population result, or a long-run quality consequence", "an independent text bank for confirmation and a declared multi-step endpoint if practical impact is claimed"),
        ("deepseek_fused_embedding_nll_reduction_order", "same generated fused embedding/NLL/normalization backward call boundary with FP32 arithmetic held fixed and only the reduction element order reversed gives a small positive aligned write interval, while the FP64 control is near identity", "a population result, a practical quality consequence, or a claim that this micro-effect explains the larger fused-region difference", "an independent text bank for confirmation; no quality endpoint is justified unless a materially larger effect is found"),
        ("qwen3vl_position_interpolation", "one real Qwen3-VL image bank shows native-dtype four-tap learned-position interpolation versus FP32 weighted summation produces stable negative aligned visual-position write scaling under the real image/text loss path", "a population or cross-checkpoint claim, an additive vector mean, or a long-run quality consequence", "an independent multimodal image bank or second checkpoint followed by a declared multi-step training endpoint"),
        ("bert_pooler_tanh_materialization", "one real BERT-tiny document-derived bank shows native pooler tanh versus FP32 tanh followed by the original write produces a positive held-out additive gradient direction at the embedding carrier; a same-boundary pooler-dense contrast crosses zero", "a population or cross-checkpoint claim, an all-parameter effect, a loss-quality consequence, or a unique lower-level tanh instruction", "an independently sampled BERT bank or second checkpoint, plus a path-preserving local-output probe if an instruction-level source is needed"),
        ("olmoe_router_expert_accumulation", "one real OLMoE text bank shows native BF16 repeated-destination expert accumulation versus FP32 accumulation followed by one BF16 write produces a stable negative aligned parameter-write effect while routing and expert selection remain unchanged", "a population or cross-checkpoint claim, an additive vector mean, a warm-state effect, or a multi-step loss consequence", "an independent OLMoE bank or checkpoint and a declared multi-step training endpoint"),
        ("granite_moe_router_score_materialization", "one real Granite text bank with native top-k indices frozen shows native model-dtype router score/weight materialization versus FP32 score/weight evaluation produces a stable negative aligned router-write effect", "a population or cross-checkpoint claim, a natural full-route claim without the frozen-selection qualification, or a multi-step loss consequence", "an independent Granite bank or checkpoint and a declared multi-step training endpoint; a full-route decomposition must separately account for changed expert selection"),
        ("gemma4_rms_row_reduction_order", "one real Gemma-4 text bank holds the RMSNorm graph and write-back fixed while reversing only the FP32 row square-sum reduction order; the confirmation half has 13/13 negative aligned input-layernorm writes", "a population or cross-checkpoint claim, a released-kernel internal-order claim, or a loss-quality consequence", "runtime source binding or generated-kernel comparison, followed by an independent state bank or declared multi-step training endpoint"),
        ("bert_fused_addmm_bias_materialization", "one real BERT-tiny CPU text bank holds the dense weight multiply and downstream path fixed while replacing native BF16 fused addmm with split matmul plus FP32 bias materialization; the confirmation half has a negative aligned LayerNorm-weight write interval and factorial controls isolate the fused-linear path", "a population or cross-checkpoint claim, a CUDA claim, an additive vector mean, or a loss-quality consequence", "an independently sampled BERT bank or checkpoint and a CUDA-specific confirmation; a quality claim needs a declared multi-step endpoint"),
        ("bert_attention_softmax_materialization", "one real BERT-tiny CUDA text bank holds Q/K/V, mask, model weights, and downstream path fixed while replacing native BF16 softmax evaluation with FP32 softmax followed by BF16 write-back; the confirmation half has a negative aligned query-write interval", "a population or cross-checkpoint claim, an additive vector mean, or a loss-quality consequence", "an independently sampled BERT bank or a declared multi-step training endpoint; a lower-level claim requires a path-preserving kernel/source intervention"),
        ("bert_attention_score_materialization", "one real BERT-tiny CUDA FP16 text bank holds Q/K operands, mask, model weights, value path and classifier fixed while replacing native FP16 QK score multiplication with FP32 multiplication followed by FP16 write-back; the confirmation half has a negative aligned query-write interval", "a population or cross-checkpoint claim, an additive vector mean, or a loss-quality consequence", "an independently sampled BERT bank or a declared multi-step training endpoint; a lower-level claim requires runtime identity capture for the score kernel"),
        ("bert_attention_value_materialization", "one real BERT-tiny CUDA FP16 text bank holds scores, softmax weights, Q/K path, mask, model weights and classifier fixed while replacing native FP16 attention-probability/value multiplication with FP32 multiplication followed by FP16 write-back; the confirmation half has a negative aligned query-write interval", "a population or cross-checkpoint claim, an additive vector mean, or a loss-quality consequence", "an independently sampled BERT bank or a declared multi-step training endpoint; a lower-level claim requires runtime identity capture for the value-contraction kernel"),
        ("bert_nll_loss_evaluation", "one real BERT-tiny 128-window document bank holds compiled logits fixed while native cross-entropy is replaced by explicit FP32 log-softmax/gather; the confirmation aligned decoder-write interval is strictly positive", "a population or cross-checkpoint claim, an additive vector mean, a loss-quality consequence, or a lower-level fused-loss instruction cause", "an independently sampled BERT bank or checkpoint and a declared multi-step endpoint; runtime identity is needed for a lower-level kernel claim"),
    ]
    active_prefixes = {
        "gemma_rms_feature_reduction_order",
        "granite_router_topk_selection",
    }
    active_rows = [row for row in rows if row[0] not in active_prefixes]
    payload = {
        "schema": "kernel-analyzer-root-cause-exhaustion-audit-v1",
        "status": "NO_ADDITIONAL_ROOT_CAUSE_CONCLUSION_FROM_RETAINED_EVIDENCE",
        "meaning": "Every stronger listed conclusion requires a new distinguishing observation; missing sufficient statistics are not treated as a negative result.",
        "counting_rule": "The active count excludes the two explicitly designated negative controls; audited_row_count retains them for calibration and boundary accounting.",
        "audited_row_count": len(rows),
        "active_problem_group_count": len(active_rows),
        "negative_control_count": len(rows) - len(active_rows),
        "rows": [{
            "problem_group": group,
            "deducible_now": current,
            "not_deducible_from_retained_data": unavailable,
            "missing_observation": missing,
            "further_offline_inference": False,
        } for group, current, unavailable, missing in rows],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
