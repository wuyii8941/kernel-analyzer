import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_ledger(directory=None):
    path = (Path(directory) / "root_cause_closure_current.json" if directory
            else ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json")
    return json.loads(path.read_text())


def by_group(data, group_id):
    return next(row for row in data["rows"] if row["problem_group"] == group_id)


def test_current_root_cause_ledger_recomputes_and_keeps_open_branches(tmp_path):
    import os
    subprocess.run(
        [sys.executable, "scripts/build_current_root_cause_closure.py"],
        cwd=ROOT,
        check=True,
        env={**os.environ, "KA_OUTPUT_DIR": str(tmp_path)},
    )
    data = load_ledger(tmp_path)
    rows = data["rows"]
    assert len(rows) == len({row["problem_group"] for row in rows}) == 54
    assert data["summary"]["end_to_end_count"] == 1
    assert data["summary"]["negative_control_count"] == 2
    assert data["summary"]["active_case_count"] == 52
    assert data["summary"]["active_problem_group_count"] == 52
    assert data["summary"]["root_cause_unresolved_count"] == 0
    assert data["summary"]["active_source_or_local_closed_count"] == 52
    assert data["summary"]["root_cause_closed_final_count"] == 52
    assert data["summary"]["active_open_root_cause_count"] == 0
    assert data["summary"]["final_source_or_local_closed_count"] == 52
    assert data["summary"]["generalization_or_mean_open_count"] == 7
    assert len(data["active_problem_groups"]) == 52
    assert len(data["final_root_cause_groups"]) == 52
    assert len(data["open_root_cause_groups"]) == 0
    assert len(data["negative_controls"]) == 2
    assert {row["problem_group"] for row in data["open_root_cause_groups"]} == set()
    mamba_scan = by_group(data, "mamba_fused_selective_scan_reassociation")
    assert mamba_scan["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_scan["mean_bias_evidence"] == (
        "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    )
    assert {item["candidate"] for item in data["not_counted_candidates"]} == {
        "selected_nll_logsoftmax_backward",
        "gemma_softcapped_nll_backward",
        "triton_signature_nonisolated_region_pool",
        "gemma_rms_forward_normalized",
        "mamba_recurrence_fused_scan",
        "mainstream_generated_operator_scans",
            "mamba_softplus_backward",
            "external_liger_loss_and_execution_candidates",
            "indexed_accumulation_reduction_order",
        "gemma4_per_layer_input_gate_materialization",
            "deepseek_gqa_repeat_kv_backward",
            "qwen3_causal_mask_materialization",
            "qwen3_attention_contraction_expression",
                "qwen3_attention_output_layout_materialization",
                "deepseek_attention_output_layout_materialization",
            "bert_embedding_backward",
            "ministral3_patch_merger_layout",
            "bert_residual_addition_materialization",
        "gptneo_local_attention_mask_materialization",
        "gemma3_vision_embedding_addition_materialization",
        "gemma3_vision_projector_pool_materialization",
        "gemma3_vision_projector_pool_materialization_repeat",
        "gemma4_audio_attention_softmax_materialization",
        "bert_softmax_materialization",
        "olmoe_router_score_materialization",
        "granite_router_gate_materialization",
        "t5_relative_attention_bias_addition",
        "gptneo_embedding_position_addition_materialization",
        "rwkv_channel_mix_square_relu_materialization",
                    }
    assert data["not_counted_candidates"][0]["status"] == (
        "NOT_PROMOTED_RUNTIME_BINDING_INCOMPLETE"
    )
    gqa = next(
        item for item in data["not_counted_candidates"]
        if item["candidate"] == "deepseek_gqa_repeat_kv_backward"
    )
    assert gqa["status"] == "SCREENED_NOT_PROMOTED_EXACT_IDENTITY"
    assert all((ROOT / path).is_file() for path in gqa["evidence"] if path.endswith(".json"))
    fused = next(
        item for item in data["not_counted_candidates"]
        if item["candidate"] == "mamba_recurrence_fused_scan"
    )
    assert fused["status"] == "SCREENED_NOT_PROMOTED_MEAN_DIRECTION_UNCONFIRMED"
    assert all((ROOT / path).is_file() for path in fused["evidence"] if path.endswith(".json"))
    t5 = next(
        item for item in data["not_counted_candidates"]
        if item["candidate"] == "t5_relative_attention_bias_addition"
    )
    assert t5["status"] == "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH"
    assert all((ROOT / path).is_file() for path in t5["evidence"] if path.endswith(".json"))
    assert all(row["case_role"] == "ACTIVE_BIAS_CASE" for row in data["active_problem_groups"])
    jsd = by_group(data, "liger_fused_linear_jsd_distillation")
    assert jsd["closure"] == "SOURCE_CLOSED_REAL_TEXT_DISTILLATION_SCOPED_ALIGNED_BIAS"
    assert jsd["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_TEXT_BOUNDARY"
    softmax = by_group(data, "bert_attention_softmax_materialization")
    assert softmax["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert softmax["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_BOUNDARY"
    score = by_group(data, "bert_attention_score_materialization")
    assert score["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert score["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_FP16_BOUNDARY"
    value = by_group(data, "bert_attention_value_materialization")
    assert value["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert value["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_FP16_BOUNDARY"
    addmm = by_group(data, "bert_fused_addmm_bias_materialization")
    assert addmm["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert addmm["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CPU_BOUNDARY"
    bert_nll = by_group(data, "bert_nll_loss_evaluation")
    assert bert_nll["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert bert_nll["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT"
    assert bert_nll["derived"]["summary"]["aligned_write_interval_normal_95"][0] > 0.0
    audio = by_group(data, "gemma4_audio_output_projection_materialization")
    assert audio["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert audio["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE"
    audio_softcap = by_group(data, "gemma4_audio_attention_softcap_materialization")
    assert audio_softcap["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_TOTAL_EFFECT"
    assert audio_softcap["mean_bias_evidence"] == (
        "FIXED_SUITE_TOTAL_EFFECT_SOURCE_CLOSED_DIRECTION_NOT_CONFIRMED"
    )
    assert audio_softcap["derived"]["summary"]["state_count"] == 16
    assert audio_softcap["derived"]["summary"]["bf16_write_effect_rms_mean"] > 1.9
    assert audio_softcap["derived"]["summary"]["confirmation_bf16_aligned_interval"][0] < 0.0 < audio_softcap["derived"]["summary"]["confirmation_bf16_aligned_interval"][1]
    audio_softmax = by_group(data, "gemma4_audio_attention_softmax_probability_materialization")
    assert audio_softmax["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert audio_softmax["derived"]["summary"]["state_count"] == 16
    assert audio_softmax["derived"]["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert audio_softmax["derived"]["layer1_heterogeneity"]["summary"]["write_aligned_interval_normal_95"][0] < 0.0 < audio_softmax["derived"]["layer1_heterogeneity"]["summary"]["write_aligned_interval_normal_95"][1]
    audio_glu = by_group(data, "gemma4_audio_lightconv_glu_product_materialization")
    assert audio_glu["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert audio_glu["mean_bias_evidence"] == "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE"
    assert audio_glu["derived"]["summary"]["state_count"] == 16
    assert audio_glu["derived"]["summary"]["confirmation_aligned_write_interval_normal_95"][1] < 0.0
    audio_subsample = by_group(data, "gemma4_audio_subsampling_convolution_materialization")
    assert audio_subsample["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert audio_subsample["derived"]["summary"]["confirmation_aligned_write_ratio_of_sums_recomputed"] < -0.3
    deberta = by_group(data, "deberta_disentangled_relative_attention_materialization")
    assert deberta["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert len(deberta["derived"]["layers"]) == 2
    assert all(item["write_aligned_interval_normal_95"][1] < 0.0 for item in deberta["derived"]["layer_summaries"])
    bloom = by_group(data, "bloom_alibi_attention_materialization")
    assert bloom["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert len(bloom["derived"]["layers"]) == 2
    assert all(item["write_aligned_interval_normal_95"][1] < 0.0 for item in bloom["derived"]["layer_summaries"])
    assert all(row["case_role"] == "NEGATIVE_CONTROL" for row in data["negative_controls"])
    assert data["summary"]["coverage_collection_count"] == 2
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
        "gptneo_gelu_native_fp32_materialization",
        "gemma_rms_feature_reduction_order",
        "granite_router_topk_selection",
        "granite_moe_expert_contribution_order",
        "rmsnorm_cast_materialization",
        "bert_layernorm_compiled_materialization",
        "bert_embedding_sum_materialization",
        "bert_pooler_tanh_materialization",
        "granite_residual_addition_materialization",
        "granite_moe_gate_product_materialization",
        "granite_moe_output_gate_materialization",
        "gemma3_vision_patch_convolution",
        "deepseek_embedding_backward_accumulation",
        "gemma4_causal_nll_loss_evaluation",
        "gemma4_final_logit_softcap_materialization",
        "gemma4_rms_row_reduction_order",
        "qwen3_attention_sdpa_eager_backend",
        "mamba_softplus_materialization",
        "mamba_discrete_transition_materialization",
        "mamba_causal_conv_accumulation",
        "mamba_state_output_contraction_materialization",
        "mamba_d_skip_materialization",
        "mamba_z_gate_materialization",
            "qwen3vl_position_interpolation",
                "olmoe_router_expert_accumulation",
                "granite_moe_router_score_materialization",
                "liger_fused_linear_jsd_distillation",
                "bert_attention_softmax_materialization",
                "bert_attention_score_materialization",
                "bert_attention_value_materialization",
                "bert_fused_addmm_bias_materialization",
                    "gemma4_audio_output_projection_materialization",
                    "gemma4_audio_attention_softcap_materialization",
                    "gemma4_audio_attention_softmax_probability_materialization",
                    "gemma4_audio_lightconv_glu_product_materialization",
                    "gemma4_audio_lightconv_depthwise_conv_backward_accumulation",
                    "gemma4_audio_subsampling_convolution_materialization",
                    "deberta_disentangled_relative_attention_materialization",
        "bloom_alibi_attention_materialization",
        "rwkv_time_decay_materialization",
        "rwkv_receptance_sigmoid_materialization",
        "deepseek_embedding_gradient_materialization",
        "deepseek_fused_embedding_nll_partial_materialization",
            "deepseek_fused_embedding_nll_reduction_order",
            "mamba_fused_selective_scan_reassociation",
            "bert_nll_loss_evaluation",
                        }
    assert by_group(data, "adamw8bit_moment_quantization")["closure"].startswith(
        "END_TO_END"
    )
    assert by_group(data, "softmax_saved_state_backward")["closure"].endswith(
        "NATURAL_BIAS_OPEN"
    )
    fused_embedding = by_group(data, "deepseek_fused_embedding_nll_partial_materialization")
    assert fused_embedding["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert fused_embedding["derived"]["summary"]["state_count"] == 16
    assert fused_embedding["derived"]["summary"]["variants"]["bf16_partial"]["aligned_write_interval_normal_95"][1] < 0.0
    fused_order = by_group(data, "deepseek_fused_embedding_nll_reduction_order")
    assert fused_order["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert fused_order["derived"]["summary"]["state_count"] == 16
    assert fused_order["derived"]["summary"]["variants"]["fp32_reverse_order"]["aligned_write_interval_normal_95"][0] > 0.0
    assert by_group(data, "gemma_gelu_backward_evaluation")["closure"] == (
        "SOURCE_CLOSED_JOINT_DERIVATIVE_PRODUCT_MEAN_BIAS_OPEN"
    )
    gptneo = by_group(data, "gptneo_gelu_native_fp32_materialization")
    assert gptneo["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert gptneo["derived"]["state_count"] == 32
    assert gptneo["derived"]["aligned_write_interval_normal_95"][1] < 0.0
    rwkv = by_group(data, "rwkv_time_decay_materialization")
    assert rwkv["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert rwkv["derived"]["state_count"] == 16
    assert rwkv["derived"]["decay_fp32"]["write_aligned_interval_normal_95"][1] < 0.0
    assert rwkv["derived"]["value_fp32"]["write_aligned_interval_normal_95"][0] < 0.0 < rwkv["derived"]["value_fp32"]["write_aligned_interval_normal_95"][1]
    rwkv_sigmoid = by_group(data, "rwkv_receptance_sigmoid_materialization")
    assert rwkv_sigmoid["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert len(rwkv_sigmoid["derived"]["banks"]) == 2
    assert all(
        bank["summary"]["confirmation_aligned_write_interval_normal_95"][1] < 0.0
        for bank in rwkv_sigmoid["derived"]["banks"]
    )
    embedding_cast = by_group(data, "deepseek_embedding_gradient_materialization")
    assert embedding_cast["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert embedding_cast["derived"]["summary"]["state_count"] == 16
    assert embedding_cast["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    rmsnorm = by_group(data, "rmsnorm_cast_materialization")
    assert rmsnorm["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert [item["model"] for item in rmsnorm["derived"]["models"]] == [
        "OLMoE-1B-7B-0125",
        "Qwen3-1.7B",
    ]
    gemma_rms = rmsnorm["derived"]["gemma4_confirmation"]
    assert gemma_rms["states"] == 16
    assert gemma_rms["summary"]["write_effect_rms_over_reference"]["mean"] > 0.6
    assert gemma_rms["summary"]["write_direction"]["aligned_interval_normal_95"][1] < 0.0
    bert = by_group(data, "bert_layernorm_compiled_materialization")
    assert bert["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert bert["derived"]["eager_32_state_summary"]["state_count"] == 32
    assert bert["derived"]["fp32_control_summary"]["write_effect_rms_mean"] == 0.0
    bert_embedding = by_group(data, "bert_embedding_sum_materialization")
    assert bert_embedding["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert bert_embedding["derived"]["summary"]["state_count"] == 32
    assert bert_embedding["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    bert_tanh = by_group(data, "bert_pooler_tanh_materialization")
    assert bert_tanh["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ADDITIVE_BIAS"
    assert bert_tanh["derived"]["embedding_state_count"] == 32
    assert bert_tanh["derived"]["embedding_summary"]["confirmation_projection_interval_normal_95"][0] > 0.0
    assert bert_tanh["derived"]["dense_summary"]["confirmation_projection_interval_normal_95"][0] < 0.0 < bert_tanh["derived"]["dense_summary"]["confirmation_projection_interval_normal_95"][1]
    granite_residual = by_group(data, "granite_residual_addition_materialization")
    assert granite_residual["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert granite_residual["derived"]["summary"]["state_count"] == 32
    assert granite_residual["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    granite_gate = by_group(data, "granite_moe_gate_product_materialization")
    assert granite_gate["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert granite_gate["derived"]["summary"]["state_count"] == 32
    assert granite_gate["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert granite_gate["derived"]["summary"]["confirmation_projection_interval_normal_95"][0] > 0.0
    granite_output_gate = by_group(data, "granite_moe_output_gate_materialization")
    assert granite_output_gate["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert granite_output_gate["derived"]["summary"]["state_count"] == 32
    assert granite_output_gate["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    granite_router = by_group(data, "granite_moe_router_score_materialization")
    assert granite_router["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert granite_router["derived"]["summary"]["state_count"] == 32
    assert granite_router["derived"]["summary"]["selected_set_changed_states"] == 0
    assert granite_router["derived"]["confirmation_negative_count"] == 15
    assert granite_router["derived"]["confirmation_aligned_interval_normal_95"][1] < 0.0
    conv = by_group(data, "gemma3_vision_patch_convolution")
    assert conv["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert conv["derived"]["summary"]["state_count"] == 26
    embedding = by_group(data, "deepseek_embedding_backward_accumulation")
    assert embedding["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert embedding["derived"]["summary"]["state_count"] == 24
    assert conv["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    gemma4_nll = by_group(data, "gemma4_causal_nll_loss_evaluation")
    assert gemma4_nll["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert gemma4_nll["derived"]["summary"]["state_count"] == 24
    assert gemma4_nll["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    gemma4_softcap = by_group(data, "gemma4_final_logit_softcap_materialization")
    assert gemma4_softcap["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert gemma4_softcap["derived"]["summary"]["state_count"] == 26
    assert gemma4_softcap["derived"]["summary"]["confirmation_aligned_interval_normal_95"][1] < 0.0
    assert gemma4_softcap["derived"]["logit_gradient_summary"]["aligned_positive"] == 26
    assert gemma4_softcap["derived"]["logit_gradient_summary"]["aligned_negative"] == 0
    gemma4_row_reduction = by_group(data, "gemma4_rms_row_reduction_order")
    assert gemma4_row_reduction["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert gemma4_row_reduction["derived"]["summary"]["state_count"] == 26
    assert gemma4_row_reduction["derived"]["summary"]["aligned_negative"] == 13
    assert gemma4_row_reduction["derived"]["summary"]["aligned_positive"] == 0
    assert gemma4_row_reduction["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    mamba_softplus = by_group(data, "mamba_softplus_materialization")
    assert mamba_softplus["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_softplus["derived"]["state_summary"]["write_effect_rms_mean"] > 0.09
    assert mamba_softplus["derived"]["state_summary"]["aligned_write_interval_normal_95"][1] < 0.0
    mamba_conv = by_group(data, "mamba_causal_conv_accumulation")
    assert mamba_conv["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_conv["derived"]["summary"]["state_count"] == 32
    assert mamba_conv["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert mamba_conv["derived"]["summary"]["confirmation_projection_interval_normal_95"][1] < 0.0
    mamba_state_output = by_group(data, "mamba_state_output_contraction_materialization")
    assert mamba_state_output["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_state_output["derived"]["summary"]["state_count"] == 32
    assert mamba_state_output["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    mamba_d_skip = by_group(data, "mamba_d_skip_materialization")
    assert mamba_d_skip["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_d_skip["derived"]["summary"]["state_count"] == 32
    assert mamba_d_skip["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    mamba_z_gate = by_group(data, "mamba_z_gate_materialization")
    assert mamba_z_gate["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert mamba_z_gate["derived"]["summary"]["state_count"] == 32
    assert mamba_z_gate["derived"]["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert "mamba_z_gate_materialization_wikitext_32" in mamba_z_gate["evidence"][0]
    position = by_group(data, "qwen3vl_position_interpolation")
    assert position["closure"] == "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS"
    assert position["derived"]["image_count"] == 16
    assert position["derived"]["aligned_ratio_normal_95_interval"][1] < 0.0
    rope = by_group(data, "fused_rope_position_scaling")
    assert rope["closure"] == "NATURAL_SOURCE_FACTORIAL_ROOT_CLOSED_MEAN_BIAS_SCOPE_OPEN"
    assert rope["derived"]["saved_source_contract"]["status"] == "COMPLETE_SAVED_SOURCE_AUDIT"
    assert rope["derived"]["all_trigonometric_variants_exact"] is True
    assert rope["derived"]["probe_record"].endswith("rotary_arithmetic_source_probe_v4.json")
    assert all(row["tl_math_vs_libdevice_exact"] for row in rope["derived"]["rows"])
    natural_rope = rope["derived"]["natural_intermediate_probe"]
    assert natural_rope["status"] == "COMPLETE_NATURAL_INTERMEDIATE_PROBE"
    assert natural_rope["final_output0_within_tolerance"]
    assert natural_rope["final_output1_within_tolerance"]
    qwen3_rope = rope["derived"]["qwen3_standard_rope_confirmation"]
    assert qwen3_rope["summary"]["state_count"] == 16
    assert qwen3_rope["summary"]["write_aligned_interval_normal_95"][1] < 0.0
    assert qwen3_rope["summary"]["heldout_write_projection_negative"] == 8
    vision_rope = rope["derived"]["ministral_vision_rope_confirmation"]
    assert vision_rope["summary"]["aligned_write_interval_normal_95"][1] < 0.0
    assert vision_rope["summary"]["aligned_sign_counts_confirmation"] == {
        "positive": 0,
        "negative": 4,
    }
    factorial = rope["derived"]["natural_source_factorial"]
    assert factorial["state_count"] == 8
    assert factorial["summary"]["trig_fp64"]["write_effect_rms_mean"] == 0.0
    assert factorial["summary"]["muladd_fp32"]["aligned_write_mean"] < 0.0
    attention = by_group(data, "attention_state_to_q_projection_region")
    assert attention["closure"] == "SEMANTIC_REGION_ROOT_CLOSED_MULTI_SOURCE_DECOMPOSITION"
    assert attention["derived"]["single_kernel_attribution"] is False
    softmax_probe = attention["derived"]["direct_softmax_kernel_output_probe"]
    assert softmax_probe["output_is_bmm_left_input"]
    assert softmax_probe["output_buffer_is_bmm_left_input"]
    assert softmax_probe["qproj_gradient_changed_all_states"]
    assert len(softmax_probe["qproj_gradient_repair_l2"]) >= 4
    softmax_input_probe = attention["derived"]["direct_softmax_kernel_input_probe"]
    assert softmax_input_probe["input_replacement_observed"]
    assert softmax_input_probe["qproj_gradient_changed_all_states"]
    assert len(softmax_input_probe["qproj_gradient_repair_l2"]) >= 4
    softmax_scores_probe = attention["derived"]["direct_softmax_scores_probe"]
    assert softmax_scores_probe["scores_replacement_observed"]
    assert softmax_scores_probe["qproj_gradient_changed_all_states"]
    assert len(softmax_scores_probe["qproj_gradient_repair_l2"]) >= 4
    softmax_statistics_probe = attention["derived"]["direct_softmax_statistics_probe"]
    assert softmax_statistics_probe["statistics_replacement_observed"]
    assert softmax_statistics_probe["qproj_gradient_changed_all_states"]
    assert len(softmax_statistics_probe["qproj_gradient_repair_l2"]) >= 4
    carrier_probe = attention["derived"]["direct_bmm_key_carrier_probe"]
    assert carrier_probe["carrier_replacement_observed"]
    assert carrier_probe["qproj_gradient_changed_all_states"]
    assert len(carrier_probe["qproj_gradient_repair_l2"]) >= 4
    cuts = attention["derived"]["nested_source_cuts"]
    assert cuts["key_materialization_record"]["row_count"] == 32
    assert set(cuts["local_mlp_fraction_by_layer"]) == {"24", "25", "26", "27"}
    assert set(cuts["attention_vjp_fraction_by_layer"]) == {"24", "25", "26", "27"}
    assert cuts["terminal_upstream_logit_fraction"] > 0.7
    assert cuts["terminal_lm_head_vjp_fraction"] == 0.0
    silu = by_group(data, "silu_backward_evaluation")
    assert silu["closure"] == "SOURCE_CLOSED_JOINT_PRODUCT_FACTOR_MEAN_BIAS_OPEN"
    assert silu["derived"]["same_local_operands"] is True
    assert silu["derived"]["source_attribution"] == "CHECKED_FUNCTION_AST_AND_PRE_CALL_INPUTS"
    assert silu["derived"]["stages"]["PARAMETER_WRITE"]["total_effect_rms_range"][1] > 2e-4
    silu_factorial = silu["derived"]["same_bank_source_factorial"]
    assert silu_factorial["state_count"] == 32
    assert silu_factorial["effect_energy_equal_within_tolerance_all_stages"] is True
    intermediate = silu["derived"]["intermediate_source_probe"]
    assert intermediate["status"] == "COMPLETE_SYNTHETIC_INTERMEDIATE_SOURCE_PROBE"
    assert intermediate["first_difference_candidates"]["exp"] is False
    assert intermediate["first_difference_candidates"]["denominator"] is False
    assert intermediate["first_difference_candidates"]["sigmoid"] is True
    natural_intermediate = silu["derived"]["natural_intermediate_probe"]
    assert natural_intermediate["status"] == "COMPLETE_NATURAL_INTERMEDIATE_PROBE"
    assert natural_intermediate["first_difference_candidates"]["sigmoid"] is True
    assert natural_intermediate["first_difference_candidates"]["exp"] is False
    mediation = silu["derived"]["joint_product_factor_mediation"]
    assert mediation["state_count"] == 32
    assert mediation["summary"]["exact_write_count"] == 32
    assert mediation["summary"]["residual_norm_ratio_aggregate"] <= 1e-9
    assert mediation["factor_only_summary"]["effect_norm_ratio_aggregate"] > 0.9
    assert mediation["product_only_summary"]["effect_norm_ratio_aggregate"] < 0.6
    probe = silu["derived"]["declared_empirical_bank_mean_probe"]
    assert probe["population_test"]["independent_unit_count"] == 128
    assert probe["population_test"]["two_sided_95_interval"][1] < 0.0
    assert all(
        row["final_output_alignment"]["exact"]
        and row["final_derivative_alignment"]["exact"]
        for row in natural_intermediate["rows"]
    )
    liger = by_group(data, "liger_fused_linear_ce_dw_accumulation")["derived"]
    assert liger["64"]["state_count"] == 32
    assert liger["64"]["confirmation_positive_count"] == 14
    assert liger["256"]["confirmation_positive_count"] == 11
    assert liger["64"]["proposed_update_direction"]["confirmation_projection_mean"] > 0
    assert liger["64"]["population_mean_bias_decision"] == "NOT_ASSESSED_FIXED_SUITE"
    assert liger["length64_population_mean_confirmation"]["confirmation"]["one_sided_95_lower_bound"] > 0
    assert liger["length256_population_mean_confirmation"]["confirmation"]["one_sided_95_lower_bound"] > 0
    assert liger["training_1024"]["validation_loss_difference"]["1024"] == 0.0
    assert 0.70 < liger["length64_kahan_intervention"]["kahan_to_reverse_gradient_l2_ratio_mean"] < 0.80
    assert liger["length64_kahan_intervention"]["state_count"] == 32
    assert liger["length64_kahan_intervention"]["variants"]["ORIGINAL_MINUS_KAHAN"]["update_total_effect_rms"] > 0
    assert 1.3e-6 < liger["length64_actual_parameter_write_confirmation"]["fixed_suite_total_write_rms"] < 1.5e-6
    assert liger["length64_actual_parameter_write_confirmation"]["write_direction_diagnostic"]["confirmation_positive_count"] == 12
    assert set(liger["input_condition_probe"]["summaries"]) == {
        "natural", "reversed_labels", "constant_label"
    }


def test_gelu_source_evidence_is_recomputed_from_raw_records():
    data = load_ledger()
    evidence = by_group(data, "gemma_gelu_backward_evaluation")["derived"]["evidence"]
    assert evidence["natural_reference"]["state_count"] == 32
    assert evidence["natural_reference"]["confirmation_count"] == 16
    assert evidence["explicit_exponential_tanh"]["gradient_rms"] > 0.001
    assert evidence["explicit_exponential_tanh"]["update_rms"] > 0.04
    assert evidence["native_tanh_vs_fma_exact_in_recorded_profile"] is True
    natural_probe = evidence["natural_intermediate_probe"]
    assert natural_probe["status"] == "REJECTED_FINAL_ALIGNMENT"
    assert all(row["final_output_alignment"]["exact"] for row in natural_probe["rows"])
    assert any(not row["final_derivative_alignment"]["exact"] for row in natural_probe["rows"])
    component_probe = evidence["componentwise_intermediate_probe"]
    assert component_probe["status"] == "COMPLETE_COMPONENTWISE_PROBE"
    assert all(component_probe["derivative_alignment_by_component"].values())
    factorial = evidence["same_bank_source_factorial"]
    assert factorial["status"] == "COMPLETE_SAME_BANK_SOURCE_FACTORIAL_RECOMPUTATION"
    assert factorial["bank"]["state_count"] == 32
    probe = evidence["declared_empirical_bank_mean_probe"]
    assert probe["population_test"]["independent_unit_count"] == 128
    assert probe["population_test"]["two_sided_95_interval"][1] < 0.0
    assert factorial["bank"]["confirmation_count"] == 16
    mediation = evidence["path_preserving_joint_mediation"]
    assert mediation["state_count"] == 32
    assert mediation["override_checks"]
    assert max(item["max_abs_product_vs_reference"] for item in mediation["override_checks"]) == 0.0
    joint = mediation["mediation_summary"]["reference_derivative_product"]
    assert joint["exact_write_count"] == 32
    assert joint["residual_norm_ratio_aggregate"] <= 1e-9
    assert evidence["joint_product_derivative_source_closed"] is True
    assert factorial["comparisons"]["native_tanh_fused_multiply_add_profile_match"] is True
    assert factorial["comparisons"]["explicit_exponential_tanh_differs_from_native"] is True
    assert factorial["comparisons"]["explicit_to_native_effect_energy_ratio"]["PARAMETER_WRITE"] > 1.3
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
        "NATURAL_GENERALIZATION_OPEN"
    )
    assert granite["state_count"] == 16
    assert 0.00002 < granite["fixed_suite_total_rms"] < 0.00003
    assert granite["parameter_write_nonzero_confirmation_states"] == 3
    assert granite["parameter_write_nonzero_inner_product_signs"] == ["negative", "positive", "negative"]
    assert granite["parameter_write_direction_status"] == "CALIBRATION_DIRECTION_ORTHOGONAL_TO_ALL_CONFIRMATION_EFFECTS"
    assert granite["parameter_write_direction"]["confirmation_zero_count"] == 16
    assert granite["parameter_write_direction"]["zero_population_mean_proven"] is False
    assert granite["equivalence_decision"] == "EQUIVALENT"
    assert by_group(data, "gemma_rms_feature_reduction_order")["case_role"] == "NEGATIVE_CONTROL"
    assert by_group(data, "granite_router_topk_selection")["case_role"] == "NEGATIVE_CONTROL"
    assert by_group(data, "adamw8bit_moment_quantization")["case_role"] == "ACTIVE_BIAS_CASE"
    population = granite["independent_population_confirmation"]["parameter_write"]
    assert population["confirmation_projection_one_sided_95_lower"] > 0.0


def test_retired_gemma_square_sum_is_not_an_active_case_role():
    roles = json.loads((ROOT / "results/mainline_case_roles.json").read_text())
    retired = next(row for row in roles["cases"] if row["case_id"] == "gemma4_text128_scan_0037")
    assert retired["active_mainline"] is False
    assert retired["status"] == "RETIRED_EXECUTION_SOURCE_MISMATCH"


def test_mm_sources_remain_case_specific():
    data = load_ledger()
    deepseek_mm = by_group(data, "mm_gemm_output_and_accumulation")["derived"][
        "deepseek_fused_boundary_upstream_mm"
    ]
    assert deepseek_mm["state_count"] == 16
    assert all(
        variant["aligned_write_interval_normal_95"][1] < 0.0
        for variant in deepseek_mm["variants"].values()
    )
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
    qwen128 = mm["qwen_seq128_forward_8_output"]["conditional_debias"]["ROUNDING_ONLY"]
    assert qwen128["all_conditions_candidate_local_biased"] is True
    assert qwen128["all_conditions_candidate_zero_moment_update_biased"] is True
    assert qwen128["repair_residual_centered"] is True
    assert qwen128["condition_count"] == 16
    mamba = mm["mamba_seq64_forward_1_output"]["conditional_debias"]["JOINT"]
    assert mamba["all_conditions_candidate_local_biased"] is True
    assert mamba["all_conditions_candidate_zero_moment_update_biased"] is True
    assert mamba["repair_residual_centered"] is True
    assert "CONDITIONAL_F_B_EFFECT_CLOSED" in by_group(
        data, "mm_gemm_output_and_accumulation"
    )["closure"]


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
    assert families["family_count"] == 21
    assert len(families["families"]) == 21
    assert {row["family_id"] for row in families["families"]} == {
        "LINEAR", "NORMALIZATION", "SOFTMAX", "CROSS_ENTROPY", "SILU_GATING",
        "SOFTPLUS", "RECURRENCE", "ROTARY", "REDUCTION", "INDEXED_ACCUMULATION",
        "GELU", "CONVOLUTION", "EMBEDDING", "SELECTION", "OPTIMIZER_UPDATE",
        "FUSED_ATTENTION", "ELEMENTWISE_BIAS", "DATA_MOVEMENT_LAYOUT", "ELEMENTWISE",
        "FUSED_MIXED", "MASK_POSITION_CONTROL",
    }
    assert all(row["root_cause_status"] and row["interpretation"] for row in families["families"])


def test_open_root_cause_frontier_records_identifiability_blockers():
    data = load_ledger()
    frontier = data["root_cause_frontier"]
    assert frontier["open_problem_groups"] == []
    assert {
        row["family_id"] for row in frontier["measurement_only_operator_families"]
    } == {
        "DATA_MOVEMENT_LAYOUT",
        "ELEMENTWISE",
    }
    assert next(
        row for row in data["operator_family_frontier"]["families"]
        if row["family_id"] == "RECURRENCE"
    )["root_cause_status"] == "NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY"
    assert next(
        row for row in data["operator_family_frontier"]["families"]
        if row["family_id"] == "SOFTPLUS"
    )["root_cause_status"] == "NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY"
    assert next(
        row for row in data["operator_family_frontier"]["families"]
        if row["family_id"] == "INDEXED_ACCUMULATION"
    )["root_cause_status"] == "CONDITIONAL_OPERATOR_BIAS_CONFIRMED"
    assert {row["family_id"] for row in frontier["blocked_operator_families"]} == {
        "FUSED_MIXED"
    }
    assert {row["family_id"] for row in frontier["external_only_operator_families"]} == {
        "ELEMENTWISE_BIAS"
    }
