import json
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_forbids_sparse_case_promotion():
    protocol = json.loads((ROOT / "results/coverage/directional_bias_protocol.json").read_text())
    assert protocol["schema"] == "kernel-analyzer-directional-bias-protocol-v3"
    assert protocol["tiers"]["T1_LOCAL"]["sparse_coordinates_may_assign_case"] is False
    assert protocol["tiers"]["T3_COHERENT"]["posthoc_coordinate_selection"] is False
    assert set(protocol["tiers"]["T3_COHERENT"]["hypotheses"]) == {"raw", "relative", "factor"}
    assert protocol["optional_stopping_after_failed_frozen_confirmation"] is False


def test_existing_cases_separate_root_and_semantic_region_counts():
    audit = json.loads((ROOT / "results/coverage/existing_case_reaudit.json").read_text())
    rows = {row["case"]: row for row in audit["rows"]}
    assert audit["schema"] == "kernel-analyzer-flash-style-reaudit-v1"
    assert audit["counts"]["strict_flash_style_cases"] == 6
    assert audit["counts"]["strict_root_arithmetic_op_pass"] == 4
    assert audit["counts"]["strict_semantic_region_pass"] == 2
    assert audit["counts"]["composite_flash_style_cases"] == 0
    assert audit["counts"]["cross_state_concrete_mechanism_passes"] == 2
    assert audit["counts"]["completed_negative_cases"] == 2
    assert audit["counts"]["needs_reconfirmation"] == 0
    assert audit["counts"]["cross_operator_property_claims"] == 0
    assert rows["seq128_lm_head_input_vjp_mm"]["flash_style"]["verdict"] == \
        "PASS_FLASH_STYLE_CASE"
    assert rows["seq128_lm_head_input_vjp_mm"]["generalizable_bias"]["verdict"] == \
        "FAIL_CROSS_STATE_NONCOHERENT"
    for name in ("liger_fused_linear_ce_dw", "phi4_seq64_lmhead_dx_mm"):
        assert rows[name]["flash_style"]["verdict"] == "PASS_FLASH_STYLE_CASE"
        assert rows[name]["generalizable_bias"]["verdict"] == \
            "PASS_CROSS_STATE_CONCRETE_MECHANISM"
    assert rows["layer23_qproj_attention_state_region"]["flash_style"]["verdict"] == \
        "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
    assert rows["mamba64_layer0_input_proj_output"]["flash_style"]["verdict"] == \
        "PASS_FLASH_STYLE_CASE"
    assert rows["mamba64_layer0_input_proj_output"]["generalizable_bias"]["verdict"] == \
        "FAIL_CROSS_STATE_CARRIER_NONCOHERENT"
    assert rows["qwen128_layer0_vproj_output"]["flash_style"]["verdict"] == \
        "FAIL_DIRECTIONAL_ACCUMULATION"
    assert rows["qwen128_layer0_vproj_output"]["flash_style"]["gates"] \
        ["F4_PAIRED_TRAJECTORY"]["status"] == "FAIL"
    assert rows["liger_fused_linear_ce_dw"]["property_positive_eligible"] is True
    assert rows["layer23_qproj_attention_state_region"]["property_positive_eligible"] is False


def test_l23_semantic_region_attribution_is_closed_but_not_single_kernel():
    case = json.loads((
        ROOT / "results/coverage/cases/l23_qproj_attention_state_region.json"
    ).read_text())
    assert case["status"] == "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
    assert all(case["gates"].values())
    assert case["classification"]["valid_flash_style_case"] is True
    assert case["classification"]["single_kernel_property_eligible"] is False
    assert case["trajectory"]["projection_strictly_increases_each_step"] is True


def test_case_protocol_separates_trajectory_from_cross_state_generalization():
    protocol = json.loads(
        (ROOT / "results/coverage/case_classification_protocol.json").read_text()
    )
    assert set(protocol["tracks"]) == {"FLASH_STYLE_CASE", "GENERALIZABLE_BIAS"}
    assert "does_not_require" in protocol["tracks"]["FLASH_STYLE_CASE"]
    assert protocol["preserves_original_measurements"]
    assert protocol["prospective_preregistration_claimed"] is False


def test_reaudit_does_not_promote_missing_trajectory_evidence():
    audit = json.loads((ROOT / "results/coverage/existing_case_reaudit.json").read_text())
    rows = {row["case"]: row for row in audit["rows"]}
    softmax = rows["qwen128_layer27_softmax_saved_state"]
    assert softmax["flash_style"]["verdict"] == \
        "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
    assert all(gate["status"] == "PASS" for gate in softmax["flash_style"]["gates"].values())
    assert softmax["generalizable_bias"]["verdict"] == "FAIL_CROSS_STATE_NONCOHERENT"
    silu = rows["qwen3vl_silu_backward_decomposition"]
    assert silu["flash_style"]["gates"]["F2_CAUSAL_REPAIR"]["status"] == "PASS"
    assert silu["flash_style"]["verdict"] == "FAIL_DIRECTIONAL_ACCUMULATION"
    assert silu["flash_style"]["gates"]["F3_REAL_CARRIER"]["status"] == "PASS"
    assert silu["flash_style"]["gates"]["F4_PAIRED_TRAJECTORY"]["status"] == "FAIL"


def test_every_retained_invocation_has_v3_bias_pipeline():
    with gzip.open(ROOT / "results/coverage/gap_matrix.json.gz", "rt") as handle:
        matrix = json.load(handle)
    protocol = json.loads((ROOT / "results/coverage/directional_bias_protocol.json").read_text())
    assert matrix["schema"] == "kernel-analyzer-exhaustive-gap-matrix-v3"
    assert len(matrix["rows"]) == matrix["denominator"] == 111529
    assert matrix["active_full_step_denominator"] == 9269 + 56411 + 8223 + 12044
    assert matrix["retained_paused_denominator"] == 25582
    assert matrix["directional_bias_protocol_sha256"] == protocol["protocol_sha256"]
    for row in matrix["rows"]:
        pipeline = row["directional_bias_pipeline"]
        assert pipeline["protocol_sha256"] == protocol["protocol_sha256"]
        assert set(pipeline) == {
            "protocol_sha256", "T1_LOCAL", "T2_CAUSAL", "T3_RAW",
            "T3_RELATIVE", "T3_FACTOR", "T4_ACCUMULATION"
        }


def test_valid_nontriton_candidate_queue_is_exhaustive():
    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    assert queue["schema"] == "kernel-analyzer-bias-candidate-queue-v2"
    assert queue["candidate_count"] == 83
    assert queue["candidate_count"] == queue["screen_positive_denominator"]["nontriton"]
    assert queue["candidate_disposition"]["precision_only_cast_null"] == 38
    assert queue["candidate_disposition"]["full_coordinate_direction_rejected"] == 3
    assert queue["candidate_disposition"]["pending_exhaustive_full_coordinate_and_fb_binding"] == 42
    assert queue["unselected_screen_positives_disposition"].startswith("NONE_FOR_VALID_NONTRITON")


def test_current_nontriton_disposition_reconciles_all_83_candidates():
    current = json.loads((
        ROOT / "results/coverage/live_contrast_dispositions.json"
    ).read_text())
    assert current["candidate_count"] == 83
    assert sum(current["counts"].values()) == 83
    assert current["counts"]["REJECTED_PRECISION_ONLY_REFERENCE_CAST_NULL"] == 38
    assert current["status"] == "COMPLETE_LIVE_FULL_COORDINATE_FOLLOWUP"
    assert current["counts"]["REJECTED_BY_CORRECTED_FULL_COORDINATE_DIRECTION"] == 38
    assert current["counts"]["REJECTED_FULL_COORDINATE_PILOT_NOT_DIRECTIONAL"] == 2
    assert current["counts"]["REJECTED_T1_SAMPLED_COORDINATE_FALSE_POSITIVE"] == 1
    assert current["counts"].get("PENDING_LIVE_FULL_COORDINATE_FOLLOWUP", 0) == 0
    assert current["counts"].get("T1_POSITIVE_PENDING_COMPLETE_CASE_GATES", 0) == 0
    assert current["counts"]["COMPLETE_BOUNDED_FLASH_STYLE_FB_BIAS_CASE"] == 3
    assert current["counts"]["COMPLETE_FB_CASE_REJECTED_DIRECTIONAL_ACCUMULATION"] == 1
