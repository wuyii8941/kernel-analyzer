import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_report():
    return json.loads(
        (ROOT / "results/property/root_cause_closure_v1/natural_bias_consequence_audit_v1.json").read_text()
    )


def test_report_separates_source_bias_and_loss_claims():
    report = load_report()
    assert report["status"] == "CONSERVATIVE_CLAIM_SCOPE_AUDIT"
    assert report["summary"]["active_problem_groups"] == 14
    assert report["summary"]["history_aligned_frequency_supported"] == 1
    assert report["summary"]["loss_improvement_confirmed"] == 1
    assert report["summary"]["negative_controls"] == 2
    assert len(report["rows"]) == 16
    assert report["retired_records"][0]["active_mainline"] is False


def test_negative_controls_are_not_natural_bias_cases():
    rows = {row["problem_group"]: row for row in load_report()["rows"]}
    assert rows["gemma_rms_feature_reduction_order"]["natural_bias_status"] == "NEGATIVE_CONTROL_NO_CONFIRMED_FIXED_SUITE_MEAN_DIRECTION"
    assert rows["granite_router_topk_selection"]["natural_bias_status"] == "IDENTITY_ON_OBSERVED_SUITE_NO_MEAN_EFFECT"


def test_claim_scope_uses_current_mean_evidence():
    rows = {row["problem_group"]: row for row in load_report()["rows"]}
    for group in (
        "liger_fused_linear_ce_dw_accumulation",
        "mm_gemm_output_and_accumulation",
        "softmax_saved_state_backward",
        "granite_moe_expert_contribution_order",
    ):
        assert rows[group]["natural_bias_status"].startswith("SUPPORTED_PROJECTED_MEAN")
