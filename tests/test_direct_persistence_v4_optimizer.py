from scripts.analyze_direct_persistence_v4_optimizer_state import analyze


def raw_payload():
    vectors = {}
    names = [
        "operator_output_error",
        "candidate_gradient",
        "repair_gradient",
        "candidate_update",
        "repair_update",
        "candidate_first_moment_before_step",
        "candidate_second_moment_before_step",
        "repair_first_moment_before_step",
        "repair_second_moment_before_step",
    ]
    for name in names:
        if name == "candidate_gradient":
            vectors[name] = [[1.0, 0.0] for _ in range(32)]
        elif name in {"repair_gradient", "repair_update", "repair_first_moment_before_step", "repair_second_moment_before_step"}:
            vectors[name] = [[0.0, 0.0] for _ in range(32)]
        elif name == "candidate_update":
            vectors[name] = [[-0.1, 0.0] for _ in range(32)]
        else:
            vectors[name] = [[0.0, 0.0] for _ in range(32)]
    return {
        "schema": "kernel-analyzer-bound-endpoint-raw-stage-v1",
        "status": "COMPLETE",
        "case_id": "toy",
        "state_ids": [str(i) for i in range(32)],
        "optimizer": {"name": "AdamW", "learning_rate": 0.1, "betas": [0.9, 0.95], "epsilon": 1e-8},
        "vectors": vectors,
    }


def test_optimizer_ablation_requires_all_raw_vectors():
    payload = raw_payload()
    del payload["vectors"]["candidate_gradient"]
    result = analyze(payload)
    assert result["status"] == "ABSTAIN_MISSING_OR_MALFORMED_CAPTURE"


def test_optimizer_ablation_reports_gradient_sgd_and_adamw_arms():
    result = analyze(raw_payload())
    assert result["status"] == "COMPLETE_SAME_STATE_OPTIMIZER_ABLATION"
    assert result["arms"]["gradient_difference"]["A32"] > 5.0
    assert result["arms"]["stateless_sgd"]["A32"] > 5.0
    assert result["arms"]["captured_adamw_moments"]["A32"] > 5.0
    assert result["arms"]["moment_reset_each_step"]["A32"] > 5.0
