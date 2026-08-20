from kernel_analyzer.bias_oracle_recovery import (
    RecoveryDisposition,
    RecoveryPrediction,
    compare_recovery,
    combine_risk_witnesses,
    predict_crossfit_projection_risk,
    predict_population_coherence_risk,
    predict_reference_relative_risk,
    predict_response_rectification_risk,
    predict_source_fidelity_boundary,
)


def test_exact_optimizer_response_even_is_direct_risk():
    artifact = {
        "aggregate": {
            "all_forward_losses_equal": True,
            "optimizer_oddness_resultant_ratio": 0.7,
            "antithetic_update_persistence": 0.5,
        },
        "records": [
            {
                "response_even_l2": 2.0,
                "response_odd_l2": 1.0,
                "response_even_energy_on_sign_crossings": 0.9,
            },
            {
                "response_even_l2": 1.0,
                "response_odd_l2": 1.0,
                "response_even_energy_on_sign_crossings": 0.1,
            },
        ],
    }
    result = predict_response_rectification_risk("case", artifact)
    assert result.disposition == (
        RecoveryDisposition.DIRECT_RISK_RESPONSE_RECTIFICATION.value
    )
    assert result.direct_risk is True
    assert result.safe_release is False
    assert result.measurements[
        "energy_weighted_response_even_on_sign_crossings"
    ] == 0.74


def test_weak_response_never_becomes_safe_release():
    artifact = {
        "aggregate": {
            "all_forward_losses_equal": True,
            "optimizer_oddness_resultant_ratio": 0.001,
            "antithetic_update_persistence": 0.5,
        },
        "records": [{
            "response_even_l2": 1.0,
            "response_odd_l2": 1.0,
            "response_even_energy_on_sign_crossings": 0.1,
        }],
    }
    result = predict_response_rectification_risk("case", artifact)
    assert result.disposition == RecoveryDisposition.UNRESOLVED.value
    assert result.routed_for_exact_followup is True
    assert result.safe_release is False


def test_failed_natural_source_fidelity_abstains():
    result = predict_source_fidelity_boundary("case", {
        "validity_gates": {"natural_source_fidelity_every_condition": False},
        "minimum_natural_source_fidelity": 0.9,
    })
    assert result.disposition == (
        RecoveryDisposition.ABSTAIN_SOURCE_FIDELITY_FAILED.value
    )
    assert result.safe_release is False


def test_recovery_comparison_distinguishes_direct_from_routed():
    predictions = [
        RecoveryPrediction(
            "direct", RecoveryDisposition.DIRECT_RISK_SOURCE.value,
            True, False, False, "test", {}, "direct",
        ),
        RecoveryPrediction(
            "escalate", RecoveryDisposition.ESCALATE_MISSING_EVENT_MOMENT.value,
            False, True, False, "test", {}, "escalate",
        ),
    ]
    audit = compare_recovery(predictions, {
        "direct": "STRICT_POSITIVE",
        "escalate": "STRICT_POSITIVE",
    })
    assert audit["strict_direct_recall"] == 0.5
    assert audit["strict_routed_recall"] == 1.0
    assert audit["false_safe_count"] == 0


def test_population_coherence_is_one_way_risk_witness():
    aligned = {"complete_gram": [[1.0] * 4 for _ in range(4)]}
    hit = predict_population_coherence_risk("hit", aligned)
    assert hit.disposition == RecoveryDisposition.DIRECT_RISK_POPULATION_COHERENCE.value
    orthogonal = {"complete_gram": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]}
    miss = predict_population_coherence_risk("miss", orthogonal)
    assert miss.direct_risk is False
    assert miss.safe_release is False
    assert miss.routed_for_exact_followup is True


def test_reference_relative_prediction_recomputes_from_rows_not_old_status():
    artifact = {
        "status": "STALE_OLD_STATUS_MUST_NOT_BE_READ",
        "rows": [{
            "condition_id": str(index),
            "error_reference_dot": -0.001,
            "error_energy": 1.0,
            "reference_energy": 1.0,
        } for index in range(8)],
    }
    result = predict_reference_relative_risk("case", artifact)
    assert result.disposition == RecoveryDisposition.DIRECT_RISK_REFERENCE_RELATIVE.value


def test_crossfit_projection_separates_confirmed_and_sign_changing():
    hit = predict_crossfit_projection_risk(
        "hit", [0.01] * 8, basis_frozen_before_evaluation=True
    )
    assert hit.disposition == RecoveryDisposition.DIRECT_RISK_CROSSFIT_PROJECTION.value
    miss = predict_crossfit_projection_risk(
        "miss", [-0.01, 0.01] * 4, basis_frozen_before_evaluation=True
    )
    assert miss.direct_risk is False
    assert miss.safe_release is False


def test_multi_witness_oracle_is_risk_or_abstain_never_safe():
    hit = RecoveryPrediction(
        case_id="case", disposition="DIRECT", direct_risk=True,
        routed_for_exact_followup=False, safe_release=False,
        evidence_kind="MOVING_FRAME", measurements={}, reason="hit",
    )
    miss = RecoveryPrediction(
        case_id="case", disposition="UNRESOLVED", direct_risk=False,
        routed_for_exact_followup=True, safe_release=False,
        evidence_kind="POPULATION", measurements={}, reason="miss",
    )
    risk = combine_risk_witnesses("case", [miss, hit])
    assert risk.verdict == "DIRECTIONAL_RISK"
    assert risk.hit_witnesses == ("MOVING_FRAME",)
    assert risk.safe_release is False
    abstain = combine_risk_witnesses("case", [miss])
    assert abstain.verdict == "ABSTAIN"
    assert abstain.safe_release is False
