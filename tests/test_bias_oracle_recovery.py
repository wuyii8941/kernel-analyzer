from kernel_analyzer.bias_oracle_recovery import (
    RecoveryDisposition,
    RecoveryPrediction,
    compare_recovery,
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
