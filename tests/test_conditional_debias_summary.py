from scripts.build_conditional_debias_summary import ROLES, build


def _layer(role):
    repair = role == "repair_local_residual"
    return {
        "status": "CONDITIONAL_CENTERED" if repair else "CONDITIONAL_BIAS",
        "cross_state_ratio": 0.0 if repair else 0.5,
        "bootstrap_lower": -0.01 if repair else 0.4,
        "bootstrap_upper": 0.01 if repair else 0.6,
        "estimand": "REPAIR_RESIDUAL" if repair else "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
        "reference": "local" if repair else "ensemble",
    }


def test_compact_summary_preserves_estimand_boundary():
    roles = {
        role: {
            "condition_count": 16,
            "status_counts": {
                "CONDITIONAL_CENTERED" if role == "repair_local_residual"
                else "CONDITIONAL_BIAS": 16,
            },
        }
        for role in ROLES
    }
    source = {
        "status": "COMPLETE_CONDITIONAL_DEBIAS_CONFIRMATION",
        "conditional_debias_enabled": True,
        "candidate_id": "candidate",
        "architecture": "qwen",
        "carrier_parameter": "weight",
        "arms": ["ROUNDING_ONLY"],
        "bindings": {},
        "conditional_debias_summary": {
            "ROUNDING_ONLY": {
                "status": "LOCAL_SOURCE_CONDITIONALLY_DEBIASED_WITH_SYSTEMATIC_CANDIDATE_F_B_EFFECT",
                "roles": roles,
                "absolute_downstream_repair_bias": "NOT_IDENTIFIABLE_MISSING_EXACT_REFERENCE",
            }
        },
        "states": [
            {
                "state_id": f"s{i}",
                "sham_exact": True,
                "arms": {
                    "ROUNDING_ONLY": {
                        "repeats": 8,
                        "conditional_debias": {
                            "layers": {role: _layer(role) for role in ROLES}
                        },
                    }
                },
            }
            for i in range(16)
        ],
    }
    summary = build(source)
    assert summary["global_direction_used_as_gate"] is False
    assert summary["arms"]["ROUNDING_ONLY"]["aggregate"][
        "absolute_downstream_repair_bias"
    ].startswith("NOT_IDENTIFIABLE")
