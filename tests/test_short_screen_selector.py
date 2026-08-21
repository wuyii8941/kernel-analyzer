import json

from scripts.select_short_screen_escalations import classify


def _row(case_id, status):
    return {
        "status": "COMPLETE",
        "protocol": {"projection_seed": 7, "null_draws": 100, "prefix_growth_mode": "after_warmup"},
        "cases": [{
            "case_id": case_id,
            "status": status,
            "steps": 8,
            "projection": {"dimension": 64},
            "screen_rule": {
                "requires_observed_above_sign_flip_95": True,
                "requires_lag1_and_at_least_two_positive_lags": True,
            },
            "observed_amplification": 1.3,
            "sign_flip_null": {"upper_95": 1.1},
        }],
    }


def test_risk_escalates_and_null_abstains():
    assert classify(_row("risk", "RISK_CANDIDATE"), "risk.json")[0]["decision"] == "ESCALATE_EXACT_TRAJECTORY"
    assert classify(_row("null", "NULL_LIKE_OR_UNRESOLVED"), "null.json")[0]["decision"] == "ABSTAIN_NO_ESCALATION"


def test_incomplete_screen_fails_closed():
    payload = {"status": "PARTIAL", "cases": []}
    assert classify(payload, "partial.json")[0]["decision"] == "ABSTAIN_INVALID"
