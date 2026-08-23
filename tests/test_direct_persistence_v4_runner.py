from scripts.prepare_direct_persistence_v4_followups import validate_pool
from scripts.run_direct_persistence_v4_screen import classify


PROTOCOL = {
    "name": "Cold-start AdamW Direct Persistence Screen",
    "optimizer": "AdamW",
    "moment_initialization": "zero at the start, then evolved normally",
}


def short_payload(status="NULL_LIKE_OR_UNRESOLVED"):
    return {
        "status": "COMPLETE",
        "direct_persistence_protocol": PROTOCOL,
        "cases": [{
            "case_id": "case",
            "status": status,
            "steps": 16,
            "projection": {"dimension": 64},
            "sign_flip_null": {"draws": 2000, "upper_95": 1.1},
            "screen_rule": {
                "requires_observed_above_sign_flip_95": True,
                "requires_lag1_and_at_least_two_positive_lags": True,
                "requires_late_prefix_growth": True,
            },
            "observed_amplification": 1.2,
        }],
    }


def test_v4_screen_uses_fail_closed_decisions():
    assert classify(short_payload("RISK_CANDIDATE"), "x.json")[0]["decision"] == "ESCALATE"
    assert classify(short_payload(), "x.json")[0]["decision"] == "NO_ESCALATION_UNDER_SHORT_SCREEN"
    assert classify({"status": "COMPLETE", "cases": []}, "x.json")[0]["decision"] == "ABSTAIN"


def test_v4_screen_rejects_missing_optimizer_identity():
    payload = short_payload()
    del payload["direct_persistence_protocol"]
    assert classify(payload, "x.json")[0]["decision"] == "ABSTAIN"


def test_heldout_pool_requires_frozen_identity_and_no_revealed_labels():
    row = {
        "case_id": "new",
        "model": "model",
        "implementation_class": "new_impl",
        "endpoint": "endpoint",
        "sequence_length": 128,
        "state_order": ["s0", "s1"],
        "state_bank_digest": "digest",
        "parameter_coordinate_digest": "coords",
        "repair": "repair",
    }
    pool = {
        "schema": "kernel-analyzer-direct-persistence-v4-heldout-pool-v1",
        "status": "FROZEN_BEFORE_REVEAL",
        "rows": [row],
    }
    assert validate_pool(pool)["status"] == "READY"
    row["label"] = False
    assert validate_pool(pool)["status"] == "ABSTAIN_INVALID_POOL"
