from kernel_analyzer.bias_consequence_v21 import BiasConsequenceTrace, ConsequenceStatus


def arm(value):
    return {"vector": [value, 0.0], "signed_value": value}


def test_four_counterfactual_recurrence_is_computed_by_library():
    trace = BiasConsequenceTrace("toy", ["0", "1"])
    for step, before, after in [("0", 0.0, 1.0), ("1", 1.0, 2.0)]:
        trace.add(
            step,
            candidate_at_candidate_state=arm(2.0),
            repair_at_candidate_state=arm(1.0),
            candidate_at_repair_state=arm(2.0),
            repair_at_repair_state=arm(1.0),
            drift_before=arm(before),
            drift_after=arm(after),
        )
    result = trace.finalize()
    assert result["status"] == ConsequenceStatus.COMPLETE
    assert result["rows"][0]["local_effect"]["signed_value"] == 1.0
    assert result["rows"][0]["feedback_effect"]["signed_value"] == 0.0
    assert result["rows"][0]["actual_drift_increment"]["signed_value"] == 1.0
    assert result["rows"][0]["recurrence_residual"]["norm"] == 0.0
    assert result["rows"][0]["local_effect"]["vector_digest"]
    assert result["rows"][0]["feedback_effect"]["vector_digest"]
    assert result["first_confirmed_bias_stage"] is None


def test_missing_arm_and_missing_drift_fail_closed():
    trace = BiasConsequenceTrace("toy", ["0"])
    trace.add(
        "0",
        candidate_at_candidate_state=arm(1.0),
        repair_at_candidate_state=arm(0.0),
        candidate_at_repair_state=arm(1.0),
        repair_at_repair_state=None,
        drift_before=arm(0.0),
        drift_after=arm(1.0),
    )
    assert trace.finalize()["status"] == ConsequenceStatus.INVALID_MALFORMED


def test_fake_recurrence_residual_is_not_an_input():
    trace = BiasConsequenceTrace("toy", ["0"])
    try:
        trace.add(
            "0",
            candidate_at_candidate_state=arm(1.0),
            repair_at_candidate_state=arm(0.0),
            candidate_at_repair_state=arm(1.0),
            repair_at_repair_state=arm(0.0),
            drift_before=arm(0.0),
            drift_after=arm(1.0),
            recurrence_residual=0.0,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("recurrence_residual must not be accepted by v2.1 API")


def test_counterfactual_digest_mismatch_fails_closed():
    trace = BiasConsequenceTrace("toy", ["0"])
    bad = {"vector": [1.0, 0.0], "signed_value": 1.0, "vector_digest": "wrong"}
    trace.add(
        "0",
        candidate_at_candidate_state=bad,
        repair_at_candidate_state=arm(0.0),
        candidate_at_repair_state=arm(1.0),
        repair_at_repair_state=arm(0.0),
        drift_before=arm(0.0),
        drift_after=arm(1.0),
    )
    assert trace.finalize()["status"] == ConsequenceStatus.INVALID_MALFORMED
