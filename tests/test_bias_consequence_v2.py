from kernel_analyzer.bias_consequence import BiasConsequenceTrace


def _v(value, norm=None):
    return {"signed_value": value, "norm": abs(value) if norm is None else norm}


def test_consequence_is_separate_and_has_no_formation_stage():
    trace = BiasConsequenceTrace("toy", ["0", "1"])
    for step in ["0", "1"]:
        trace.add(step, local_increment=_v(1), feedback_increment=_v(0, 0),
                  actual_drift_increment=_v(1), final_drift=_v(1), recurrence_residual=0)
    result = trace.finalize()
    assert result["status"] == "COMPLETE"
    assert result["formation_point"] == "NOT_APPLICABLE"
    assert result["first_confirmed_bias_stage"] is None


def test_missing_step_fails_closed():
    trace = BiasConsequenceTrace("toy", ["0", "1"])
    trace.add("0", local_increment=_v(1), feedback_increment=_v(0, 0),
              actual_drift_increment=_v(1), final_drift=_v(1))
    assert trace.finalize()["status"] == "UNRESOLVED_MISSING_STEP"
