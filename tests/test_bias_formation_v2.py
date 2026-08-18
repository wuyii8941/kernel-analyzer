from kernel_analyzer.bias_formation import BiasFormationTrace, FormationPolicy


def _summary(ratio: float):
    return {
        "coordinate_count": 2,
        "vector_digest": "digest-" + str(ratio),
        "mean_vector_energy": ratio,
        "total_error_energy": 1.0,
        "u_statistic": ratio,
        "bootstrap_lower": ratio,
        "bootstrap_upper": ratio,
        "complete_gram": [[1.0, ratio], [ratio, 1.0]],
    }


def _trace(local, gradient, update, *, missing_digest=False):
    cal = ["c0", "c1"]
    conf = ["e0", "e1"]
    trace = BiasFormationTrace(
        "toy", cal, conf,
        FormationPolicy(min_states=2, bootstrap_samples=32),
    )
    for state in cal + conf:
        trace.add(
            state,
            "calibration" if state.startswith("c") else "confirmation",
            common_state_digest=None if missing_digest else "same-state-match-" + state,
            local_endpoint=local,
            parameter_gradient=gradient,
            effective_update=update,
        )
    return trace.finalize()


def test_missing_layer_fails_closed_and_does_not_impute_zero():
    result = _trace(_summary(0.0), None, _summary(0.0))
    assert result["status"] == "UNRESOLVED_MISSING_LAYER"
    assert result["first_confirmed_bias_stage"] is None


def test_upstream_unresolved_cannot_be_first_confirmed_bias():
    ambiguous = _summary(0.02)
    ambiguous["bootstrap_lower"] = 0.0
    ambiguous["bootstrap_upper"] = 0.1
    result = _trace(ambiguous, _summary(1.0), _summary(1.0))
    assert result["first_observed_biased_stage"] == "PARAMETER_GRADIENT"
    assert result["first_confirmed_bias_stage"] is None
    assert result["formation_point"] == "UNRESOLVED"


def test_nonfinite_and_malformed_are_distinct_fail_closed_states():
    nonfinite = _trace({"vector": [float("nan")]}, _summary(0.0), _summary(0.0))
    malformed = _trace({"not_a_complete_summary": True}, _summary(0.0), _summary(0.0))
    assert nonfinite["status"] == "INVALID_NONFINITE"
    assert malformed["status"] == "INVALID_MALFORMED"


def test_missing_projection_provenance_is_invalid_projection():
    projected = dict(_summary(1.0))
    projected["signed_projection"] = 1.0
    result = _trace(_summary(0.0), projected, _summary(0.0))
    assert result["status"] == "INVALID_PROJECTION"


def test_formation_marks_candidate_measurements_but_not_candidate_blind():
    result = _trace(_summary(0.0), _summary(0.0), _summary(0.0))
    assert result["measurement_kind"] == "candidate_repair_ground_truth"
    assert result["uses_candidate_measurements"] is True
    assert result["verdict_blind"] is True
    assert "candidate_blind" not in result
    assert result["trajectory_drift_in_formation"] is False


def test_four_synthetic_transition_patterns():
    centered = _summary(0.0)
    biased = _summary(1.0)
    assert _trace(centered, centered, centered)["status"] == "COMPLETE"
    local = _trace(biased, biased, biased)
    assert local["first_confirmed_bias_stage"] == "LOCAL_ENDPOINT"
    gradient = _trace(centered, biased, biased)
    assert gradient["first_confirmed_bias_stage"] == "PARAMETER_GRADIENT"
    update = _trace(centered, centered, biased)
    assert update["first_confirmed_bias_stage"] == "EFFECTIVE_UPDATE"
