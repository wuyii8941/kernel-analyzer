import random

from kernel_analyzer.bias_formation_v21 import (
    BiasFormationTrace,
    FormationPolicy,
    FormationStatus,
    summarize_state_vectors,
    summarize_streamed_state_vectors,
)


POLICY = FormationPolicy(min_states=16, bootstrap_samples=2000)


def _common(weights="w", optimizer="o"):
    return {
        "candidate_weights_digest": weights,
        "repair_weights_digest": weights,
        "candidate_optimizer_digest": optimizer,
        "repair_optimizer_digest": optimizer,
        "input_digest": "input",
        "rng_digest": "rng",
        "scheduler_digest": "scheduler",
        "loss_scaler_digest": "scale",
    }


def _independent_vectors(seed=1, n=16, d=16):
    rng = random.Random(seed)
    return [[-1.0 if rng.randrange(2) else 1.0 for _ in range(d)] for _ in range(n)]


def test_independent_zero_mean_vectors_are_not_stably_biased():
    cert = summarize_state_vectors(_independent_vectors(), policy=POLICY)
    assert cert.status != FormationStatus.BIASED.value
    assert abs(cert.cross_state_ratio) < POLICY.bias_margin


def test_identical_vectors_are_biased():
    cert = summarize_state_vectors([[1.0] * 16 for _ in range(16)], policy=POLICY)
    assert cert.status == FormationStatus.BIASED.value
    assert cert.bootstrap_lower >= POLICY.bias_margin


def test_alternating_vectors_record_cancellation_not_positive_bias():
    vectors = [[1.0] * 16 if i % 2 == 0 else [-1.0] * 16 for i in range(16)]
    cert = summarize_state_vectors(vectors, policy=POLICY)
    assert cert.cross_state_u_statistic < 0
    assert cert.status in {
        FormationStatus.CENTERED.value,
        FormationStatus.CANCELING_STRUCTURE.value,
        FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value,
    }
    assert cert.status != FormationStatus.BIASED.value


def test_streamed_and_dense_population_are_identical_and_order_invariant():
    vectors = _independent_vectors(seed=4)
    dense = summarize_state_vectors(vectors, state_ids=[str(i) for i in range(16)], policy=POLICY)
    streamed = summarize_streamed_state_vectors((row for row in vectors), state_ids=[str(i) for i in range(16)], policy=POLICY)
    perm = list(reversed(vectors))
    reordered = summarize_state_vectors(perm, state_ids=[str(i) for i in range(16)], policy=POLICY)
    assert dense.complete_gram == streamed.complete_gram
    assert dense.cross_state_u_statistic == streamed.cross_state_u_statistic
    assert dense.status == streamed.status
    assert reordered.status == dense.status
    assert reordered.cross_state_ratio == dense.cross_state_ratio


def _trace(local, gradient, update):
    trace = BiasFormationTrace("toy", [f"c{i}" for i in range(16)], [f"e{i}" for i in range(16)], POLICY)
    for partition, prefix in (("calibration", "c"), ("confirmation", "e")):
        for i in range(16):
            state_id = prefix + str(i)
            trace.add(state_id, partition, common_state_certificate=_common(),
                      local_endpoint=local[i], parameter_gradient=gradient[i], effective_update=update[i])
    return trace.finalize()


def test_local_centered_gradient_biased_first_stage_is_gradient():
    centered = _independent_vectors(seed=9)
    biased = [[1.0] * 16 for _ in range(16)]
    result = _trace(centered, biased, biased)
    assert result["first_confirmed_bias_stage"] == "PARAMETER_GRADIENT"


def test_component_digest_mismatch_is_invalid_common_state():
    trace = BiasFormationTrace("toy", ["c0"] * 0 + [f"c{i}" for i in range(16)], [f"e{i}" for i in range(16)], POLICY)
    vectors = _independent_vectors(seed=11)
    bad = _common(weights="repair-does-not-match")
    bad["candidate_weights_digest"] = "candidate"
    for partition, prefix in (("calibration", "c"), ("confirmation", "e")):
        for i in range(16):
            state_id = prefix + str(i)
            trace.add(state_id, partition, common_state_certificate=bad,
                      local_endpoint=vectors[i], parameter_gradient=vectors[i], effective_update=vectors[i])
    assert trace.finalize()["status"] == FormationStatus.INVALID_COMMON_STATE.value
