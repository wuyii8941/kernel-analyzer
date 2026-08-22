import random

import numpy as np
import pytest

from kernel_analyzer.bias_formation_v21 import (
    BiasFormationTrace,
    FormationPolicy,
    FormationStatus,
    summarize_state_vectors,
    summarize_streamed_state_vectors,
    summarize_streamed_state_vector_files,
)


POLICY = FormationPolicy(min_states=16, bootstrap_samples=2000)


def _common(weights="w", optimizer="o"):
    return {
        "candidate_weights_digest": weights,
        "repair_weights_digest": weights,
        "candidate_optimizer_digest": optimizer,
        "repair_optimizer_digest": optimizer,
        "candidate_input_digest": "input",
        "repair_input_digest": "input",
        "candidate_rng_digest": "rng",
        "repair_rng_digest": "rng",
        "candidate_scheduler_digest": "scheduler",
        "repair_scheduler_digest": "scheduler",
        "candidate_loss_scaler_digest": "scale",
        "repair_loss_scaler_digest": "scale",
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


def test_complete_population_can_be_inconclusive_without_being_short():
    rng = random.Random(1)
    rho = 0.20
    vectors = []
    for _ in range(16):
        signs = [-1.0 if rng.randrange(2) else 1.0 for _ in range(64)]
        vectors.append([
            rho ** 0.5 + (1.0 - rho) ** 0.5 * value
            for value in signs
        ])
    cert = summarize_state_vectors(vectors, policy=POLICY)
    assert cert.status == FormationStatus.UNRESOLVED_INCONCLUSIVE.value
    assert len(cert.state_ids) == POLICY.min_states


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


def test_file_streamed_and_dense_population_are_identical(tmp_path):
    vectors = _independent_vectors(seed=14)
    dense = summarize_state_vectors(
        vectors, state_ids=[str(i) for i in range(16)], policy=POLICY,
    )
    files = []
    for index, values in enumerate(vectors):
        path = tmp_path / f"state-{index}.f32"
        np.asarray(values, dtype=np.float32).tofile(path)
        files.append({
            "state_id": str(index), "path": str(path),
            "coordinate_count": len(values), "vector_digest": f"digest-{index}",
            "storage_dtype": "float32",
        })
    streamed = summarize_streamed_state_vector_files(
        files, layer="LOCAL_ENDPOINT", partition="confirmation", policy=POLICY,
    )
    assert dense.status == streamed.status
    assert dense.cross_state_ratio == pytest.approx(streamed.cross_state_ratio)
    assert np.asarray(dense.complete_gram) == pytest.approx(
        np.asarray(streamed.complete_gram), rel=1e-6, abs=1e-6,
    )


def test_file_streamed_scale_reuses_one_vector_without_changing_gram(tmp_path):
    vectors = _independent_vectors(seed=15)
    files = []
    scaled = []
    for index, values in enumerate(vectors):
        path = tmp_path / f"scaled-{index}.f32"
        np.asarray(values, dtype=np.float32).tofile(path)
        scale = -1.0e-4
        files.append({
            "state_id": str(index), "path": str(path),
            "coordinate_count": len(values), "storage_dtype": "float32",
            "scale": scale,
        })
        scaled.append(np.asarray(values) * scale)
    dense = summarize_state_vectors(
        scaled, state_ids=[str(i) for i in range(16)], policy=POLICY,
    )
    streamed = summarize_streamed_state_vector_files(
        files, layer="EFFECTIVE_UPDATE", partition="confirmation", policy=POLICY,
    )
    assert np.asarray(dense.complete_gram) == pytest.approx(
        np.asarray(streamed.complete_gram), rel=1e-6, abs=1e-18,
    )


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


def test_non_weight_component_digest_mismatch_is_also_invalid():
    trace = BiasFormationTrace("toy", [f"c{i}" for i in range(16)], [f"e{i}" for i in range(16)], POLICY)
    vectors = _independent_vectors(seed=12)
    bad = _common()
    bad["candidate_rng_digest"] = "candidate-rng"
    bad["repair_rng_digest"] = "repair-rng"
    for partition, prefix in (("calibration", "c"), ("confirmation", "e")):
        for i in range(16):
            state_id = prefix + str(i)
            trace.add(state_id, partition, common_state_certificate=bad,
                      local_endpoint=vectors[i], parameter_gradient=vectors[i], effective_update=vectors[i])
    assert trace.finalize()["status"] == FormationStatus.INVALID_COMMON_STATE.value
