from kernel_analyzer.joint_event_oracle import (
    certify_joint_event_gram,
    gram_from_event_vectors,
)


def certify(vectors):
    return certify_joint_event_gram(
        gram_from_event_vectors(vectors),
        coordinate_count=len(vectors[0]),
        random_sign_draws=1000,
        seed=7,
    )


def test_aligned_events_are_risky_without_fitting_a_carrier():
    result = certify([[1.0, 0.0]] * 8)
    assert result.status == "COHERENT_JOINT_EVENT_RISK"
    assert result.normalized_cross_event_coherence == 1.0


def test_balanced_antithetic_events_are_canceling():
    result = certify([[1.0, 0.0], [-1.0, 0.0]] * 4)
    assert result.status == "CANCELING_EVENT_STRUCTURE"
    assert result.resultant_energy == 0.0


def test_orthogonal_events_are_not_called_bias():
    result = certify([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert result.status == "NO_COHERENT_JOINT_EVENT_RISK"
    assert result.normalized_cross_event_coherence == 0.0


def test_malformed_gram_fails_closed():
    try:
        certify_joint_event_gram([[1.0, 2.0], [0.0, 1.0]], coordinate_count=2)
    except ValueError as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("asymmetric Gram was accepted")
