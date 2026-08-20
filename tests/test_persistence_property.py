import math

import numpy as np

from kernel_analyzer.persistence_property import (
    aligned_level_statistics_from_gram,
    five_level_signature,
    path_statistics_from_gram,
    semantic_orbit_statistics_from_gram,
)


def _gram(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix @ matrix.T


def test_path_statistics_separates_coherent_and_canceling_sequences():
    coherent = _gram([[1, 0]] * 16)
    canceling = _gram([[1, 0] if index % 2 == 0 else [-1, 0] for index in range(16)])
    yes = path_statistics_from_gram(coherent, sign_flip_draws=1000, seed=2)
    no = path_statistics_from_gram(canceling, sign_flip_draws=1000, seed=2)
    assert yes["coherence_amplification"] == 4.0
    assert yes["above_sign_flip_95"] is True
    assert no["coherence_amplification"] == 0.0
    assert no["lag_correlation"][0]["normalized_correlation"] == -1.0


def test_final_amplification_is_order_invariant_but_lag_curve_is_not():
    rows = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
    first = path_statistics_from_gram(rows @ rows.T, sign_flip_draws=200, seed=4)
    permutation = [0, 2, 1, 3]
    permuted = rows[permutation]
    second = path_statistics_from_gram(permuted @ permuted.T, sign_flip_draws=200, seed=4)
    assert first["coherence_amplification"] == second["coherence_amplification"]
    assert first["lag_correlation"] != second["lag_correlation"]


def test_semantic_orbit_separates_mean_from_default_schedule_residual():
    # state-major rows; both states have orbit mean [1, 0], while the default
    # schedule has a state-changing residual on the second coordinate.
    rows = [[1, 1], [1, -1], [1, -1], [1, 1]]
    result = semantic_orbit_statistics_from_gram(
        _gram(rows), state_ids=("s0", "s1"), variant_ids=("default", "other"),
        default_variant="default", sign_flip_draws=200, seed=3,
    )
    assert math.isclose(result["orbit_mean"]["coherence_amplification"], math.sqrt(2))
    assert result["default_minus_orbit_mean"]["coherence_amplification"] == 0.0
    assert result["orbit_mean_energy_fraction"] == 0.5


def test_aligned_levels_report_feedback_alignment_without_posthoc_basis():
    # state-major L/B/D with D=L+B. L cancels; B and D persist.
    rows = []
    for sign in (1.0, -1.0):
        local = np.array([sign, 0.0])
        feedback = np.array([0.0, 1.0])
        actual = local + feedback
        rows.extend([local, feedback, actual])
    result = aligned_level_statistics_from_gram(
        _gram(rows), state_ids=("s0", "s1"), level_ids=("local", "feedback", "actual"),
        sign_flip_draws=200, seed=8,
    )
    assert result["levels"]["local"]["coherence_amplification"] == 0.0
    assert math.isclose(result["levels"]["feedback"]["coherence_amplification"], math.sqrt(2))
    assert result["resultant_cosines"]["feedback__actual"] == 1.0


def test_five_level_signature_keeps_incompatible_coordinate_spaces_separate():
    result = five_level_signature(
        independent_grams={
            "epsilon": _gram([[1, 0], [-1, 0]]),
            "gradient": _gram([[1, 0, 0], [1, 0, 0]]),
        },
        state_ids=("a", "b"),
        aligned_lbd_gram=_gram([
            [1, 0], [0, 0], [1, 0],
            [1, 0], [0, 0], [1, 0],
        ]),
        sign_flip_draws=200,
    )
    assert result["independent_coordinate_spaces"]["epsilon"]["coherence_amplification"] == 0.0
    assert result["independent_coordinate_spaces"]["gradient"]["coherence_amplification"] == math.sqrt(2)
    assert result["uses_trajectory_or_seup_verdict_as_input"] is False
