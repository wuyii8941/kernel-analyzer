from __future__ import annotations

import pytest
import numpy as np

from kernel_analyzer.training_equivalence import (
    classify_fixed_suite_update_equivalence,
    classify_training_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    simultaneous_intervals_from_joint_gram,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def _gram(effects: np.ndarray, repairs: np.ndarray) -> dict:
    return {
        "effect_effect": (effects @ effects.T).tolist(),
        "repair_repair": (repairs @ repairs.T).tolist(),
        "effect_repair": (effects @ repairs.T).tolist(),
    }


def test_equivalence_requires_every_interval_inside_margin() -> None:
    result = classify_training_equivalence(
        {
            "additive": [-0.0002, 0.0003],
            "repair_aligned": [-0.003, 0.004],
            "residual_direction": [-0.0004, 0.0004],
        },
        MARGINS,
    )
    assert result["decision"] == "EQUIVALENT_UNDER_PROTOCOL"


def test_nonsignificant_interval_crossing_margin_is_inconclusive() -> None:
    result = classify_training_equivalence(
        {
            "additive": [-0.0012, 0.0002],
            "repair_aligned": [-0.003, 0.004],
            "residual_direction": [-0.0004, 0.0004],
        },
        MARGINS,
    )
    assert result["decision"] == "INCONCLUSIVE"


def test_effect_entirely_beyond_margin_is_material() -> None:
    result = classify_training_equivalence(
        {
            "additive": [0.0011, 0.0014],
            "repair_aligned": [-0.003, 0.004],
            "residual_direction": [-0.0004, 0.0004],
        },
        MARGINS,
    )
    assert result["decision"] == "MATERIAL_EFFECT"


def test_detectable_effect_can_still_be_small() -> None:
    result = classify_training_equivalence(
        {
            "additive": [0.0001, 0.0004],
            "repair_aligned": [-0.003, 0.004],
            "residual_direction": [-0.0004, 0.0004],
        },
        MARGINS,
    )
    assert result["decision"] == "DETECTABLE_BUT_SMALL"


def test_consequence_failure_overrides_small_update_intervals() -> None:
    result = classify_training_equivalence(
        {
            "additive": [-0.0002, 0.0003],
            "repair_aligned": [-0.003, 0.004],
            "residual_direction": [-0.0004, 0.0004],
        },
        MARGINS,
        material_consequence_failed=True,
    )
    assert result["decision"] == "MATERIAL_CONSEQUENCE"


def test_missing_branch_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_training_equivalence({"additive": [0.0, 0.0]}, MARGINS)


def test_joint_gram_recovers_additive_and_aligned_effects() -> None:
    rng = np.random.default_rng(7)
    repairs = rng.normal(size=(32, 6))
    fixed = np.array([1.0, -0.5, 0.25, 0.0, 0.0, 0.0])
    effects = 0.04 * repairs + 0.20 * fixed + rng.normal(scale=0.001, size=(32, 6))
    gram = {
        "effect_effect": (effects @ effects.T).tolist(),
        "repair_repair": (repairs @ repairs.T).tolist(),
        "effect_repair": (effects @ repairs.T).tolist(),
    }
    intervals = simultaneous_intervals_from_joint_gram(gram)
    assert intervals["repair_aligned"][0] > 0.0
    assert intervals["additive"][0] > 0.0
    assert set(intervals) == {"additive", "repair_aligned", "residual_direction"}


def test_joint_gram_accepts_exact_identity() -> None:
    count = 32
    zeros = np.zeros((count, count), dtype=np.float64)
    gram = {
        "effect_effect": zeros.tolist(),
        "repair_repair": np.eye(count, dtype=np.float64).tolist(),
        "effect_repair": zeros.tolist(),
    }
    assert simultaneous_intervals_from_joint_gram(gram) == {
        "additive": [0.0, 0.0],
        "repair_aligned": [0.0, 0.0],
        "residual_direction": [0.0, 0.0],
    }


def test_zero_summary_is_not_called_exact_without_direct_verification() -> None:
    result = classify_fixed_suite_update_equivalence(
        {name: [0.0, 0.0] for name in MARGINS},
        MARGINS,
        total_rms=0.0,
        total_rms_margin=0.01,
    )
    assert result["decision"] == "FIXED_SUITE_UPDATE_EQUIVALENT"
    assert result["exact_identity_verified"] is False


def test_directly_verified_zero_vector_can_be_called_exact() -> None:
    result = classify_fixed_suite_update_equivalence(
        {name: [0.0, 0.0] for name in MARGINS},
        MARGINS,
        total_rms=0.0,
        total_rms_margin=0.01,
        exact_identity_verified=True,
    )
    assert result["decision"] == "EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE"
    assert result["exact_identity_verified"] is True


def test_orthogonal_direction_shift_cannot_receive_equivalence() -> None:
    repairs = np.tile(np.array([1.0, 0.0, 0.0]), (32, 1))
    effects = np.zeros((32, 3), dtype=np.float64)
    effects[:16, 1] = 1e-4
    effects[16:, 2] = 10.0
    gram = _gram(effects, repairs)
    intervals = simultaneous_intervals_from_joint_gram(gram)
    assert all(value == [0.0, 0.0] for value in intervals.values())
    result = classify_fixed_suite_update_equivalence(
        intervals, MARGINS,
        total_rms=fixed_suite_total_rms_from_joint_gram(gram),
        total_rms_margin=0.01,
    )
    assert result["decision"] == "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"


def test_large_centered_unseen_direction_is_caught_by_energy() -> None:
    repairs = np.tile(np.array([1.0, 0.0, 0.0]), (32, 1))
    effects = np.zeros((32, 3), dtype=np.float64)
    effects[:16, 1] = 1e-4
    effects[16:, 2] = np.tile([1.0, -1.0], 8)
    gram = _gram(effects, repairs)
    result = classify_fixed_suite_update_equivalence(
        simultaneous_intervals_from_joint_gram(gram), MARGINS,
        total_rms=fixed_suite_total_rms_from_joint_gram(gram),
        total_rms_margin=0.01,
    )
    assert result["decision"] == "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"


def test_material_residual_direction_is_not_equivalent() -> None:
    repairs = np.tile(np.array([1.0, 0.0]), (32, 1))
    effects = np.tile(np.array([0.0, 0.002]), (32, 1))
    gram = _gram(effects, repairs)
    result = classify_fixed_suite_update_equivalence(
        simultaneous_intervals_from_joint_gram(gram), MARGINS,
        total_rms=fixed_suite_total_rms_from_joint_gram(gram),
        total_rms_margin=0.01,
    )
    assert result["decision"] == "MATERIAL_EFFECT"


def test_profile_material_result_is_preserved_when_energy_also_exceeds() -> None:
    repairs = np.tile(np.array([1.0, 0.0]), (32, 1))
    effects = 0.02 * repairs
    gram = _gram(effects, repairs)
    result = classify_fixed_suite_update_equivalence(
        simultaneous_intervals_from_joint_gram(gram), MARGINS,
        total_rms=fixed_suite_total_rms_from_joint_gram(gram),
        total_rms_margin=0.01,
    )
    assert result["decision"] == "MATERIAL_EFFECT"
    assert "FULL_UPDATE_RMS_EXCEEDS_ITS_MARGIN" in result["failure_reasons"]


def test_aligned_center_is_ratio_of_sums_with_unequal_repair_energy() -> None:
    scales = np.tile([0.1, 10.0], 16)
    gains = np.tile([0.02, 0.002], 16)
    repairs = np.stack([scales, np.zeros(32)], axis=1)
    effects = gains[:, None] * repairs
    gram = _gram(effects, repairs)
    intervals = simultaneous_intervals_from_joint_gram(gram)
    expected = float(np.sum(gains[16:] * scales[16:] ** 2) / np.sum(scales[16:] ** 2))
    center = sum(intervals["repair_aligned"]) / 2.0
    assert center == pytest.approx(expected)
    assert center != pytest.approx(float(np.mean(gains[16:])))


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (0.0095, "FIXED_SUITE_UPDATE_EQUIVALENT"),
        (0.0100, "INCONCLUSIVE"),
        (0.0105, "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"),
    ],
)
def test_full_update_rms_boundary(scale: float, expected: str) -> None:
    intervals = {name: [0.0, 0.0] for name in MARGINS}
    result = classify_fixed_suite_update_equivalence(
        intervals, MARGINS, total_rms=scale, total_rms_margin=0.01,
    )
    assert result["decision"] == expected
