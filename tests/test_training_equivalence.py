from __future__ import annotations

import pytest
import numpy as np

from kernel_analyzer.training_equivalence import (
    classify_training_equivalence,
    simultaneous_intervals_from_joint_gram,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


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
