from __future__ import annotations

import numpy as np

from kernel_analyzer.training_bias_profile import (
    holm_adjusted_p,
    matched_training_bias_profile,
)


def _split() -> tuple[list[int], list[int]]:
    return list(range(16)), list(range(16, 32))


def _independent_units() -> list[str]:
    return [f"cal-{i}" for i in range(16)] + [f"confirm-{i}" for i in range(16)]


def test_fixed_suite_does_not_invent_population_uncertainty() -> None:
    rng = np.random.default_rng(3)
    repair = rng.normal(size=(32, 16))
    effect = 0.01 + 0.001 * rng.normal(size=(32, 16))
    cal, conf = _split()
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=cal,
        confirmation_indices=conf,
        inference_unit_ids=None,
    )
    assert result["status"] == "DESCRIPTIVE_FIXED_SUITE_ONLY"
    assert result["population_inference"] is None
    assert result["suite"]["mean_effect_over_repair_rms"] > 0.0


def test_one_continuous_training_path_is_not_sixteen_independent_states() -> None:
    rng = np.random.default_rng(5)
    repair = rng.normal(size=(32, 16))
    effect = 0.02 + 0.001 * rng.normal(size=(32, 16))
    cal, conf = _split()
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=cal,
        confirmation_indices=conf,
        inference_unit_ids=["one-training-run"] * 32,
    )
    assert result["status"] == "DESCRIPTIVE_FIXED_SUITE_ONLY"
    assert result["abstention_reason"] == "CALIBRATION_AND_CONFIRMATION_SHARE_TRAINING_UNITS"


def test_fixed_additive_effect_repeats_on_independent_confirmation_units() -> None:
    rng = np.random.default_rng(7)
    repair = rng.normal(size=(32, 64))
    direction = np.zeros(64)
    direction[:8] = 0.08
    effect = direction + 0.01 * rng.normal(size=(32, 64))
    cal, conf = _split()
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=cal,
        confirmation_indices=conf,
        inference_unit_ids=_independent_units(),
        signflip_draws=2000,
        seed=9,
    )
    branch = result["population_inference"]["branches"]["additive"]
    assert result["status"] == "POPULATION_INFERENCE_COMPLETE"
    assert branch["estimate"] > 0.0
    assert branch["confidence_interval_95"][0] > 0.0
    assert branch["raw_confirmed"]


def test_rotating_repair_aligned_effect_is_detected_without_fixed_direction() -> None:
    rng = np.random.default_rng(11)
    repair = rng.normal(size=(32, 128))
    repair /= np.linalg.norm(repair, axis=1, keepdims=True)
    effect = 0.08 * repair + 0.005 * rng.normal(size=(32, 128))
    cal, conf = _split()
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=cal,
        confirmation_indices=conf,
        inference_unit_ids=_independent_units(),
        signflip_draws=2000,
        seed=13,
    )
    aligned = result["population_inference"]["branches"]["repair_aligned"]
    assert aligned["estimate"] > 0.05
    assert aligned["confidence_interval_95"][0] > 0.0
    assert aligned["raw_confirmed"]


def test_declared_aligned_effect_is_energy_weighted() -> None:
    repair = np.ones((32, 1))
    effect = np.zeros((32, 1))
    repair[16] = 1.0
    repair[17:] = 10.0
    effect[16] = 1.0
    cal, conf = _split()
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=cal,
        confirmation_indices=conf,
        inference_unit_ids=None,
    )
    expected = 1.0 / (1.0 + 15.0 * 100.0)
    assert np.isclose(result["suite"]["repair_aligned_effect"], expected)
    assert not np.isclose(result["suite"]["repair_aligned_effect"], 1.0 / 16.0)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjusted_p({"a": 0.001, "b": 0.02, "c": 0.04})
    assert adjusted == {"a": 0.003, "b": 0.04, "c": 0.04}
