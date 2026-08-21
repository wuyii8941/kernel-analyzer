from __future__ import annotations

import numpy as np
import pytest

from kernel_analyzer.short_persistence import (
    SharedShortPersistenceScreen,
    count_sketch,
    count_sketch_chunks,
)


def test_count_sketch_is_deterministic_and_chunk_invariant() -> None:
    values = np.linspace(-1.0, 1.0, 4097)
    first = count_sketch(values, projection_dim=32, seed=11, chunk_size=31)
    second = count_sketch(values, projection_dim=32, seed=11, chunk_size=4096)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, count_sketch(values, projection_dim=32, seed=12))


def test_streamed_count_sketch_matches_dense_vector() -> None:
    values = np.arange(257, dtype=np.float64) - 31.0
    dense = count_sketch(values, projection_dim=32, seed=19)
    streamed, coordinate_count = count_sketch_chunks(
        [values[:31], values[31:129], values[129:]],
        projection_dim=32,
        seed=19,
    )
    assert coordinate_count == values.size
    assert np.array_equal(dense, streamed)


def test_persistent_path_is_a_risk_candidate() -> None:
    screen = SharedShortPersistenceScreen(
        projection_dim=32, projection_seed=7, expected_steps=16, null_draws=300
    )
    for _ in range(16):
        screen.add("persistent", np.array([1.0, 0.5, 0.0, 0.0]))
    result = screen.finalize()["cases"][0]
    assert result["status"] == "RISK_CANDIDATE"
    assert result["positive_lag_count"] >= 2
    assert result["prefix_growth_after_short_warmup"]


def test_alternating_path_is_not_persistent() -> None:
    screen = SharedShortPersistenceScreen(
        projection_dim=32, projection_seed=7, expected_steps=16, null_draws=300
    )
    for step in range(16):
        screen.add("alternating", ((-1.0) ** step) * np.array([1.0, 0.5, 0.0, 0.0]))
    result = screen.finalize()["cases"][0]
    assert result["status"] == "NULL_LIKE_OR_UNRESOLVED"
    assert not result["lag1_positive"]


def test_after_warmup_prefix_gate_keeps_growth_after_initial_variability() -> None:
    screen = SharedShortPersistenceScreen(
        projection_dim=32, projection_seed=7, expected_steps=8, null_draws=300,
        prefix_growth_mode="after_warmup",
    )
    for step in range(8):
        screen.add("warmup_then_persistent", np.array([1.0 + 0.1 * step, 0.5, 0.0, 0.0]))
    result = screen.finalize()["cases"][0]
    assert result["screen_rule"]["prefix_growth_mode"] == "after_warmup"
    assert result["prefix_growth_after_short_warmup"]


def test_shared_screen_fails_closed_on_missing_or_changing_coordinates() -> None:
    screen = SharedShortPersistenceScreen(expected_steps=4, null_draws=100)
    for step in range(3):
        screen.add("missing", np.ones(4))
    with pytest.raises(ValueError, match="incomplete"):
        screen.finalize()

    screen = SharedShortPersistenceScreen(expected_steps=4, null_draws=100)
    screen.add("changing", np.ones(4))
    with pytest.raises(ValueError, match="coordinate count"):
        screen.add("changing", np.ones(5))
