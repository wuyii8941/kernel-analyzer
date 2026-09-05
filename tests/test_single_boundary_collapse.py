from __future__ import annotations

import numpy as np
import pytest

from kernel_analyzer.single_boundary_collapse import (
    balanced_sign,
    balanced_signs,
    classify_paired_loss_collapse,
    prefix_statistics,
)


def test_balanced_sign_block_is_exactly_balanced_and_repeatable() -> None:
    first = balanced_signs(block_size=32, seed=7, block_index=3)
    second = balanced_signs(block_size=32, seed=7, block_index=3)
    assert first == second
    assert first.count(1) == 16
    assert first.count(-1) == 16


def test_balanced_sign_matches_block_helper() -> None:
    values = [balanced_sign(step, block_size=32, seed=11) for step in range(64)]
    assert sum(values[:32]) == 0
    assert sum(values[32:]) == 0


def test_equal_norm_sequences_have_equal_energy() -> None:
    coherent = prefix_statistics([2.0] * 32, [4.0] * 32, [2.0] * 32)
    balanced = prefix_statistics(
        [2.0] * 32,
        [4.0] * 32,
        [2.0 * balanced_sign(step, seed=5) for step in range(32)],
    )
    assert coherent["relative_injection_energy"] == balanced["relative_injection_energy"]
    assert coherent["relative_mean_direction_energy"] > 0.0
    assert balanced["relative_mean_direction_energy"] == pytest.approx(0.0)


def test_single_loss_spike_is_not_collapse() -> None:
    repair = np.ones(600)
    candidate = np.ones(600)
    candidate[200] = 100.0
    result = classify_paired_loss_collapse(candidate, repair)
    assert result["collapsed"] is False


def test_sustained_loss_ratio_is_collapse() -> None:
    repair = np.ones(700)
    candidate = np.ones(700)
    candidate[100:] = 2.0
    result = classify_paired_loss_collapse(candidate, repair)
    assert result["collapsed"] is True
    assert result["reason"] == "SUSTAINED_LOSS_RATIO"


def test_nonfinite_repair_invalidates_comparison() -> None:
    repair = np.ones(600)
    candidate = np.ones(600)
    repair[20] = np.nan
    result = classify_paired_loss_collapse(candidate, repair)
    assert result["collapsed"] is False
    assert result["reason"] == "REPAIR_NONFINITE"
