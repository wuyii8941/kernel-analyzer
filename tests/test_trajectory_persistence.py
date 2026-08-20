from __future__ import annotations

import math

import torch

from kernel_analyzer.trajectory_persistence import OrderedVectorPath


def test_coherent_path_exceeds_diffusive_scale() -> None:
    path = OrderedVectorPath(total_steps=16, calibration_steps=4)
    for _ in range(16):
        path.add(torch.tensor([1.0, 0.0]))
    result = path.finalize()
    assert result["resultant_over_path"] == 1.0
    assert result["coherence_amplification"] == 4.0
    assert result["evaluation_signed_persistence"] == 1.0


def test_canceling_path_has_zero_resultant() -> None:
    path = OrderedVectorPath(total_steps=16, calibration_steps=4)
    for step in range(16):
        path.add(torch.tensor([1.0 if step % 2 == 0 else -1.0, 0.0]))
    result = path.finalize()
    assert result["resultant_l2"] == 0.0
    assert result["coherence_amplification"] == 0.0


def test_orthogonal_path_matches_diffusive_scale() -> None:
    path = OrderedVectorPath(total_steps=8, calibration_steps=2)
    for step in range(8):
        value = torch.zeros(8)
        value[step] = 1.0
        path.add(value)
    result = path.finalize()
    assert math.isclose(result["coherence_amplification"], 1.0, rel_tol=1e-6)
