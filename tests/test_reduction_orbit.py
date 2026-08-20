import math

import pytest
import torch

from kernel_analyzer.reduction_orbit import frozen_permutations, gemm_reduction_orbit


def test_exact_candidate_has_zero_orbit_mean_and_variance():
    left = torch.tensor([[1.0, 2.0, 3.0]])
    right = torch.tensor([[1.0], [2.0], [3.0]])
    result = gemm_reduction_orbit(
        left, right, permutations=frozen_permutations(3, 4, 7),
    )
    assert result["orbit_mean_l2"] == 0.0
    assert result["orbit_variance_energy"] == 0.0
    assert result["training_bias_or_persistence_verdict"] is False


def test_order_dependent_candidate_is_detected_without_changing_real_target():
    left = torch.tensor([[1.0e8, 1.0, -1.0e8]])
    right = torch.ones((3, 1))

    def sequential(a, b):
        products = a.unsqueeze(-1) * b
        value = products[..., 0, :]
        for index in range(1, products.shape[-2]):
            value = (value + products[..., index, :]).float()
        return value

    result = gemm_reduction_orbit(
        left, right, permutations=frozen_permutations(3, 8, 11), candidate=sequential,
    )
    assert result["orbit_variance_energy"] > 0.0
    assert math.isfinite(result["orbit_mean_energy_fraction"])


def test_invalid_orbit_member_fails_closed():
    with pytest.raises(ValueError):
        gemm_reduction_orbit(
            torch.ones((1, 3)), torch.ones((3, 1)),
            permutations=(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2])),
        )
