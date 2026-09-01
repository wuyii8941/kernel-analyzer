import math

import torch

from kernel_analyzer.fp32_reduction_order import (
    balanced_fp32_sum,
    ordered_rms_norm,
    permuted_sequential_fp32_sum,
    sequential_fp32_sum,
)


def test_two_fp32_orders_have_equal_real_sum_but_different_rounding() -> None:
    values = torch.tensor([[1.0e20], [1.0], [-1.0e20], [1.0]], dtype=torch.float32)
    sequential = sequential_fp32_sum(values)
    balanced = balanced_fp32_sum(values)
    assert sequential.item() == 1.0
    assert balanced.item() == 0.0
    assert math.fsum(float(value) for value in values.reshape(-1)) == 2.0


def test_permuted_sum_rejects_non_permutation() -> None:
    values = torch.ones(4, 2, dtype=torch.float32)
    try:
        permuted_sequential_fp32_sum(values, torch.tensor([0, 1, 1, 3]))
    except ValueError as error:
        assert "not a permutation" in str(error)
    else:
        raise AssertionError("invalid reduction order was accepted")


def test_ordered_rmsnorm_changes_only_weight_gradient() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(2, 7, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    upstream = torch.randn_like(hidden)

    results = []
    for order in ("sequential", "balanced"):
        x = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        output = ordered_rms_norm(x, w, epsilon=1e-6, reduction=order)
        output.backward(upstream)
        results.append((output.detach(), x.grad.detach(), w.grad.detach()))

    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    expected = (upstream * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)).double()
    expected = expected.reshape(-1, 8).sum(0)
    assert torch.allclose(results[0][2].double(), expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(results[1][2].double(), expected, rtol=1e-5, atol=1e-6)
