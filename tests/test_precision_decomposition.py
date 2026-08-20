from __future__ import annotations

import pytest
import torch

from kernel_analyzer.precision import (
    decompose_low_precision_output,
    low_precision_neighbors,
    source_aligned_mm_output,
    stochastic_round_to_low_precision,
)


def test_precision_decomposition_closes_and_separates_kernel_difference() -> None:
    high = torch.tensor([1.003, -2.007, 0.125], dtype=torch.float32)
    rounded = high.to(torch.bfloat16)
    actual = rounded.clone()
    actual[2] = torch.nextafter(
        actual[2], torch.tensor(float("inf"), dtype=torch.bfloat16)
    )
    terms = decompose_low_precision_output(actual, high)
    assert torch.equal(
        terms["kernel"] + terms["output_rounding"], terms["total"]
    )
    assert torch.count_nonzero(terms["kernel"]).item() == 1
    assert torch.count_nonzero(terms["output_rounding"]).item() == 2


def test_precision_decomposition_rejects_non_fp32_reference() -> None:
    low = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="FP32"):
        decompose_low_precision_output(low, low)


def test_precision_decomposition_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        decompose_low_precision_output(
            torch.ones(2, dtype=torch.bfloat16), torch.ones(3, dtype=torch.float32)
        )


def test_low_precision_neighbors_bracket_reference() -> None:
    high = torch.tensor([1.003, -2.007, 0.125], dtype=torch.float32)
    lower, upper, probability = low_precision_neighbors(high, torch.bfloat16)
    assert torch.all(lower.float() <= high)
    assert torch.all(high <= upper.float())
    expected = lower.float() + probability * (upper.float() - lower.float())
    assert torch.allclose(expected, high, rtol=0.0, atol=2.0 ** -23)
    assert lower[2] == upper[2]
    assert probability[2] == 0


def test_stochastic_rounding_is_empirically_unbiased() -> None:
    high = torch.tensor([1.003, -2.007, 0.127], dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)
    draws = [
        stochastic_round_to_low_precision(
            high, torch.bfloat16, generator=generator,
        ).delivered.float()
        for _ in range(20000)
    ]
    empirical = torch.stack(draws).mean(0)
    assert torch.allclose(empirical, high, rtol=0.0, atol=2.5e-4)


def test_stratified_rounding_ensemble_is_ulp_over_count_accurate() -> None:
    high = torch.tensor([-1.003, 0.101, 2.019], dtype=torch.float32)
    draws = []
    for index in range(8):
        generator = torch.Generator().manual_seed(91)
        draws.append(stochastic_round_to_low_precision(
            high, torch.bfloat16, generator=generator,
            stratum_index=index, stratum_count=8,
        ).delivered.float())
    empirical = torch.stack(draws).mean(0)
    lower, upper, _ = low_precision_neighbors(high, torch.bfloat16)
    bound = (upper.float() - lower.float()) / 8 + 1e-7
    assert torch.all((empirical - high).abs() <= bound)


def test_source_aligned_modes_do_not_confuse_rounding_and_kernel() -> None:
    high = torch.tensor([1.003, -2.007, 0.127], dtype=torch.float32)
    rounded = high.to(torch.bfloat16)
    actual = rounded.clone()
    actual[2] = torch.nextafter(
        actual[2], torch.tensor(float("inf"), dtype=torch.bfloat16)
    )
    sham = source_aligned_mm_output(actual, high, "SHAM")
    kernel = source_aligned_mm_output(actual, high, "KERNEL_ONLY")
    assert torch.equal(sham.delivered, actual)
    assert torch.equal(kernel.delivered, rounded)

    generator = torch.Generator().manual_seed(7)
    rounding = source_aligned_mm_output(
        actual, high, "ROUNDING_ONLY", generator=generator,
    )
    joint = source_aligned_mm_output(
        actual, high, "JOINT", generator=generator,
    )
    assert rounding.delivered.dtype == actual.dtype
    assert joint.delivered.dtype == actual.dtype
    assert torch.equal(joint.delivered, joint.base)


def test_randomized_modes_require_explicit_generator() -> None:
    high = torch.tensor([1.003], dtype=torch.float32)
    actual = high.to(torch.bfloat16)
    with pytest.raises(ValueError, match="explicit generator"):
        source_aligned_mm_output(actual, high, "ROUNDING_ONLY")


def test_rounding_repair_rejects_values_outside_low_precision_support() -> None:
    high = torch.tensor([1.0e10], dtype=torch.float32)
    with pytest.raises(ValueError, match="outside finite"):
        low_precision_neighbors(high, torch.float16)
