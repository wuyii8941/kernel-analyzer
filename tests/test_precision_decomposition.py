from __future__ import annotations

import pytest
import torch

from kernel_analyzer.precision import decompose_low_precision_output


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
