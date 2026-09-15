import pytest
import torch

from kernel_analyzer.rotary_trig_variants import fp64_reference, rotary_variant


def test_fp64_reference_shape_and_representation():
    query = torch.randn(32, 128, 128).to(torch.bfloat16)
    frequency = torch.randn(64)
    position = torch.arange(128, dtype=torch.int64)
    result = fp64_reference(query, frequency, position)
    assert result.shape == query.shape
    assert result.dtype == query.dtype
    assert torch.isfinite(result).all()


def test_variant_rejects_non_cuda_boundary():
    query = torch.zeros(32, 128, 128, dtype=torch.bfloat16)
    frequency = torch.zeros(64)
    position = torch.arange(128, dtype=torch.int64)
    with pytest.raises(ValueError, match="declared boundary"):
        rotary_variant(query, frequency, position, trig="tl_math")


def test_variant_name_is_explicitly_limited():
    query = torch.empty(32, 128, 128, dtype=torch.bfloat16)
    frequency = torch.empty(64)
    position = torch.arange(128, dtype=torch.int64)
    with pytest.raises(ValueError, match="trig must"):
        rotary_variant(query, frequency, position, trig="unknown")
