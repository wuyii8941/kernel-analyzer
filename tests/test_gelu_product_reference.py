import pytest
import torch

from kernel_analyzer.gelu_product_reference import evaluate


def test_tanh_intervention_preserves_shape_dtype_and_finiteness():
    gradient = torch.tensor([0.5, -0.25, 1.0], dtype=torch.bfloat16)
    multiplier = torch.tensor([1.0, 0.75, -0.5], dtype=torch.bfloat16)
    saved_input = torch.tensor([-1.25, 0.125, 2.0], dtype=torch.bfloat16)
    native = evaluate(gradient, multiplier, saved_input,
                      variant="NATIVE_TANH_SOURCE_ORDER")
    explicit = evaluate(gradient, multiplier, saved_input,
                        variant="EXP_TANH_SOURCE_ORDER")
    fused = evaluate(gradient, multiplier, saved_input,
                     variant="NATIVE_TANH_FUSED_MULTIPLY_ADD")
    assert native.shape == explicit.shape == fused.shape == gradient.shape
    assert native.dtype == explicit.dtype == fused.dtype == torch.bfloat16
    assert all(torch.isfinite(value).all() for value in (native, explicit, fused))


def test_unknown_tanh_intervention_is_rejected():
    value = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unsupported reference precision"):
        evaluate(value, value, value, variant="UNKNOWN")
