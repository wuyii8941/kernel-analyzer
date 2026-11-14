"""Independent attention reference for the other direct normalized-layer caller."""
import pytest
import torch
import triton

from fla.layers.rebased import ReBasedLinearAttention
from fla.utils import assert_close


@pytest.mark.parametrize('mode', ['chunk', 'fused_chunk'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_rebased_normalized_caller(mode, dtype, monkeypatch):
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    if hasattr(triton, 'knobs'):
        monkeypatch.setattr(triton.knobs.language, 'fp32_default', 'ieee')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(419)
    layer = ReBasedLinearAttention(
        hidden_size=16, feature_dim=4, num_heads=2,
        num_key_value_heads=2, mode=mode,
    ).to(device='cuda', dtype=dtype)
    x = torch.randn(1, 17, 16, device='cuda', dtype=dtype, requires_grad=True)
    # Reuse the physical producer values; the attention computation is an
    # independent quadratic implementation in ordinary PyTorch.
    with torch.no_grad():
        q = layer.feature_map(layer.q_proj(x).reshape(1, 17, 2, 4)).float().transpose(1, 2)
        k = layer.feature_map(layer.k_proj(x).reshape(1, 17, 2, 4)).float().transpose(1, 2)
        v = layer.v_proj(x).reshape(1, 17, 2, 8).float().transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)).tril()
        attention = (scores @ v)/(scores.sum(-1, keepdim=True)+1e-10)
        expected = layer.o_proj(attention.transpose(1, 2).reshape_as(x).to(dtype))
    actual = layer(x)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    tol = {torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    assert_close('rebased_output', expected, actual, tol, err_atol=1e-6)
    actual.backward(torch.ones_like(actual))
    for value in [x, *layer.parameters()]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
