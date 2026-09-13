from unittest.mock import patch
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchao')

from torchao.optim import AdamW8bit
from kernel_analyzer.adamw8bit_error_compensation import AdamW8bitErrorCompensated
from kernel_analyzer.compensation_control import CompensationControl, TensorScalarCompensationControl


@pytest.mark.parametrize('mode', ['on_matches_frozen', 'tensor_off_matches_eager'])
def test_shared_gradient_recurrence(mode):
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(4096, generator=generator)
    left, right = (torch.nn.Parameter(base.clone()) for _ in range(2))
    settings = dict(lr=1e-3, betas=(.9,.999), weight_decay=.01)
    if mode == 'on_matches_frozen':
        first = CompensationControl([left], compensation_enabled=True, **settings)
        second = AdamW8bitErrorCompensated([right], **settings)
    else:
        first = TensorScalarCompensationControl([left], compensation_enabled=False, **settings)
        second = AdamW8bit([right], block_size=256, **settings)
    for _ in range(3):
        gradient = torch.randn(4096, generator=generator)
        left.grad, right.grad = gradient.clone(), gradient.clone()
        first.step()
        with patch('torch.compile', side_effect=lambda f, **kwargs: f):
            second.step()
        assert torch.equal(left, right)
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.equal(first.state[left][key].dequantize(), second.state[right][key].dequantize())
