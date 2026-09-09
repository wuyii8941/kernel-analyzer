import pytest
import torch
from kernel_analyzer.softcapped_nll_reference import softcapped_nll_backward


@pytest.mark.parametrize('scale', [0., 1., 30., 1000.])
@pytest.mark.parametrize('cotangent', [-2., 0., 3.])
def test_matches_natural_loss_autograd(scale, cotangent):
    g = torch.Generator().manual_seed(41)
    x = (scale*torch.randn(4, 11, generator=g, dtype=torch.float64)).requires_grad_()
    y = torch.tensor([0, 10, -100, 4])
    z = 30*torch.tanh(x/30)
    maximum = z.detach().amax(1)
    logsum = torch.logsumexp(z.detach()-maximum[:, None], 1)
    actual = softcapped_nll_backward(x.detach(), y, maximum, logsum,
                                   torch.tensor(3.), torch.tensor(cotangent), cap=30.)
    expected = torch.autograd.grad(torch.nn.functional.cross_entropy(z, y)*cotangent, x)[0]
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[2]) == 0


def test_consumes_saved_normalization_not_new_softmax():
    actual = softcapped_nll_backward(torch.zeros(1, 2), torch.tensor([0]),
        torch.zeros(1), torch.zeros(1), torch.tensor(1.), torch.tensor(1.), cap=30.)
    torch.testing.assert_close(actual, torch.tensor([[0., 1.]], dtype=torch.float64))


@pytest.mark.parametrize('cap', [0., -1., float('inf'), float('nan'), True])
def test_invalid_cap_rejected(cap):
    with pytest.raises(ValueError, match='softcap'):
        softcapped_nll_backward(torch.zeros(1, 2), torch.tensor([0]), torch.zeros(1),
            torch.zeros(1), torch.tensor(1.), torch.tensor(1.), cap=cap)


def test_infinite_input_not_hidden_by_tanh():
    with pytest.raises(ValueError, match='Finite floating'):
        softcapped_nll_backward(torch.tensor([[float('inf'), 0.]]), torch.tensor([0]),
            torch.zeros(1), torch.zeros(1), torch.tensor(1.), torch.tensor(1.), cap=30.)
