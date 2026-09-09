import pytest
import torch
from kernel_analyzer.selected_nll_reference import selected_nll, selected_nll_backward


def test_matches_cross_entropy_with_ignored_labels():
    logits = torch.tensor([[1., 2., 3.], [-3., 1., 0.], [2., -1., 4.]], dtype=torch.float64)
    labels = torch.tensor([2, -100, 0])
    maximum = logits.max(1).values
    logsum = (logits-maximum[:, None]).exp().sum(1).log()
    loss, count = selected_nll(logits, labels, maximum, logsum)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels))
    assert count == 2


def test_invalid_and_all_ignored_labels_rejected():
    x = torch.ones(2, 3)
    for labels in (torch.tensor([-1, 0]), torch.tensor([3, 0]), torch.tensor([-100, -100])):
        with pytest.raises(ValueError):
            selected_nll(x, labels, torch.ones(2), torch.ones(2))


def test_normalization_terms_are_inputs_not_recomputed():
    x = torch.ones(2, 3)
    loss, _ = selected_nll(x, torch.tensor([0, 1]), torch.ones(2), torch.tensor([2., 4.]))
    assert loss == 3


def test_natural_normalization_backward_matches_cross_entropy():
    x = torch.tensor([[1., 2., -1.], [3., -2., 4.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, -100])
    maximum = x.max(1).values
    logsum = (x-maximum[:, None]).exp().sum(1).log()
    loss, _ = selected_nll(x, labels, maximum, logsum)
    actual = torch.autograd.grad(loss, x, retain_graph=True)[0]
    expected = torch.autograd.grad(torch.nn.functional.cross_entropy(x, labels), x)[0]
    torch.testing.assert_close(actual, expected)


def test_fixed_normalization_is_not_natural_backward():
    x = torch.tensor([[1., 2., -1.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1])
    maximum = x.detach().max(1).values
    logsum = (x.detach()-maximum[:, None]).exp().sum(1).log()
    loss, _ = selected_nll(x, labels, maximum, logsum)
    gradient = torch.autograd.grad(loss, x)[0]
    torch.testing.assert_close(gradient, torch.tensor([[0., -1., 0.]], dtype=torch.float64))


def test_explicit_backward_matches_full_loss_with_nonunit_cotangent():
    x = torch.tensor([[1., 2., -1.], [3., 0., 2.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, -100])
    maximum = x.detach().max(1).values
    logsum = (x.detach()-maximum[:, None]).exp().sum(1).log()
    actual = selected_nll_backward(x.detach(), labels, maximum, logsum,
                                   torch.tensor(1.), torch.tensor(-2.))
    expected = torch.autograd.grad(-2*torch.nn.functional.cross_entropy(x, labels), x)[0]
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='total weight'):
        selected_nll_backward(x.detach(), labels, maximum, logsum, torch.tensor(2.), torch.tensor(1.))


def test_backward_consumes_saved_normalization_without_renormalizing():
    x = torch.zeros(1, 2)
    result = selected_nll_backward(x, torch.tensor([0]), torch.zeros(1), torch.zeros(1),
                                    torch.tensor(1.), torch.tensor(1.))
    torch.testing.assert_close(result, torch.tensor([[0., 1.]], dtype=torch.float64))


@pytest.mark.parametrize('scale', [0., 1., 100.])
@pytest.mark.parametrize('cotangent', [-3., 0., 2.])
def test_backward_agrees_across_saturation_and_signed_cotangents(scale, cotangent):
    generator = torch.Generator().manual_seed(719)
    x = (torch.randn(5, 17, generator=generator, dtype=torch.float64)*scale).requires_grad_()
    labels = torch.tensor([0, 16, -100, 3, 8])
    maximum = x.detach().max(1).values
    logsum = torch.logsumexp(x.detach()-maximum[:, None], dim=1)
    result = selected_nll_backward(x.detach(), labels, maximum, logsum,
                                   torch.tensor(4.), torch.tensor(cotangent))
    expected = torch.autograd.grad(torch.nn.functional.cross_entropy(x, labels)*cotangent, x)[0]
    torch.testing.assert_close(result, expected, atol=1e-14, rtol=1e-12)


def test_nonfinite_saved_state_and_cotangent_rejected():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='Finite'):
        selected_nll_backward(x, torch.tensor([0]), torch.zeros(1), torch.tensor([float('nan')]),
                              torch.tensor(1.), torch.tensor(1.))
    with pytest.raises(ValueError, match='grad_output'):
        selected_nll_backward(x, torch.tensor([0]), torch.zeros(1), torch.zeros(1),
                              torch.tensor(1.), torch.tensor(float('inf')))
