"""Independent scalar formulas for the losses required by the attorch public-call trial."""
from fractions import Fraction as Q

import pytest
import torch
import torch.nn.functional as F

from kernel_spec import registry
from kernel_spec.num import evaluate

PROFILE = 'torch210_unweighted_f32'


def T(v):
    return {'t': 'f32', 'v': v}


def request(kind, x, y, reduction='mean', threshold=.5, g=None):
    kwargs = {'reduction': reduction, 'beta' if kind == 'smooth_l1' else 'delta': threshold}
    return {'op': ('vjp.' if g is not None else '') + 'aten.' + kind + '_loss',
            'args': ([T(g)] if g is not None else []) + [T(x), T(y)], 'kwargs': kwargs}


def ref(q):
    return registry.load('HUBER').adapter.reference(PROFILE, q)


@pytest.mark.parametrize('kind,threshold', [('smooth_l1', 0), ('smooth_l1', .5), ('smooth_l1', 2), ('huber', .5), ('huber', 2)])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_fraction_formula_and_complete_public_autograd(kind, threshold, reduction):
    x = [[-3., -.5, -.25, 0., .25, .5, 3.]]
    y = [[0.] * 7]
    upstream = [[-.5, 1., 2., 3., -2., .25, .5]] if reduction == 'none' else -2.
    losses, gradients = [], []
    b = Q(threshold)
    for i, v in enumerate(x[0]):
        d = Q(v)
        sign = int(d > 0) - int(d < 0)
        if kind == 'smooth_l1' and b == 0:
            loss, derivative = abs(d), Q(sign)
        elif abs(d) < b:
            loss, derivative = d*d/2, d
            if kind == 'smooth_l1':
                loss, derivative = loss/b, derivative/b
        else:
            loss, derivative = abs(d)-b/2, Q(sign)
            if kind == 'huber':
                loss, derivative = loss*b, derivative*b
        scale = Q(upstream[0][i] if reduction == 'none' else upstream)
        losses.append(loss)
        gradients.append(derivative*scale/(7 if reduction == 'mean' else 1))
    loss_expected = [losses] if reduction == 'none' else sum(losses)/(7 if reduction == 'mean' else 1)
    loss_result = ref(request(kind, x, y, reduction, threshold)).fields['out']
    dx, dy = ref(request(kind, x, y, reduction, threshold, upstream)).fields['out']
    assert loss_result == loss_expected
    assert dx == [gradients]
    assert dy == [[-v for v in gradients]]
    a, target = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
    kwargs = {'reduction': reduction, 'beta' if kind == 'smooth_l1' else 'delta': threshold}
    loss = getattr(F, kind + '_loss')(a, target, **kwargs)
    da, db = torch.autograd.grad(loss, (a, target), torch.tensor(upstream))
    torch.testing.assert_close(loss.detach(), torch.tensor(loss_expected, dtype=torch.float32), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(da, torch.tensor(dx, dtype=torch.float32), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(db, torch.tensor(dy, dtype=torch.float32), rtol=2e-6, atol=1e-7)


def test_none_requires_complete_upstream_and_target_gradient():
    q = request('huber', [[0., .25, 2.]], [[0., 0., 0.]], 'none', .5, [[1., 2., 3.]])
    adapter = registry.load('HUBER').adapter
    good = {'out': [{'t': 'f32', 'v': [[0, .5, 1.5]], 'shape': [1, 3]},
                    {'t': 'f32', 'v': [[0, -.5, -1.5]], 'shape': [1, 3]}]}
    assert adapter.check_output(PROFILE, q, good)
    assert not adapter.check_output(PROFILE, q, {'out': good['out'][:1]})
    q['args'][0] = T(1.)
    assert evaluate(ref, q).outcome == 'invalid_input'


@pytest.mark.parametrize('kind,threshold,outcome', [('huber', 0, 'invalid_input'), ('smooth_l1', -1, 'invalid_input'),
    ('huber', float('inf'), 'unsupported'), ('huber', float('nan'), 'invalid_input'),
    ('smooth_l1', True, 'invalid_input'), ('huber', Q(1, 10), 'invalid_input')])
def test_parameter_domain(kind, threshold, outcome):
    assert evaluate(ref, request(kind, [1.], [0.], threshold=threshold)).outcome == outcome


def test_physical_inputs_and_unselected_options():
    q = request('huber', [1.], [0.])
    q['args'][0] = T([Q(1, 10)])
    assert evaluate(ref, q).outcome == 'invalid_input'
    q['args'][0] = {'t': 'f64', 'v': [1.]}
    assert evaluate(ref, q).outcome == 'unsupported'
    q['args'][0] = T([1.])
    q['kwargs']['weight'] = None
    assert evaluate(ref, q).outcome == 'unsupported'


def test_binary64_threshold_is_not_silently_rounded_to_float32():
    result = ref(request('smooth_l1', [1/32], [0.], 'sum', .1)).fields['out']
    assert result == Q(1, 32)**2 / (2 * Q(.1))


@pytest.mark.parametrize('kind', ['smooth_l1', 'huber'])
def test_exact_central_difference_inside_each_polynomial_region(kind):
    for x in (-2., -.25, .25, 2.):
        h = Q(1, 64)
        upper = ref(request(kind, [Q(x)+h], [0.], 'sum', .5)).fields['out']
        lower = ref(request(kind, [Q(x)-h], [0.], 'sum', .5)).fields['out']
        dx, dy = ref(request(kind, [x], [0.], 'sum', .5, 1.)).fields['out']
        assert dx[0] == (upper-lower)/(2*h)
        assert dy[0] == -dx[0]
