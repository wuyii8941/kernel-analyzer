"""All public reductions: exact independent formulas, output shape and complete VJPs."""
from fractions import Fraction as Q

import pytest
import torch
import torch.nn.functional as F

from kernel_spec import registry
from kernel_spec.num import evaluate

FULL = 'torch210_unweighted_f32_same_shape'
MEAN = 'torch210_mean_f32_same_shape'


def tensor(v):
    return {'t': 'f32', 'v': v}


def request(kind, reduction, x, target, g=None):
    return {'op': ('vjp.' if g is not None else '')+'aten.'+kind+'_loss',
            'args': ([tensor(g)] if g is not None else [])+[tensor(x), tensor(target)],
            'kwargs': {'reduction': reduction}}


def ref(kind, query, profile=FULL):
    return registry.load(kind.upper()).adapter.reference(profile, query)


@pytest.mark.parametrize('kind', ['mse', 'l1'])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_exact_formula_complete_public_outputs_and_vjp(kind, reduction):
    x = [[0., -.5, 1.], [2., -3., .25]]
    target = [[0., .5, -.5], [-1., -3., .5]]
    g = [[-2., 0., .5], [3., -1., 2.]] if reduction == 'none' else -2.
    losses, dx = [], []
    for row in range(2):
        values, derivatives = [], []
        for col in range(3):
            d = Q(x[row][col])-Q(target[row][col])
            values.append(d*d if kind == 'mse' else abs(d))
            derivative = 2*d if kind == 'mse' else Q(int(d>0)-int(d<0))
            derivatives.append(derivative*Q(g[row][col] if reduction=='none' else g)/(6 if reduction=='mean' else 1))
        losses.append(values);dx.append(derivatives)
    expected = losses if reduction=='none' else sum(sum(row) for row in losses)/(6 if reduction=='mean' else 1)
    fwd, bwd = request(kind,reduction,x,target),request(kind,reduction,x,target,g)
    assert ref(kind,fwd).fields['out'] == expected
    assert ref(kind,bwd).fields['out'] == [dx,[[-v for v in row] for row in dx]]
    a,b = torch.tensor(x,requires_grad=True),torch.tensor(target,requires_grad=True)
    out = getattr(F,kind+'_loss')(a,b,reduction=reduction)
    da,db = torch.autograd.grad(out,(a,b),torch.tensor(g))
    torch.testing.assert_close(out.detach(),torch.tensor(expected,dtype=torch.float32))
    torch.testing.assert_close(da,torch.tensor(dx,dtype=torch.float32))
    torch.testing.assert_close(db,-torch.tensor(dx,dtype=torch.float32))


@pytest.mark.parametrize('kind', ['mse', 'l1'])
def test_none_full_shape_and_no_scalar_upstream_broadcast(kind):
    q = request(kind,'none',[[0.,2.]],[[0.,0.]],[[3.,-.5]])
    adapter = registry.load(kind.upper()).adapter
    expected = [[0.,-2.]] if kind=='mse' else [[0.,-.5]]
    good = {'out':[{'t':'f32','shape':[1,2],'v':expected},
                   {'t':'f32','shape':[1,2],'v':[[-v for v in expected[0]]]}]}
    assert adapter.check_output(FULL,q,good)
    assert not adapter.check_output(FULL,q,{'out':good['out'][:1]})
    forward = request(kind,'none',[[0.,2.]],[[0.,0.]])
    assert not adapter.check_output(FULL,forward,{'out':{'t':'f32','shape':[],'v':4. if kind=='mse' else 2.}})
    q['args'][0] = tensor(1.)
    assert evaluate(ref,kind,q).outcome == 'invalid_input'
    q['args'][0] = tensor([[1.,1.]])
    q['kwargs']['reduction'] = 'sum'
    assert evaluate(ref,kind,q).outcome == 'invalid_input'


@pytest.mark.parametrize('kind', ['mse', 'l1'])
def test_old_mean_profile_still_refuses_other_tasks_and_has_same_answer(kind):
    q = request(kind,'mean',[.5,-1.,0.],[0.,.5,0.],2.)
    assert ref(kind,q,MEAN).fields == ref(kind,q,FULL).fields
    q['kwargs']['reduction'] = 'sum'
    assert evaluate(ref,kind,q,MEAN).outcome == 'unsupported'
    assert evaluate(ref,kind,q,FULL).outcome == 'value'


@pytest.mark.parametrize('kind', ['mse', 'l1'])
def test_scalar_domain_and_unselected_options(kind):
    for reduction in ['none','sum','mean']:
        q=request(kind,reduction,2.,1.,-.5)
        assert ref(kind,q).fields['out'] == ([-1.,1.] if kind=='mse' else [-.5,.5])
    q=request(kind,'bogus',[1.],[0.])
    assert evaluate(ref,kind,q).outcome=='invalid_input'
    q['kwargs']={'reduction':'none','weight':None}
    assert evaluate(ref,kind,q).outcome=='unsupported'
    q['kwargs']={'reduction':'sum'}
    q['args'][0]=tensor([Q(1,10)])
    assert evaluate(ref,kind,q).outcome=='invalid_input'
