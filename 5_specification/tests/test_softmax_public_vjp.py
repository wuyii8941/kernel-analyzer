"""Original-input softmax VJP, independently checked with a dense Arb Jacobian."""
from fractions import Fraction as Q

import flint
import numpy as np
import pytest
import torch

from kernel_spec import registry
from kernel_spec.num import ends, evaluate


def query(x, g, dim):
    return {"op": "vjp.aten.softmax", "args": [
        {"t": "f32", "v": g}, {"t": "f32", "v": x}, dim]}


def reference(q):
    return registry.load("SOFTMAX").adapter.reference("real", q)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_independent_dense_jacobian(dim):
    x = np.array([[-2., .5, 3.], [1., -1., .25]])
    g = np.array([[.25, -2., 1.], [-.5, 3., 2.]])
    result = np.array(reference(query(x.tolist(), g.tolist(), dim)).fields["out"], dtype=object)
    old = flint.ctx.prec
    try:
        flint.ctx.prec = 512
        for row_x, row_g, row_out in zip(np.moveaxis(x, dim, -1).reshape(-1, x.shape[dim]),
                                       np.moveaxis(g, dim, -1).reshape(-1, x.shape[dim]),
                                       np.moveaxis(result, dim, -1).reshape(-1, x.shape[dim])):
            exps = [flint.arb(float(v)).exp() for v in row_x]
            p = [v / sum(exps) for v in exps]
            for j, actual in enumerate(row_out):
                expected = sum(flint.arb(float(row_g[i])) * p[i] * ((i == j) - p[j]) for i in range(len(p)))
                def endpoint(value):
                    m, e = value.man_exp()
                    return Q(int(m)) * Q(2) ** int(e)
                lo, hi = ends(actual)
                assert lo <= endpoint(expected.lower()) <= endpoint(expected.upper()) <= hi
    finally:
        flint.ctx.prec = old


def test_exact_uniform_logits_with_nonuniform_upstream():
    assert reference(query([[0., 0.]], [[1., 0.]], 1)).fields["out"] == [[Q(1, 4), Q(-1, 4)]]


def test_public_autograd_crosscheck_with_nonconstant_upstream():
    x = torch.tensor([[0., 1., -2.], [3., -1., .5]], requires_grad=True)
    g = torch.tensor([[1., -2., .5], [-3., 2., 0.]])
    dx, = torch.autograd.grad(torch.softmax(x, 1), x, g)
    ref = reference(query(x.detach().tolist(), g.tolist(), 1)).fields["out"]
    points = [[float(sum(ends(v)) / 2) for v in row] for row in ref]
    torch.testing.assert_close(dx, torch.tensor(points), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("q,outcome", [
    (query([[0., 0.]], [1., 0.], 1), "invalid_input"),
    (query([[0., float('-inf')]], [[1., 0.]], 1), "unsupported"),
    (query([[0., 0.]], [[1., float('inf')]], 1), "unsupported"),
    (query([[0., 0.]], [[1., 0.]], True), "invalid_input"),
])
def test_domain_refusal(q, outcome):
    assert evaluate(reference, q).outcome == outcome


def test_scalar_empty_and_dtype_validation():
    assert reference(query(2., 3., 0)).fields["out"] == 0
    assert reference(query([[]], [[]], 1)).fields["out"] == [[]]
    q = query([[0., 0.]], [[1., 0.]], 1)
    q['args'][0]['t'] = 'f64'
    assert evaluate(reference, q).outcome == 'invalid_input'


def test_complete_output_checker_rejects_omitted_coordinate():
    adapter = registry.load('SOFTMAX').adapter
    q = query([[0., 0.]], [[1., 0.]], 1)
    assert adapter.check_output('real', q, {'out': {'t': 'f32', 'v': [[Q(1, 4), Q(-1, 4)]], 'shape': [1, 2]}})
    assert not adapter.check_output('real', q, {'out': {'t': 'f32', 'v': [[Q(1, 4)]], 'shape': [1, 1]}})
