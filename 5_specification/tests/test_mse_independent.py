"""Independent checks for the reviewed PyTorch 2.10 mean-MSE profile."""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as torch_functional

from kernel_spec import registry
from kernel_spec.num import evaluate


MSE = registry.load("MSE").adapter
PROFILE = "torch210_mean_f32_same_shape"


def T(values, shape=None, dtype="f32"):
    out = {"t": dtype, "v": values}
    if shape is not None:
        out["shape"] = shape
    return out


def forward_query(values, target, *, reduction="mean"):
    return {
        "op": "aten.mse_loss",
        "args": [T(values), T(target)],
        "kwargs": {"reduction": reduction},
    }


def vjp_query(upstream, values, target, *, reduction="mean"):
    return {
        "op": "vjp.aten.mse_loss",
        "args": [T(upstream, []), T(values), T(target)],
        "kwargs": {"reduction": reduction},
    }


def _flatten(values):
    if isinstance(values, list):
        return [item for value in values for item in _flatten(value)]
    return [Fraction(values)]


def _reshape(values, template):
    iterator = iter(values)

    def build(node):
        if isinstance(node, list):
            return [build(child) for child in node]
        return next(iterator)

    result = build(template)
    with pytest.raises(StopIteration):
        next(iterator)
    return result


def independent(values, target, upstream=1):
    flat_values, flat_target = _flatten(values), _flatten(target)
    assert len(flat_values) == len(flat_target) > 0
    differences = [a - b for a, b in zip(flat_values, flat_target)]
    loss = sum((difference * difference for difference in differences), Fraction()) / len(differences)
    dx = [Fraction(upstream) * 2 * difference / len(differences) for difference in differences]
    return loss, [_reshape(dx, values), _reshape([-value for value in dx], target)]


@pytest.mark.parametrize("values,target,upstream", [
    ([[1, 3]], [[0, 1]], 2),
    ([[-2, 0], [4, 8]], [[-2, 1], [2, 4]], -1),
    ([Fraction(1, 2), Fraction(-3, 4), 2], [0, Fraction(1, 4), -1], Fraction(1, 2)),
    ([[[1, 2]], [[-1, 0]]], [[[2, 0]], [[-1, 4]]], 0),
])
def test_reference_matches_independent_exact_formula(values, target, upstream):
    expected_loss, expected_vjp = independent(values, target, upstream)
    forward = forward_query(values, target)
    backward = vjp_query(upstream, values, target)
    forward_before, backward_before = copy.deepcopy(forward), copy.deepcopy(backward)

    assert MSE.reference(PROFILE, forward).fields["out"] == expected_loss
    assert MSE.reference(PROFILE, backward).fields["out"] == expected_vjp
    assert forward == forward_before
    assert backward == backward_before


def test_public_pytorch_function_and_autograd_follow_the_selected_task():
    generator = torch.Generator().manual_seed(20261019)
    checked = 0
    for shape in ((1,), (7,), (3, 5), (2, 3, 4)):
        for upstream in (0.5, 1.0, -2.0):
            values = torch.randn(shape, generator=generator, dtype=torch.float32, requires_grad=True)
            target = torch.randn(shape, generator=generator, dtype=torch.float32, requires_grad=True)
            values_before, target_before = values.detach().clone(), target.detach().clone()
            loss = torch_functional.mse_loss(values, target, reduction="mean")
            (loss * upstream).backward()

            reference_loss = MSE.reference(PROFILE, forward_query(values_before.tolist(), target_before.tolist()))
            reference_vjp = MSE.reference(
                PROFILE, vjp_query(upstream, values_before.tolist(), target_before.tolist()))
            dx, dtarget = reference_vjp.fields["out"]
            assert float(loss.detach()) == pytest.approx(float(reference_loss.fields["out"]), rel=2e-6, abs=2e-7)
            torch.testing.assert_close(values.grad.double(), torch.tensor(dx, dtype=torch.float64), rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(target.grad.double(), torch.tensor(dtarget, dtype=torch.float64), rtol=2e-6, atol=2e-7)
            assert torch.equal(values.detach(), values_before)
            assert torch.equal(target.detach(), target_before)
            checked += 1
    assert checked == 12


def test_complete_output_checker_requires_scalar_loss_and_both_gradients():
    forward = forward_query([[1, 3]], [[0, 1]])
    loss = {"out": {"t": "f32", "v": Fraction(5, 2), "shape": []}}
    assert MSE.check_output(PROFILE, forward, loss)
    assert not MSE.check_output(PROFILE, forward, {"out": {"t": "f32", "v": [Fraction(5, 2)], "shape": [1]}})

    backward = vjp_query(2, [[1, 3]], [[0, 1]])
    complete = {"out": [
        {"t": "f32", "v": [[2, 4]], "shape": [1, 2]},
        {"t": "f32", "v": [[-2, -4]], "shape": [1, 2]},
    ]}
    assert MSE.check_output(PROFILE, backward, complete)
    assert not MSE.check_output(PROFILE, backward, {"out": complete["out"][:1]})
    wrong_dtype = copy.deepcopy(complete)
    wrong_dtype["out"][1]["t"] = "f64"
    assert not MSE.check_output(PROFILE, backward, wrong_dtype)
    wrong_shape = copy.deepcopy(complete)
    wrong_shape["out"][0]["shape"] = [2]
    assert not MSE.check_output(PROFILE, backward, wrong_shape)
    wrong_value = copy.deepcopy(complete)
    wrong_value["out"][1]["v"][0][1] = 4
    assert not MSE.check_output(PROFILE, backward, wrong_value)


def test_unselected_calls_and_nonphysical_records_are_refused():
    base = forward_query([1, 2], [0, 1])
    cases = (
        ({**base, "args": [T([1, 2], dtype="f64"), T([0, 1])]}, "unsupported"),
        ({**base, "kwargs": {"reduction": "sum"}}, "unsupported"),
        ({**base, "kwargs": {"weight": None}}, "unsupported"),
        ({**base, "kwargs": {"size_average": None}}, "unsupported"),
        ({**base, "args": [T([Fraction(1, 10)]), T([0])]}, "invalid_input"),
        ({**base, "args": [T([], [0]), T([], [0])]}, "invalid_input"),
        ({**base, "args": [T([1, 2]), T([[0, 1]])]}, "invalid_input"),
        ({**base, "args": [T([float("inf"), 2]), T([0, 1])]}, "unsupported"),
        ({**base, "args": [T([float("nan"), 2]), T([0, 1])]}, "invalid_input"),
        ({"op": "vjp.aten.mse_loss", "args": [T([1]), T([1]), T([0])]}, "invalid_input"),
    )
    for query, outcome in cases:
        assert evaluate(MSE.reference, PROFILE, query).outcome == outcome
