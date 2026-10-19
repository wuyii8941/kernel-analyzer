from __future__ import annotations

import math

from kernel_spec import registry
from kernel_spec.num import evaluate


ACT = registry.load("ACT").adapter


def T(values, dtype="f32"):
    return {"t": dtype, "v": values}


def result(profile, op, args, kwargs=None):
    query = {"op": op, "args": args}
    if kwargs is not None:
        query["kwargs"] = kwargs
    return evaluate(ACT.reference, profile, query)


def test_runtime_nonfinite_values_follow_the_declared_domain():
    for value in (math.inf, -math.inf, "inf", "-inf"):
        assert result("torch", "aten.relu", [T([value])]).outcome == "unsupported"
    assert result("torch", "aten.relu", [T([math.nan])]).outcome == "invalid_input"


def test_profiles_refuse_nonfloating_tensor_records():
    for dtype in ("i64", "i32", "b8"):
        assert result("torch", "aten.relu", [T([1], dtype)]).outcome == "unsupported"


def test_selected_backward_boundary_requires_matching_shapes():
    binary = (
        ("aten.threshold_backward", [T([1]), T([[1], [2]]), 0]),
        ("aten.gelu_backward", [T([1]), T([[1], [2]])]),
        ("aten.silu_backward", [T([1]), T([[1], [2]])]),
        ("aten.softplus_backward", [T([1]), T([[1], [2]]), 1, 20]),
        ("aten.tanh_backward", [T([1]), T([[1], [2]])]),
        ("aten.sigmoid_backward", [T([1]), T([[1], [2]])]),
        ("vjp.aten.tanh", [T([1]), T([[1], [2]])]),
        ("vjp.aten.sigmoid", [T([1]), T([[1], [2]])]),
    )
    for op, args in binary:
        assert result("torch", op, args).outcome == "invalid_input"


def test_softplus_threshold_equality_uses_formula_backward_branch():
    got = result("torch", "aten.softplus_backward", [T([3]), T([2])], {"beta": 2, "threshold": 4})
    assert got.outcome == "value"
    value = got.fields["out"][0]
    lo, hi = value.a, value.b
    assert float(lo) < 3 and float(hi) < 3


def test_formula_profile_is_distinct_from_the_torch_branch():
    torch_result = result("torch", "aten.softplus", [T([21])], {"beta": 1, "threshold": 20})
    formula_result = result("softplus_formula", "aten.softplus", [T([21])],
                            {"beta": 1, "threshold": 20})
    assert torch_result.fields["out"][0] == 21
    lo, _ = formula_result.fields["out"][0].a, formula_result.fields["out"][0].b
    assert float(lo) > 21


def test_complete_output_checker_keeps_tensor_metadata_and_all_glu_gradients():
    relu = {"op": "aten.relu", "args": [T([-1, 2])]}
    assert ACT.check_output("torch", relu, {"out": {"t": "f32", "v": [0, 2], "shape": [2]}})
    assert not ACT.check_output("torch", relu, {"out": {"t": "f64", "v": [0, 2], "shape": [2]}})
    assert not ACT.check_output("torch", relu, {"out": {"t": "f32", "v": [0], "shape": [1]}})

    backward = {"op": "act.swiglu_backward", "args": [T([1]), T([0]), T([2])]}
    complete = {"out": [
        {"t": "f32", "v": [1], "shape": [1]},
        {"t": "f32", "v": [0], "shape": [1]},
    ]}
    assert ACT.check_output("torch", backward, complete)
    assert not ACT.check_output("torch", backward, {"out": complete["out"][:1]})
