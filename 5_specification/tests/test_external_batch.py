"""Reporting must retain refusals and distinguish equality from enclosure overlap."""
import importlib.util
from pathlib import Path
from fractions import Fraction

import pytest
import torch

from kernel_spec.num import from_interval


path = Path(__file__).resolve().parents[1] / "corpus/external_batch/run.py"
spec = importlib.util.spec_from_file_location("external_batch", path)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_overlap_is_not_equality_and_other_contradictions_are_retained():
    got = batch.reference_comparison(torch.tensor([1., 3., 4.]),
                                    [from_interval(0, 2), Fraction(2), Fraction(4)])
    assert got["inside_nonpoint_unresolved"] == 1
    assert got["outside_mathematical_enclosure"] == 1
    assert got["exact_equal"] == 1
    assert got["residual_intervals"][1] == {"lower": "1", "upper": "1"}


def test_missing_gradient_cannot_pass_standard_comparison():
    with pytest.raises(ValueError, match="missing"):
        batch.standard_comparison({"out": torch.tensor(0.)},
                                  {"out": torch.tensor(0.), "dx": torch.tensor([0.])})


@pytest.mark.parametrize("observed", [torch.ones(1, 2), torch.ones(2, dtype=torch.float64)])
def test_shape_and_dtype_cannot_be_hidden_by_flattening(observed):
    with pytest.raises(ValueError, match="shape/dtype"):
        batch.reference_comparison(observed, [Fraction(1), Fraction(1)])


def test_softmax_vjp_receives_original_input_and_upstream():
    queries, limitation = batch.mathematical_queries(
        {"op": "softmax", "contract": "SOFTMAX"}, torch.zeros(2, 4), None, torch.ones(2, 4))
    assert len(queries) == 2
    assert queries[0][-1] == ["out"]
    assert queries[1][-1] == ["dx"]
    assert queries[1][2]["op"] == "vjp.aten.softmax"
    assert queries[1][2]["args"][1]["v"] == [[0.] * 4] * 2
    assert not limitation


def test_missing_math_reference_does_not_invent_an_answer():
    queries, limitation = batch.mathematical_queries(
        {"op": "unassigned", "contract": None}, torch.ones(2, 4), torch.zeros(2, 4), torch.tensor(1.))
    assert queries == []
    assert "No admitted mathematical contract" in limitation


@pytest.mark.parametrize("op,parameters", [
    ("smooth_l1", {"reduction": "none", "beta": 0.}),
    ("smooth_l1", {"reduction": "sum", "beta": .1}),
    ("huber", {"reduction": "none", "delta": 2.}),
])
def test_selected_loss_parameters_reach_all_three_references(monkeypatch, op, parameters):
    from types import SimpleNamespace
    seen = []
    def constructor(**kwargs):
        seen.append(kwargs)
        return kwargs
    monkeypatch.setattr(batch.importlib, "import_module", lambda _: SimpleNamespace(**{batch.LOSSES[op]: constructor}))
    case = {"op": op, "library": "external", "contract": None, "parameters": parameters}
    _, standard = batch.functions(case)
    assert seen == [parameters]
    assert standard.reduction == parameters["reduction"]
    parameter = "beta" if op == "smooth_l1" else "delta"
    assert getattr(standard, parameter) == parameters[parameter]
    g = torch.ones(1, 2) if parameters["reduction"] == "none" else torch.tensor(1.)
    queries, limitation = batch.mathematical_queries(case, torch.ones(1, 2), torch.zeros(1, 2), g)
    assert not limitation
    assert all(query[2]["kwargs"] == parameters for query in queries)


def test_unknown_operation_cannot_silently_use_softmax_reference():
    queries, limitation = batch.mathematical_queries(
        {"op": "unassigned", "contract": "SOFTMAX"}, torch.ones(2, 4), None, torch.ones(2, 4))
    assert queries == []
    assert limitation


def test_explicit_mean_profile_keeps_reference_refusal_instead_of_substituting_mean():
    from kernel_spec import registry
    queries, _ = batch.mathematical_queries(
        {"op": "mse", "contract": "MSE", "profile": "torch210_mean_f32_same_shape",
         "parameters": {"reduction": "sum"}},
        torch.tensor([[1., 2.]]), torch.zeros(1, 2), torch.tensor(1.))
    assert queries[0][2]["kwargs"] == {"reduction": "sum"}
    cid, profile, query, _ = queries[0]
    assert registry.query(cid, profile, query).outcome == "unsupported"


def test_nonmean_loss_uses_reviewed_full_profile_without_changing_task():
    from kernel_spec import registry
    queries, limitation = batch.mathematical_queries(
        {"op": "mse", "contract": "MSE", "parameters": {"reduction": "sum"}},
        torch.tensor([[1., 2.]]), torch.zeros(1, 2), torch.tensor(1.))
    assert not limitation
    cid, profile, query, _ = queries[0]
    assert profile == "torch210_unweighted_f32_same_shape"
    result = registry.query(cid, profile, query)
    assert result.outcome == "value", result.note
    assert result.fields['out'] == 5


def test_unknown_loss_parameter_cannot_be_ignored():
    with pytest.raises(ValueError, match="parameter"):
        batch.mathematical_queries(
            {"op": "huber", "contract": None, "parameters": {"weight": 2}},
            torch.ones(1, 2), torch.zeros(1, 2), torch.tensor(1.))


@pytest.mark.parametrize("op,parameter", [("smooth_l1", "beta"), ("huber", "delta")])
def test_new_loss_reference_preserves_selected_task_and_both_gradients(op, parameter):
    from kernel_spec import registry
    queries, limitation = batch.mathematical_queries(
        {"op": op, "contract": None}, torch.tensor([[.25, 2.]]), torch.zeros(1, 2), torch.tensor(-2.))
    assert not limitation
    assert [q[-1] for q in queries] == [["out"], ["dx", "dtarget"]]
    for cid, profile, query, names in queries:
        assert cid == "HUBER"
        assert query["kwargs"] == {"reduction": "mean", parameter: .5}
        result = registry.query(cid, profile, query)
        assert result.outcome == "value", result.note
        if len(names) == 2:
            assert query["args"][0]["v"] == -2.
            dx, dt = result.fields["out"]
            expected = [-.5, -1.] if op == "smooth_l1" else [-.25, -.5]
            assert batch.reference_comparison(torch.tensor([expected]), dx)["exact_equal"] == 2
            assert batch.reference_comparison(-torch.tensor([expected]), dt)["exact_equal"] == 2
