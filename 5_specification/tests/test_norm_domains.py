"""Input-domain distinctions required by the PyTorch 2.10 normalisation interfaces."""
from fractions import Fraction

from kernel_spec import registry
from kernel_spec.num import evaluate


def _query(profile, **updates):
    inp = {"op": "layer_norm", "x": [[-1, 1]], "normalized_shape": [2],
           "weight": None, "bias": None, "eps": Fraction(-3, 4), "upstream": [[1, 0]]}
    inp.update(updates)
    return evaluate(registry.load("NORM").adapter.reference, profile, inp)


def test_negative_eps_is_valid_when_every_radicand_is_positive():
    result = _query("torch")
    assert result.outcome == "value", result.note
    assert result.fields == {
        "y": [[-2, 2]],
        "saved": {"mean": [0], "rstd": [2]},
        "grad": {"x": [[-3, 3]]},
    }


def test_nonpositive_radicand_is_mathematically_undefined():
    for eps in (-1, -2):
        result = _query("torch", eps=eps)
        assert result.outcome == "domain_undefined", (eps, result.outcome, result.note)


def test_public_group_norm_rejects_the_single_value_configuration():
    group = {"op": "group_norm", "x": [[1]], "num_groups": 1,
             "weight": None, "bias": None, "eps": Fraction(1, 4), "upstream": [[1]]}
    reference = registry.load("NORM").adapter.reference
    native = evaluate(reference, "torch", group)
    public = evaluate(reference, "torch_autograd", group)
    assert native.outcome == "value", native.note
    assert native.fields["y"] == [[0]]
    assert native.fields["saved"] == {"mean": [[1]], "rstd": [[2]]}
    assert native.fields["grad"] == {"x": [[0]]}
    assert public.outcome == "invalid_input", public.note


def test_public_group_norm_uses_the_exact_pytorch_batch_size_check():
    group = {"op": "group_norm", "x": [[1], [2]], "num_groups": 1,
             "weight": None, "bias": None, "eps": Fraction(1, 4), "upstream": [[1], [1]]}
    result = evaluate(registry.load("NORM").adapter.reference, "torch_autograd", group)
    assert result.outcome == "value", result.note
    assert result.fields == {"y": [[0], [0]], "grad": {"x": [[0], [0]]}}
