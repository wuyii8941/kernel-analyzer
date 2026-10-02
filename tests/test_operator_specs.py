import pytest
import torch

from kernel_analyzer import SumSpec, check_tilelang_bias


def test_exact_sum_recovers_small_term_and_preserves_declared_shape():
    x = torch.tensor([[1e16, 1., -1e16], [2., 3., 4.]], dtype=torch.float64)
    assert torch.equal(SumSpec(keepdim=True).reference(x), torch.tensor([[1.], [9.]], dtype=torch.float64))
    y = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    assert torch.equal(SumSpec(axis=0, input_arg="values").reference(values=y), torch.tensor([4., 6.], dtype=torch.float64))


def test_sum_does_not_present_rounded_reference_as_exact():
    with pytest.raises(ValueError, match="not representable"):
        SumSpec().reference(torch.tensor([1e16, 1.], dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        SumSpec().reference(torch.tensor([float("inf")]))


def test_sum_contract_rejects_wrong_axis_and_excessive_size():
    with pytest.raises(ValueError, match="axis"):
        SumSpec(axis=3).reference(torch.ones(4))
    with pytest.raises(ValueError, match="max_elements"):
        SumSpec(max_elements=2).reference(torch.ones(4))


def test_builtin_spec_reuses_checker_and_rejects_ambiguous_semantics():
    def make_inputs(index):
        return (torch.tensor([2. ** 24, (index + 1) / 32, -2. ** 24]),)

    def sequential_sum(x):
        return (x[0] + x[1] + x[2]).reshape(1)

    report = check_tilelang_bias(sequential_sum, make_inputs=make_inputs,
                                specification=SumSpec(keepdim=True), samples=32)
    assert report["measurement_status"] == "VALID"
    assert report["status"] == "SYSTEMATIC_BIAS_CONFIRMED"
    assert report["output"]["mean_bias_decision"] == "CONFIRMED"
    assert report["backend_contract"]["reference_is_user_supplied"] is False
    assert report["specification"]["reference_source"] == "builtin_exact_rational_sum"
    with pytest.raises(ValueError, match="exactly one"):
        check_tilelang_bias(sequential_sum, make_inputs=make_inputs)
    with pytest.raises(ValueError, match="exactly one"):
        check_tilelang_bias(sequential_sum, lambda x: x, make_inputs, specification=SumSpec())
    with pytest.raises(ValueError, match="output checks"):
        check_tilelang_bias(sequential_sum, make_inputs=make_inputs,
                            specification=SumSpec(), check_backward=True)


def test_symmetric_scaling_is_not_labeled_fixed_mean_bias():
    report = check_tilelang_bias(
        lambda x: 1.25 * x, lambda x: x,
        lambda index: (torch.tensor([1. if index % 2 else -1.]),), samples=32,
    )
    assert report["output"]["aligned_effect_decision"] == "CONFIRMED"
    assert report["output"]["mean_bias_decision"] == "NOT_ASSESSED"
    assert report["output"]["mean_effect_norm_ratio"] == 0.


def test_softmax_mass_probe_does_not_hide_output_rounding():
    from kernel_analyzer import check_softmax_saved_state
    probability = torch.tensor([1., 2.**-12], dtype=torch.float16)
    assert probability.sum().item() == 1.
    report = check_softmax_saved_state(lambda x: x, lambda i: (probability,), samples=2)
    assert report["input_consistency"]["status"] == "VIOLATION_OBSERVED"
    assert report["input_consistency"]["property_residual_mean"] == 2.**-12
