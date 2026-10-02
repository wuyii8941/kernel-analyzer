import pytest
import torch

from kernel_analyzer.tilelang_adapter import (
    TileLangCallConfig,
    check_tilelang_bias,
    check_tilelang_odd_symmetry,
    check_tilelang_reduction_order,
    check_tilelang_softmax_saved_state,
    compile_tilelang_program,
    select_tilelang_output,
    wrap_tilelang_callable,
)


def _inputs(index):
    return (torch.tensor([float(index + 1,)], dtype=torch.float32),)


def test_output_selection_supports_tuple_and_mapping():
    assert select_tilelang_output((1, 2), TileLangCallConfig(output_index=1)) == 2
    assert select_tilelang_output({"out": 3}, TileLangCallConfig(output_key="out")) == 3


def test_output_selection_rejects_ambiguous_config():
    with pytest.raises(ValueError):
        TileLangCallConfig(output_index=0, output_key="out")


def test_wrapper_preserves_callable_arguments_and_selects_output():
    def kernel(x):
        return (x + 1, x + 2)

    wrapped = wrap_tilelang_callable(kernel, output_index=1)
    assert torch.equal(wrapped(torch.tensor([2.])), torch.tensor([4.]))


def test_tilelang_check_reuses_common_checker():
    report = check_tilelang_bias(
        lambda x: x + 0.01,
        lambda x: x,
        _inputs,
        samples=8,
        calibration_samples=4,
    )
    assert report["backend"] == "tilelang"
    assert report["backend_contract"]["automatic_root_cause_analysis"] is False
    assert report["backend_contract"]["autotune_selection_must_be_frozen_before_confirmation"] is True
    assert "stages" in report


def test_tilelang_check_forwards_effect_margins_to_positive_control():
    report = check_tilelang_bias(
        lambda x: x + 1.0,
        lambda x: x,
        _inputs,
        samples=8,
        calibration_samples=4,
        directional_margin=0.1,
        aligned_projection_margin=0.1,
    )
    assert report["status"] == "SYSTEMATIC_BIAS_CONFIRMED"
    assert report["output"]["directional_margin"] == pytest.approx(0.1)


def test_tilelang_odd_symmetry_needs_only_candidate_and_inputs():
    report = check_tilelang_odd_symmetry(
        lambda x: x + 0.25,
        _inputs,
        samples=4,
    )
    assert report["backend"] == "tilelang"
    assert report["family"] == "odd_symmetry"
    assert report["property_decision"] == "PROPERTY_VIOLATION_OBSERVED"
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_compile_reports_optional_dependency():
    try:
        import tilelang  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="TileLang is not installed"):
            compile_tilelang_program(object())


def test_tilelang_softmax_family_reuses_property_contract():
    def candidate(scores, maximum, denominator):
        return {"probability": (scores - maximum[:, None]).exp() / denominator[:, None]}

    def make_inputs(index):
        del index
        scores = torch.zeros((2, 4), dtype=torch.float32)
        maximum = torch.zeros((2,), dtype=torch.float32)
        denominator = torch.full((2,), 4.0, dtype=torch.float32)
        return scores, maximum, denominator

    report = check_tilelang_softmax_saved_state(
        candidate,
        make_inputs,
        candidate_output_key="probability",
        samples=4,
    )
    assert report["backend"] == "tilelang"
    assert report["family"] == "softmax_saved_probability"
    assert report["input_consistency"]["status"] == "PASSED"
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_tilelang_softmax_family_retains_property_violation():
    def candidate(scores, maximum, denominator):
        return (scores - maximum[:, None]).exp() / denominator[:, None]

    def make_inputs(index):
        del index
        return (
            torch.zeros((2, 4), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float32),
            torch.full((2,), 8.0, dtype=torch.float32),
        )

    report = check_tilelang_softmax_saved_state(candidate, make_inputs, samples=4)
    assert report["input_consistency"]["status"] == "VIOLATION_OBSERVED"
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_tilelang_reduction_family_wraps_candidate_and_variant_outputs():
    def candidate(values):
        return {"out": values.sum(dim=-1)}

    def variant(values):
        return {"out": values.flip(-1).sum(dim=-1)}

    def make_inputs(index):
        return (torch.arange(8, dtype=torch.float32).reshape(2, 4) + index,)

    def make_negated_inputs(index):
        return (-make_inputs(index)[0],)

    report = check_tilelang_reduction_order(
        candidate,
        variant,
        make_inputs,
        candidate_output_key="out",
        variant_output_key="out",
        make_negated_inputs=make_negated_inputs,
        samples=4,
    )
    assert report["backend"] == "tilelang"
    assert report["family"] == "same_dtype_reduction_order"
    assert report["measurement_status"] == "VALID"
    assert report["symmetry"]["status"] == "VALID"
    assert report["bias_decision"] == "NOT_ASSESSED"
