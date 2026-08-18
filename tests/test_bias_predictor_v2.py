import pytest

from kernel_analyzer.bias_predictor import ReferenceOnlyInputs, validate_reference_payload


def test_reference_only_inputs_reject_candidate_and_verdict_leaks():
    with pytest.raises(ValueError):
        ReferenceOnlyInputs({"x": 1}, {}, {}, {"t4_verdict": "PASS"})
    with pytest.raises(ValueError):
        validate_reference_payload({"candidate_output": [1, 2]})


def test_reference_only_inputs_accept_reference_semantics():
    value = ReferenceOnlyInputs(
        {"operand_range": 2.0},
        {"reduction_extent": 128},
        {"vjp": "analytic"},
        {"semantic_region": "mm"},
    )
    assert value.reference_operands["operand_range"] == 2.0
