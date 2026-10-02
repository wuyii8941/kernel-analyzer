import torch

from kernel_analyzer import (
    check_odd_symmetry,
    check_reduction_order,
    check_softmax_saved_state,
    diagnose_kernel,
)


def test_odd_symmetry_probe_uses_one_callable_and_detects_fixed_offset():
    def candidate(x):
        return 1.5 * x + 0.25

    report = check_odd_symmetry(
        candidate,
        lambda index: (torch.tensor([float(index + 1)]),),
        samples=4,
    )

    assert report["measurement_status"] == "VALID"
    assert report["property_decision"] == "PROPERTY_VIOLATION_OBSERVED"
    assert report["bias_signal"] == "BIAS_CANDIDATE_PROPERTY_VIOLATION"
    assert report["bias_decision"] == "NOT_ASSESSED"
    assert "METAMORPHIC_PROPERTY_VIOLATION" in report["evidence_classes"]


def test_odd_symmetry_probe_passes_signed_linear_callable():
    report = check_odd_symmetry(
        lambda x: 1.5 * x,
        lambda index: (torch.tensor([float(index + 1)]),),
        samples=4,
    )

    assert report["measurement_status"] == "VALID"
    assert report["property_decision"] == "PROPERTY_NOT_VIOLATED_IN_DECLARED_SAMPLE"
    assert report["bias_signal"] == "NO_PROPERTY_VIOLATION_OBSERVED"
    assert report["bias_decision"] == "NOT_ASSESSED"


from kernel_analyzer.fp32_reduction_order import balanced_fp32_sum, sequential_fp32_sum
from kernel_analyzer.softmax_saved_state_diagnostic import evaluate as evaluate_saved_state


def test_existing_same_dtype_reduction_family_is_detected_as_controlled_evidence():
    values = torch.tensor([1.0e8, 1.0, -1.0e8, 1.0], dtype=torch.float32)
    report = diagnose_kernel(
        sequential_fp32_sum,
        lambda index: (values.clone(),),
        rounding_variants={"BALANCED_FP32_ORDER": balanced_fp32_sum},
        samples=4,
    )

    assert report["measurement_status"] == "VALID"
    assert "ROUNDING_VARIANT_DIFFERENCE_OBSERVED" in report["evidence_classes"]
    assert "CONTROLLED_DIRECTIONAL_EVIDENCE" in report["evidence_classes"]
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_controlled_arithmetic_difference_is_reported_without_bias_verdict():
    def candidate(x):
        # Deliberately emulate a low-precision accumulator before a sum.
        values = x.to(torch.float16)
        total = values[:, 0]
        for column in range(1, values.shape[1]):
            total = total + values[:, column]
        return total.to(torch.float32)

    def fp32_variant(x):
        return x.sum(dim=-1)

    def make_inputs(index):
        del index
        return (torch.tensor([[4096.0, 1.0, -1.0, 0.25]], dtype=torch.float32),)

    report = diagnose_kernel(
        candidate,
        make_inputs,
        rounding_variants={"FP32_ACCUMULATION": fp32_variant},
        samples=4,
    )

    assert report["measurement_status"] == "VALID"
    assert report["diagnostic_status"] == "EVIDENCE_ONLY"
    assert report["bias_decision"] == "NOT_ASSESSED"
    assert "ROUNDING_VARIANT_DIFFERENCE_OBSERVED" in report["evidence_classes"]
    assert "CONTROLLED_DIRECTIONAL_EVIDENCE" in report["evidence_classes"]
    assert report["rounding_variants"]["FP32_ACCUMULATION"]["difference_observed_count"] == 4


def test_saved_state_consistency_violation_is_reported_without_reference():
    def candidate(scores, maximum, denominator):
        return (scores - maximum[:, None]).exp() / denominator[:, None]

    def consistency(index, args, kwargs, output):
        del index, kwargs
        mass = output.sum(dim=-1)
        defect = (mass - 1.0).abs()
        return {
            "passed": bool(torch.all(defect <= 1e-6)),
            "max_row_mass_defect": float(defect.max().item()),
            "saved_denominator": float(args[2][0].item()),
        }

    def make_inputs(index):
        del index
        return (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
            torch.full((1,), 8.0, dtype=torch.float32),
        )

    report = diagnose_kernel(
        candidate,
        make_inputs,
        input_consistency_check=consistency,
        samples=4,
    )

    assert report["bias_decision"] == "NOT_ASSESSED"
    assert report["input_consistency"]["status"] == "VIOLATION_OBSERVED"
    assert "INPUT_CONSISTENCY_VIOLATION" in report["evidence_classes"]


def test_existing_saved_probability_diagnostic_is_detected_as_consistency_evidence():
    def candidate(scores, maximum, denominator):
        return evaluate_saved_state(
            scores.to(torch.bfloat16), maximum, denominator, scale=1.0
        )["reconstructed_probability"]

    def consistency(index, args, kwargs, output):
        del index, kwargs
        defect = (output.sum(dim=-1) - 1.0).abs()
        return {"passed": bool(torch.all(defect <= 1e-6)), "max_defect": float(defect.max())}

    def make_inputs(index):
        del index
        return (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
            torch.full((1,), 8.0, dtype=torch.float32),
        )

    report = diagnose_kernel(
        candidate,
        make_inputs,
        input_consistency_check=consistency,
        samples=4,
    )

    assert report["input_consistency"]["status"] == "VIOLATION_OBSERVED"
    assert "INPUT_CONSISTENCY_VIOLATION" in report["evidence_classes"]
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_declared_spec_deviation_is_separate_from_bias():
    def candidate(x):
        return x * torch.sigmoid(x)

    def declared_spec(index, args, kwargs, output):
        del index, kwargs
        expected = args[0]
        error = (output - expected).abs().max()
        return {"passed": bool(error <= 1e-6), "max_abs_spec_error": float(error.item())}

    report = diagnose_kernel(
        candidate,
        lambda index: (torch.tensor([1.0 + index]),),
        specification_check=declared_spec,
        samples=4,
    )

    assert report["declared_specification"]["status"] == "VIOLATION_OBSERVED"
    assert "DECLARED_SPEC_DEVIATION" in report["evidence_classes"]
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_missing_contracts_are_explicitly_not_declared():
    report = diagnose_kernel(
        lambda x: x,
        lambda index: (torch.tensor([float(index)]),),
        samples=2,
    )

    assert report["diagnostic_status"] == "EVIDENCE_ONLY"
    assert report["rounding_variants"] == {}
    assert report["input_consistency"]["status"] == "NOT_DECLARED"
    assert report["declared_specification"]["status"] == "NOT_DECLARED"
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_softmax_family_contract_is_reusable_without_a_reference():
    def candidate(scores, maximum, denominator):
        return (scores - maximum[:, None]).exp() / denominator[:, None]

    def inputs(index):
        del index
        return (
            torch.zeros((2, 4), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float32),
            torch.full((2,), 4.0, dtype=torch.float32),
        )

    report = check_softmax_saved_state(candidate, inputs, samples=8)
    assert report["family"] == "softmax_saved_probability"
    assert report["measurement_status"] == "VALID"
    assert report["input_consistency"]["status"] == "PASSED"
    assert report["input_consistency"]["property_mean_inference"]["decision"] == (
        "NOT_IDENTIFIABLE"
    )  # identical zero residuals are not a population theorem
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_softmax_family_contract_reports_a_saved_state_violation():
    def candidate(scores, maximum, denominator):
        return (scores - maximum[:, None]).exp() / denominator[:, None]

    def inputs(index):
        del index
        return (
            torch.zeros((2, 4), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float32),
            torch.full((2,), 8.0, dtype=torch.float32),
        )

    report = check_softmax_saved_state(candidate, inputs, samples=8)
    assert "INPUT_CONSISTENCY_VIOLATION" in report["evidence_classes"]
    endpoint = report["input_consistency"]["property_mean_inference"]
    assert endpoint["decision"] == "NOT_IDENTIFIABLE"
    assert report["bias_decision"] == "NOT_ASSESSED"


def test_reduction_family_adds_declared_odd_symmetry_check():
    def sequential(x):
        total = x[0]
        for value in x[1:]:
            total = total + value
        return total

    def balanced(x):
        return (x[0] + x[1]) + (x[2] + x[3])

    def inputs(index):
        del index
        return (torch.tensor([1.0e8, 1.0, -1.0e8, 1.0], dtype=torch.float32),)

    def negated(index):
        return tuple(-value for value in inputs(index))

    report = check_reduction_order(
        sequential,
        balanced,
        inputs,
        samples=8,
        make_negated_inputs=negated,
    )
    variant = report["rounding_variants"]["DECLARED_REDUCTION_ORDER_VARIANT"]
    assert report["family"] == "same_dtype_reduction_order"
    assert report["symmetry"]["status"] == "VALID"
    assert max(item["relative_odd_residual"] for item in report["symmetry"]["records"]) == 0.0
    assert variant["signed_mean_inference"]["decision"] == "NOT_IDENTIFIABLE"
    assert report["bias_decision"] == "NOT_ASSESSED"
