from kernel_analyzer.analysis_result import AnalysisResult


def test_negative_measurement_is_still_a_valid_analysis() -> None:
    row = AnalysisResult(
        case_id="control",
        contrast_id="NATURAL_IMPLEMENTATION",
        measurement_status="VALID",
        claim_scope="FIXED_SUITE_UPDATE",
        bias_analysis={"direction": "NOT_CONFIRMED"},
        equivalence_decision="EQUIVALENT",
    ).to_dict()
    assert row["measurement_status"] == "VALID"
    assert row["bias_analysis"]["direction"] == "NOT_CONFIRMED"
    assert row["training_outcome"] == "NOT_MEASURED"
