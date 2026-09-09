import pytest

from scripts.run_optimizer_state_component_mechanism import summarize


def protocol():
    return {"case_id": "case", "prediction_fixed_before_component_variants": {}}


def raw(default=0.05, first=0.02, second=0.04, first_identity=0, second_identity=0):
    def row(m1, m2, write):
        return {"confirmation_relative_rms": {
            "MOMENT1_STATE": m1, "MOMENT2_STATE": m2, "PARAMETER_WRITE": write,
        }}
    return {"variants": {
        "DEFAULT_8BIT": row(0.03, 0.005, default),
        "FP32_FIRST_MOMENT": row(first_identity, 0.005, first),
        "FP32_SECOND_MOMENT": row(0.03, second_identity, second),
    }}


def test_component_prediction_requires_identity_and_parameter_reduction():
    result = summarize(protocol(), raw())
    assert result["prediction_result"] == "CONFIRMED"
    assert result["recommended_training_variant"] == "FP32_FIRST_MOMENT"


@pytest.mark.parametrize("kwargs", [
    {"first": 0.06}, {"first_identity": 1e-8}, {"second_identity": 1e-8},
])
def test_component_prediction_fails_closed(kwargs):
    assert summarize(protocol(), raw(**kwargs))["prediction_result"] == "NOT_CONFIRMED"
