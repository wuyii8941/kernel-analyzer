import pytest

from scripts.run_optimizer_quantization_mechanism import summarize


def protocol():
    return {
        "case_id": "case",
        "prediction_fixed_before_variants": {"success_rule": "frozen"},
    }


def raw(values):
    return {
        "schema": "optimizer-quantization-mechanism-raw-v1",
        "variants": {
            str(size): {"confirmation_relative_rms": {
                "MOMENT1_STATE": row[0], "MOMENT2_STATE": row[1],
                "PARAMETER_WRITE": row[2],
            }} for size, row in values.items()
        },
    }


def test_frozen_block_order_prediction_is_machine_checked():
    result = summarize(protocol(), raw({
        64: (0.1, 0.2, 0.3), 256: (0.2, 0.1, 0.4), 1024: (0.3, 0.4, 0.5),
    }))
    assert result["prediction_result"] == "CONFIRMED"
    assert result["recommended_training_variant"] == 64
    assert result["training_selection_status"] == "READY_FOR_HUMAN_REVIEW"


def test_failed_primary_order_is_not_promoted_to_training():
    result = summarize(protocol(), raw({
        64: (0.1, 0.2, 0.3), 256: (0.2, 0.3, 0.4), 1024: (0.15, 0.4, 0.5),
    }))
    assert result["prediction_result"] == "NOT_CONFIRMED"
    assert result["recommended_training_variant"] is None


def test_missing_variant_fails_closed():
    with pytest.raises(ValueError, match="incomplete"):
        summarize(protocol(), raw({64: (0.1, 0.2, 0.3), 256: (0.2, 0.3, 0.4)}))
