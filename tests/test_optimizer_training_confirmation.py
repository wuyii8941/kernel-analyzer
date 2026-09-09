import pytest

from scripts.run_optimizer_training_confirmation import (
    CONDITIONS, STEPS, selected_start_blocks, summarize, t_interval,
)


def test_stream_selection_is_deterministic_and_separated():
    values = selected_start_blocks()
    assert values == [21871, 18687, 31029, 81149, 26283, 71944, 11720, 47948]
    assert len(set(values)) == 8
    assert min(abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]) >= 1200


def stream(index, reference, block256, block64):
    values = {"FP32_ADAMW": reference, "ADAMW8BIT_BLOCK256": block256,
              "ADAMW8BIT_BLOCK64": block64}
    return {"stream_index": index, "status": "COMPLETE", "records": [
        {"condition": condition,
         "evaluation_loss_by_step": {str(STEPS): [value, value]}}
        for condition, value in values.items()
    ]}


def protocol():
    return {"material_loss_margin": 0.01,
            "primary_contrast": "ADAMW8BIT_BLOCK256_MINUS_FP32_ADAMW"}


def test_confirmation_detects_material_effect_and_successful_modification():
    streams = [stream(i, 3.0, 3.02 + i * 0.0001, 3.002 + i * 0.00001)
               for i in range(8)]
    result = summarize(protocol(), streams)
    assert result["primary"]["decision"] == "MATERIAL_EFFECT"
    assert result["modified_variant"]["decision"] == "CONFIRMED_CLOSER_TO_REFERENCE"


def test_confirmation_does_not_promote_uncertain_result():
    signs = [-1, 1] * 4
    streams = [stream(i, 3.0, 3.0 + signs[i] * 0.02,
                      3.0 + signs[i] * (0.019 if i % 4 < 2 else 0.021))
               for i in range(8)]
    result = summarize(protocol(), streams)
    assert result["primary"]["decision"] == "INCONCLUSIVE"
    assert result["modified_variant"]["decision"] == "NOT_CONFIRMED"


def test_interval_requires_independent_values():
    with pytest.raises(ValueError):
        t_interval([1.0])
