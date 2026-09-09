import pytest

from scripts.run_optimizer_hybrid_training_confirmation import (
    CONDITIONS, EVALUATION_STEPS, SELECTED_STARTS, STEPS, summarize, validate_starts,
)


def stream(index, default, hybrid, reference=3.0):
    values = {"ADAMW8BIT_BLOCK256": default,
              "FP32_FIRST_MOMENT_BLOCK256": hybrid, "FP32_ADAMW": reference}
    return {"stream_index": index, "records": [
        {"condition": condition, "status": "COMPLETE",
         "training_loss": [value] * STEPS,
         "evaluation_loss_by_step": {str(step): [value, value]
                                     for step in EVALUATION_STEPS}}
        for condition, value in values.items()
    ], "status": "COMPLETE"}


def test_frozen_start_selection_reproduces():
    validate_starts()
    assert len(SELECTED_STARTS) == 8


def test_material_training_improvement_requires_interval_above_margin():
    protocol = {"material_improvement_margin": 0.01}
    streams = [stream(i, 3.03 + i * 0.0001, 3.01) for i in range(8)]
    assert summarize(protocol, streams)["primary"]["decision"] == "MATERIAL_IMPROVEMENT"


def test_unstable_training_improvement_is_not_confirmed():
    protocol = {"material_improvement_margin": 0.01}
    signs = [-1, 1] * 4
    streams = [stream(i, 3.02, 3.02 + signs[i] * 0.02) for i in range(8)]
    assert summarize(protocol, streams)["primary"]["decision"] == "NOT_CONFIRMED"


def test_summary_rejects_missing_condition():
    protocol = {"material_improvement_margin": 0.01}
    streams = [stream(i, 3.03, 3.01) for i in range(8)]
    streams[0]["records"].pop()
    with pytest.raises(ValueError, match="condition records"):
        summarize(protocol, streams)


def test_summary_rejects_incomplete_training_record():
    protocol = {"material_improvement_margin": 0.01}
    streams = [stream(i, 3.03, 3.01) for i in range(8)]
    streams[0]["records"][0]["training_loss"].pop()
    with pytest.raises(ValueError, match="condition record"):
        summarize(protocol, streams)
