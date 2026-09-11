from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_adamw8bit_error_compensation_training.py"


def module():
    spec = importlib.util.spec_from_file_location("compensation_training", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def test_start_blocks_are_reproducible_and_separated():
    value = module()
    first = value.selected_start_blocks()
    assert first == value.selected_start_blocks()
    assert len(first) == value.STREAMS
    assert all(value.START_RANGE[0] <= item < value.START_RANGE[1] for item in first)
    assert all(abs(left - right) >= value.MINIMUM_SEPARATION
               for index, left in enumerate(first) for right in first[index + 1:])


def test_summary_uses_only_preregistered_primary_contrast():
    value = module()
    protocol = {"material_improvement_margin": 0.01,
                "primary_contrast": "DEFAULT_MINUS_COMPENSATED"}
    streams = []
    for index in range(value.STREAMS):
        endpoint = {
            "FP32_ADAMW": 2.0,
            "ADAMW8BIT_BLOCK256": 2.04 + index * 0.001,
            "ADAMW8BIT_COMPENSATED_BLOCK256": 2.01,
        }
        records = []
        for condition in value.CONDITIONS:
            records.append({
                "condition": condition,
                "evaluation_loss_by_step": {str(value.STEPS): [endpoint[condition]]},
                "training_steps_per_second_with_evaluation_overhead": 1.0,
                "peak_allocated_bytes": 1,
            })
        streams.append({"stream_index": index, "status": "COMPLETE", "records": records})
    result = value.summarize(protocol, streams)
    assert result["primary"]["decision"] == "MATERIAL_IMPROVEMENT"
    assert result["primary"]["mean"] > 0.01
