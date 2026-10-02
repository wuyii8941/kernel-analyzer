import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rotary_contract_records_saved_source_and_remaining_classes():
    path = ROOT / "results/property/root_cause_closure_v1/rotary_source_contract_v1.json"
    data = json.loads(path.read_text())
    assert data["status"] == "COMPLETE_SAVED_SOURCE_AUDIT"
    assert data["input_contract"] == {
        "query_storage": "bf16",
        "frequency_storage": "fp32",
        "position_storage": "int64",
        "intermediate_arithmetic": "fp32 after input loads",
        "output_storage": "bf16",
    }
    assert data["required_operations"] == {
        "tl_math.cos": True,
        "tl_math.sin": True,
        "tl_math.log": True,
        "libdevice.floor": True,
    }
    assert data["excluded_on_tested_device"] == [
        "tl_math versus libdevice trigonometric entry point"
    ]
    assert set(data["source_candidates_left_after_probe"]) == {
        "fused FP32 operation ordering",
        "where BF16 materialization occurs relative to the fused expression",
        "real model operand and position distributions",
    }
