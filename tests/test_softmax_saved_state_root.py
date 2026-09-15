import json
from pathlib import Path

from scripts.analyze_softmax_saved_state_root import build_payload


ROOT = Path(__file__).resolve().parents[1]


def test_saved_softmax_root_result_recomputes_from_same_call_records():
    source = ROOT / "results/property/numerical_coverage_v1/qwen128_softmax_mechanism_v1"
    stored = json.loads((ROOT / "results/property/root_cause_closure_v1/softmax_saved_state.json").read_text())
    recomputed = build_payload(source)
    assert recomputed == stored
    assert stored["calls_with_nonzero_defect"] == stored["call_count"] == 56
    assert stored["nonzero_row_count"] == stored["row_count"] == 114688
    assert stored["maximum_defect_after_consistent_renormalization"] < 1e-14
