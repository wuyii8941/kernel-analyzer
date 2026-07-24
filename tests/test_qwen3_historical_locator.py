from theory_oracle.qwen3_blind_locator_v0_1 import first_trace_mismatch


def test_trace_alignment_reports_first_generic_operation_mismatch():
    reference = [
        {"index": 0, "op": "mm", "outputs": [{"sha256": "a", "shape": [2]}]},
        {"index": 1, "op": "view", "outputs": [{"sha256": "b", "shape": [2]}]},
    ]
    candidate = [
        {"index": 0, "op": "mm", "outputs": [{"sha256": "a", "shape": [2]}]},
        {"index": 1, "op": "transpose", "outputs": [{"sha256": "c", "shape": [2]}]},
    ]
    result = first_trace_mismatch(reference, candidate)
    assert result == {
        "index": 1,
        "reference": "view",
        "candidate": "transpose",
        "kind": "operation_sequence",
    }


def test_trace_alignment_does_not_infer_a_root_cause_when_equal():
    trace = [{"index": 0, "op": "view", "outputs": [{"sha256": "a", "shape": [2]}]}]
    assert first_trace_mismatch(trace, trace) is None
