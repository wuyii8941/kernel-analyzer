import pytest

from scripts.run_grouped_causal_softmax_capture import select


def fixture():
    contract = {"output_pointers": ["in_out_ptr0", "out_ptr0", "out_ptr1", "out_ptr2"]}
    case = {"task_id": "forward:1:out_ptr2", "case_id": "softmax-1",
            "expected_symbol": "softmax", "reference_output_pointer": "out_ptr2",
            "reference_method": "GROUPED_CAUSAL_SOFTMAX_FORWARD_COMMON_INPUT"}
    return {"schema": "grouped-causal-softmax-forward-bound-plan-v1",
            "cases": [case], "contracts": {"softmax": contract}}, case


def test_selects_exact_frozen_case_without_results():
    plan, case = fixture()
    assert select(plan, [case]) == plan["contracts"]
    plan["bias_result"] = 999
    assert select(plan, [case]) == plan["contracts"]


def test_changed_case_or_output_rejected():
    plan, case = fixture()
    with pytest.raises(ValueError, match="differs"):
        select(plan, [dict(case, reference_output_pointer="out_ptr1")])
