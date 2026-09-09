import pytest

from scripts.freeze_family_optimizer_followup import select


def test_followup_selection_uses_frozen_direction_and_magnitude_rule():
    plan={"cases":[{"case_id":"a"},{"case_id":"b"},{"case_id":"c"}]}
    summary={"records":[
        {"case_id":"a","update_rms_relative":.03,"repair_aligned_direction":"NEGATIVE"},
        {"case_id":"b","update_rms_relative":.05,"repair_aligned_direction":"NEGATIVE"},
        {"case_id":"c","update_rms_relative":.2,
         "repair_aligned_direction":"NOT_REPRODUCED_ACROSS_ALL_RECORDED_VIEWS"},
    ]}
    result=select(plan,summary,minimum_update_rms=.02,maximum_cases=1,
                  required_direction="repair_aligned_direction")
    assert [case["case_id"] for case in result["cases"]]==["b"]
    assert result["data_use"].startswith("RESULT_AWARE")


def test_followup_selection_rejects_invalid_policy():
    with pytest.raises(ValueError):
        select({"cases":[]},{"records":[]},minimum_update_rms=0,maximum_cases=1,
               required_direction="repair_aligned_direction")
