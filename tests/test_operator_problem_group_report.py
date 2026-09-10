from scripts.build_operator_problem_group_report import _markdown


def test_markdown_keeps_positions_and_problem_groups_separate():
    report = {
        "valid_position_count": 12,
        "catalogue_family_count": 1,
        "catalogue": [{
            "label": "Family", "classified_positions": 20,
            "valid_measurement_positions": 12, "valid_triton_positions": 7,
            "valid_other_or_undeclared_positions": 5,
            "additional_measurement_artifacts": 1,
        }],
        "priority_problem_groups": [{
            "problem_group": "GROUP", "actual_implementation": "TRITON",
            "chain_status": "OPEN",
        }],
    }
    text = _markdown(report)
    assert "位置数不是独立问题数" in text
    assert "12" in text and "GROUP" in text
