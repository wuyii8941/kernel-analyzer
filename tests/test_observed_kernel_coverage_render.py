from scripts.render_observed_kernel_coverage import render


def test_render_keeps_support_stages_separate():
    summary = {
        "release_count": 2, "distinct_release_task_package_count": 1,
        "duplicate_release_directory_count": 1, "position_count": 4,
        "canonical_position_count": 2, "kernel_invocation_count": 3,
        "distinct_release_qualified_position_count": 2,
        "position_support_status_counts": {"VALID_MEASUREMENT_COMPLETED": 1, "READY_FOR_MEASUREMENT": 1},
        "position_family_counts": {"LINEAR": 4},
        "distinct_position_family_counts": {"LINEAR": 4},
        "distinct_position_support_status_counts": {"VALID_MEASUREMENT_COMPLETED": 1, "READY_FOR_MEASUREMENT": 1},
        "family_support_status_counts": {"LINEAR": {
            "IDENTIFIED": 1, "REFERENCE_AVAILABLE": 1,
            "READY_FOR_MEASUREMENT": 1, "VALID_MEASUREMENT_COMPLETED": 1,
        }},
        "distinct_family_support_status_counts": {"LINEAR": {
            "IDENTIFIED": 1, "REFERENCE_AVAILABLE": 1,
            "READY_FOR_MEASUREMENT": 1, "VALID_MEASUREMENT_COMPLETED": 1,
        }},
        "family_implementation_kind_counts": {"LINEAR": {"TRITON": 3, "EXTERN": 1}},
        "distinct_family_implementation_kind_counts": {"LINEAR": {"TRITON": 3, "EXTERN": 1}},
    }
    text = render(summary)
    assert "| 矩阵乘法与线性层 (`LINEAR`) | 4 | 3 | 1 | 1 | 1 | 1 | 1 |" in text
    assert "不表示发现 bias" in text
