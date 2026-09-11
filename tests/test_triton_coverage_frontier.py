from scripts.build_triton_coverage_frontier import build


def test_frontier_deduplicates_release_copies_and_does_not_use_outcomes():
    catalog = {
        "positions": [
            {
                "release": "/r-copy", "canonical_release": "/r",
                "release_task_package_sha256": "pkg", "task_id": "t",
                "formal_pointer": "out", "phase": "BACKWARD",
                "operator_family": "SOFTMAX", "implementation_kind": "TRITON",
                "support_status": "READY_FOR_MEASUREMENT",
                "classification_confidence": "AUDITED",
                "available_reference_bindings": [{"family": "SOFTMAX_BACKWARD"}],
            },
            {
                "release": "/r", "canonical_release": "/r",
                "release_task_package_sha256": "pkg", "task_id": "t",
                "formal_pointer": "out", "phase": "BACKWARD",
                "operator_family": "SOFTMAX", "implementation_kind": "TRITON",
                "support_status": "READY_FOR_MEASUREMENT",
                "is_canonical_release_package": True,
                "classification_confidence": "AUDITED",
                "available_reference_bindings": [{"family": "SOFTMAX_BACKWARD"}],
            },
            {
                "release": "/r", "canonical_release": "/r",
                "release_task_package_sha256": "pkg2", "task_id": "t2",
                "formal_pointer": "out", "phase": "FORWARD",
                "operator_family": "SOFTMAX", "implementation_kind": "TRITON",
                "support_status": "VALID_MEASUREMENT_COMPLETED",
                "classification_confidence": "EXACT_ENDPOINT",
            },
        ]
    }
    result = build(catalog, input_sha256="digest")
    assert result["position_count_after_deduplication"] == 2
    assert result["triton_position_count"] == 2
    assert result["triton_valid_measurement_count"] == 1
    assert result["triton_ready_for_measurement_count"] == 1
    family = result["families"][0]
    assert family["next_action"] == "RUN_DECLARED_MEASUREMENT"
    assert family["next_representative"]["task_id"] == "t"
    assert result["selection_uses_numerical_outcomes"] is False
