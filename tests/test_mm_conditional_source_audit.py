from scripts.audit_mm_conditional_sources import review


def document(mode, errors):
    return {"candidate_id": "test", "carrier_parameter": "weight",
            "states": [{"state_id": "s", "arms": {mode: {
                "repeat_local_summaries": [{"kernel_residual_preservation_error": e} for e in errors]
            }}}]}


def test_rounding_requires_every_record_zero():
    zero = {"l2": 0, "max_abs": 0, "nonzero": 0}
    small = {"l2": 1e-12, "max_abs": 1e-12, "nonzero": 1}
    check = review(document("ROUNDING_ONLY", [zero, small]))["checks"]["ROUNDING_ONLY"]
    assert check["interpretation"] == "SOURCE_ISOLATION_NOT_VERIFIED"
    assert review(document("ROUNDING_ONLY", [zero]))["checks"]["ROUNDING_ONLY"]["all_recorded_preservation_errors_zero"]


def test_joint_does_not_require_preservation():
    check = review(document("JOINT", [{"l2": 2, "max_abs": 1, "nonzero": 3}]))["checks"]["JOINT"]
    assert check["interpretation"] == "PRESERVATION_NOT_REQUIRED_JOINT_REMOVES_KERNEL_RESIDUAL"


def test_missing_measurement_is_not_zero():
    assert not review(document("ROUNDING_ONLY", [None]))["checks"]["ROUNDING_ONLY"]["all_recorded_preservation_errors_zero"]
