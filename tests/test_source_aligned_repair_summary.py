from scripts.build_source_aligned_repair_summary import DECLARED


def test_previously_failed_repairs_bind_source_aligned_full_arms() -> None:
    rows = {row["case_id"]: row for row in DECLARED}
    assert rows["qwen128_vproj_mm"]["full_arm"] == "ROUNDING_ONLY"
    assert rows["qwen64_vproj_mm"]["full_arm"] == "JOINT"
    assert rows["mamba_seq64_input_proj"]["full_arm"] == "JOINT"


def test_historical_accumulation_arm_is_not_relabeled() -> None:
    assert {row["old_arm"] for row in DECLARED} == {"KERNEL_ONLY"}
