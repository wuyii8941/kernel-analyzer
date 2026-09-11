from scripts.run_adamw8bit_trajectory_response_audit import CONDITIONS, CHECKPOINTS


def test_trajectory_audit_has_fixed_conditions_and_checkpoints():
    assert CONDITIONS == ("FP32_ADAMW", "ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
    assert CHECKPOINTS == (0, 256, 512, 768, 1024)
