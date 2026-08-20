from scripts.build_tcmp_capability_map import ORBIT_BY_PROOF_KIND


def test_tcmp_orbits_are_declared_only_for_mathematical_semantic_units() -> None:
    assert ORBIT_BY_PROOF_KIND["TWO_DIMENSIONAL_LINEAR_MM_ADJOINT"] == (
        "JOINT_CONTRACTION_AXIS_PERMUTATION"
    )
    assert "STANDARD_DECOMPOSED_FP32_SILU_ADJOINT" not in ORBIT_BY_PROOF_KIND
    assert "RSQRT_DIRECT_SAVED_OUTPUT_ADJOINT" not in ORBIT_BY_PROOF_KIND
