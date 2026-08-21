from kernel_analyzer.implementation_identity import (
    build_implementation_identity,
    novelty_label,
)


def contract(shape, stride):
    return {
        "shape": shape, "stride": stride, "dtype": "torch.bfloat16",
        "device_type": "cuda", "layout": "torch.strided", "storage_offset": 0,
    }


def identity(shape=(8, 16), program="a", operation="triton_red_fused_sum_0"):
    return build_implementation_identity(
        backend="inductor-triton", implementation_kind="TRITON", phase="BACKWARD",
        operation=operation, program_digest=program,
        semantic_operations=["aten.sum"], fusion_boundary=["sum"],
        operand_contracts={"in_ptr0": contract(list(shape), [shape[1], 1])},
    )


def test_exact_identity_is_value_blind_and_shape_specific():
    first = identity()
    assert first == identity()
    assert first["candidate_tensor_values_used"] is False
    assert first["exact_implementation_id"] != identity((4, 16))["exact_implementation_id"]


def test_symbol_ordinals_do_not_create_new_pattern():
    first = identity(operation="triton_red_fused_sum_0")
    second = identity(program="b", operation="triton_red_fused_sum_19")
    assert first["exact_implementation_id"] != second["exact_implementation_id"]
    assert first["implementation_pattern_id"] == second["implementation_pattern_id"]


def test_novelty_levels_are_distinct():
    development = [identity()]
    assert novelty_label(identity(), development) == "SEEN_EXACT_IMPL_NEW_OPERANDS"
    assert novelty_label(identity(program="b"), development) == "NEW_EXACT_IMPL_SEEN_PATTERN"
    new_pattern = build_implementation_identity(
        backend="extern-cublas", implementation_kind="EXTERN", phase="BACKWARD",
        operation="sum", semantic_operations=["aten.sum"],
        operand_contracts={"arg0": contract([8, 16], [16, 1])},
    )
    assert novelty_label(new_pattern, development) == "NEW_IMPL_PATTERN"
    new_family = build_implementation_identity(
        backend="custom", implementation_kind="EXTERN", phase="FORWARD",
        operation="topk", semantic_operations=["aten.topk"],
        operand_contracts={"arg0": contract([8, 16], [16, 1])},
    )
    assert novelty_label(new_family, development) == "NEW_SEMANTIC_FAMILY"
