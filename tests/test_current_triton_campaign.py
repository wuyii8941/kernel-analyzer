from scripts.build_current_qwen_triton_campaign import current_exact_adapter, pointer_abi
from scripts.audit_qwen_executed_release_identity import exact_flow_checks


def test_pointer_abi_retains_write_only_inout_as_prelaunch_input() -> None:
    row = {
        "boundary_witness": {
            "formal_to_actual_pointer_binding": {
                "in_out_ptr0": {"loaded": False, "stored": True},
                "in_ptr0": {"loaded": True, "stored": False},
                "out_ptr0": {"loaded": False, "stored": True},
            }
        }
    }

    inputs, outputs = pointer_abi(row)

    assert inputs == ("in_out_ptr0", "in_ptr0")
    assert outputs == ("in_out_ptr0", "out_ptr0")


def test_pointer_abi_does_not_promote_plain_write_only_output_to_input() -> None:
    row = {
        "boundary_witness": {
            "formal_to_actual_pointer_binding": {
                "out_ptr0": {"loaded": False, "stored": True},
            }
        }
    }

    inputs, outputs = pointer_abi(row)

    assert inputs == ()
    assert outputs == ("out_ptr0",)


def test_prepare_softmax_requires_exact_generated_symbol() -> None:
    key = (
        "FORWARD",
        (
            "aten._unsafe_view", "aten._to_copy", "aten.view",
            "prims.prepare_softmax_online",
        ),
        ("in_ptr0",),
        ("out_ptr0",),
    )

    symbol = "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_16"
    assert current_exact_adapter(key, symbol) == symbol
    assert current_exact_adapter(key, symbol + "_near_match") is None


def test_embedding_atomic_vjp_requires_exact_semantics_and_abi() -> None:
    symbol = (
        "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_"
        "mul_nll_loss_forward_pow_sum_view_20"
    )
    key = (
        "BACKWARD",
        (
            "aten.nll_loss_forward", "aten.view", "aten.add", "aten.mul",
            "aten._to_copy", "aten.sum", "aten.pow", "aten.expand",
            "aten.div", "aten.embedding_dense_backward",
        ),
        (
            "in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4",
            "in_ptr5", "in_ptr6", "in_ptr7", "out_ptr2",
        ),
        ("out_ptr2",),
    )

    assert current_exact_adapter(key, symbol) == (
        "forkcert:embedding-rmsnorm-scatter-vjp"
    )
    wrong_inputs = (*key[2][:-1],)
    assert current_exact_adapter((key[0], key[1], wrong_inputs, key[3]), symbol) is None


def test_direct_aten_identity_includes_accumulation_semantics() -> None:
    row = {
        "phase": "BACKWARD",
        "kind": "DIRECT_ATEN",
        "symbol": "index_put_",
        "call_expression": "aten.index_put_(buf, [indices], values, True)",
        "source_line_sha256": "line-hash",
        "boundary_witness": {
            "boundary_source": "DIRECT_ATEN_INPLACE_SCHEMA_AND_CALL_AST",
            "mutated_target": "buf",
            "accumulate_expression": "True",
        },
    }
    assert all(exact_flow_checks(row, row).values())
    near_match = {**row, "boundary_witness": {**row["boundary_witness"], "accumulate_expression": "False"}}
    checks = exact_flow_checks(row, near_match)
    assert checks["accumulate_expression"] is False


def test_external_identity_includes_callsite_and_tensor_abi() -> None:
    row = {
        "phase": "FORWARD",
        "kind": "EXTERN",
        "symbol": "mm",
        "call_expression": "extern_kernels.mm(a, b, out=buf)",
        "source_line_sha256": "line-hash",
        "input_tensor_variables": ["a", "b"],
        "output_tensor_variables": ["buf"],
        "input_storage_root_variables": ["a", "b"],
        "output_storage_root_variables": ["buf"],
        "boundary_witness": {"boundary_source": "EXPLICIT_EXTERNAL_OUT_KEYWORD_AST"},
    }
    assert all(exact_flow_checks(row, row).values())
    near_match = {**row, "input_tensor_variables": ["b", "a"]}
    assert exact_flow_checks(row, near_match)["input_tensor_variables"] is False


def test_seq256_split_rms_final_requires_exact_generated_program_and_abi() -> None:
    key = (
        "BACKWARD",
        ("aten.view", "aten.add", "aten._to_copy", "aten.mul", "aten.sum"),
        ("in_ptr0",),
        ("out_ptr0",),
    )
    symbol = "triton_per_fused__to_copy_mul_sum_view_4"
    assert current_exact_adapter(key, symbol, 256) == "forkcert:rms-weight-split-final"
    assert current_exact_adapter(key, symbol + "_near_match", 256) is None


def test_seq256_split_rms_partial_rejects_seq64_adapter_alias() -> None:
    key = (
        "BACKWARD",
        ("aten.view", "aten.add", "aten._to_copy", "aten.mul", "aten.sum"),
        ("in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3"),
        ("out_ptr0",),
    )
    symbol = "triton_red_fused__to_copy_add_mul_sum_view_7"
    assert current_exact_adapter(key, symbol, 256) == "forkcert:rms-weight-split-partial-two"
    wrong_abi = (key[0], key[1], key[2][:-1], key[3])
    assert current_exact_adapter(wrong_abi, symbol, 256) is None
    assert current_exact_adapter(key, symbol, 128) is None


def test_seq256_loss_split_requires_exact_semantics_and_pointer_abi() -> None:
    key = (
        "FORWARD",
        (
            "aten._unsafe_view", "aten._to_copy", "aten.view",
            "prims.prepare_softmax_online", "aten._log_softmax",
        ),
        ("in_out_ptr0", "in_ptr0"),
        ("in_out_ptr0", "out_ptr0"),
    )
    symbol = (
        "triton_red_fused__log_softmax__to_copy__unsafe_view_"
        "prepare_softmax_online_view_16"
    )
    assert current_exact_adapter(key, symbol, 256) == "forkcert:loss-softmax-seq256-partial"
    assert current_exact_adapter((key[0], key[1], ("in_ptr0",), key[3]), symbol, 256) is None
