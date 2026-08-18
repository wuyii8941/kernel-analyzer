from scripts.aot_capture import AOTForwardBackwardCapture


def test_nondifferentiable_terminal_output_closes_forward_only_bridge() -> None:
    row = {
        "is_user_output": False,
        "forward_input_matches": [],
        "downstream_forward_input_matches": [],
        "backward_input_matches": [],
        "runtime_dtype": "torch.int64",
        "non_differentiable_terminal_boundary": True,
        "terminal_boundary_classification_uses_name_or_shape": False,
    }
    assert AOTForwardBackwardCapture._forward_output_resolved(row)


def test_unconsumed_differentiable_output_remains_unresolved() -> None:
    row = {
        "is_user_output": False,
        "forward_input_matches": [],
        "downstream_forward_input_matches": [],
        "backward_input_matches": [],
        "runtime_dtype": "torch.bfloat16",
        "non_differentiable_terminal_boundary": False,
        "terminal_boundary_classification_uses_name_or_shape": False,
    }
    assert not AOTForwardBackwardCapture._forward_output_resolved(row)
