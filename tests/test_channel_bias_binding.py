from scripts.bind_channel_bias_cases import bind


def fixtures():
    pairs = {"records": [{
        "closed_task_id": "forward:4:in_out_ptr0", "bias_name": "primals_6",
        "bias_symbol": "triton_bias", "exact_aot_endpoint_id": "forward:graph0:conv",
    }]}
    paths = {
        "parameter_binding_records": [{
            "primal": "primals_6", "name": "layer.conv.bias",
            "aliases": ["layer.conv.bias"], "status": "EXACT_MODULE_STACK_PARAMETER_BINDING",
        }],
        "rows": [{"endpoint": "forward:graph0:conv",
                  "parameters": [{"name": "layer.conv.bias"}]}],
    }
    tasks = {"rows": [{
        "task_id": "forward:4:in_out_ptr0", "implementation_kind": "TRITON",
        "formal_pointer": "in_out_ptr0", "symbol": "triton_bias",
        "status": "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT",
        "exact_aot_endpoint_id": "forward:graph0:conv",
    }]}
    return pairs, paths, tasks


def test_exact_bias_primal_is_selected_without_layer_rules():
    cases, unresolved = bind(*fixtures())
    assert unresolved == []
    assert cases[0]["carrier"] == "layer.conv.bias"
    assert cases[0]["task_id"] == "forward:4:in_out_ptr0"
    assert cases[0]["parameter_selection_rule"] == (
        "EXACT_BIAS_PRIMAL_TO_MODULE_PARAMETER")


def test_missing_parameter_reach_is_not_silently_replaced():
    pairs, paths, tasks = fixtures()
    paths["rows"][0]["parameters"] = []
    cases, unresolved = bind(pairs, paths, tasks)
    assert cases == []
    assert unresolved[0]["reasons"] == [
        "BIAS_PARAMETER_NOT_REACHABLE_FROM_ENDPOINT"]
