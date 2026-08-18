import torch

from scripts.fx_replay import (
    ReferenceCutReplayInterpreter,
    prepare_reference_cut_tasks,
)


def test_reference_cut_value_sink_receives_exact_runtime_endpoint() -> None:
    module = torch.fx.symbolic_trace(lambda x: torch.neg(x))
    task = {
        "task_id": "neg-endpoint",
        "cut_id": "neg-endpoint",
        "phase": "FORWARD",
        "graph_index": 0,
        "expected_graph_code_sha256": __import__("hashlib").sha256(
            module.code.encode()
        ).hexdigest(),
        "aot_node_ids": ["forward:graph0:neg"],
        "aot_node_names": ["neg"],
        "expected_boundary_inputs": [{
            "source_node": "x",
            "consumer_node_id": "forward:graph0:neg",
            "consumer_argument_path": ["args", 0],
        }],
        "expected_boundary_outputs": [{
            "source_node_id": "forward:graph0:neg",
            "consumer_node_id": "forward:graph0:output",
            "consumer_argument_path": ["args", 0],
        }],
        "required_extractor_schema": "forkcert.fx-reference-cut.v2",
    }
    extracted, certificates = prepare_reference_cut_tasks(
        graph_module=module, phase="FORWARD", graph_index=0, tasks=[task]
    )
    observed = []
    interpreter = ReferenceCutReplayInterpreter(
        module, extracted,
        value_sink=lambda replay_task, values: observed.append(
            (replay_task.task_id, values[0].clone())
        ),
    )
    value = torch.tensor([1.0, -2.0])
    actual = interpreter.run(value)
    assert torch.equal(actual, -value)
    assert certificates[0]["graph_binding"]["port_routes_exact"]
    assert observed[0][0] == "neg-endpoint"
    assert torch.equal(observed[0][1], -value)
    assert interpreter.observations[0]["bitwise_equal"]


def test_reference_cut_capture_only_mode_skips_replay_but_keeps_endpoint() -> None:
    module = torch.fx.symbolic_trace(lambda x: torch.neg(x))
    task = {
        "task_id": "neg-endpoint",
        "cut_id": "neg-endpoint",
        "phase": "FORWARD",
        "graph_index": 0,
        "expected_graph_code_sha256": __import__("hashlib").sha256(
            module.code.encode()
        ).hexdigest(),
        "aot_node_ids": ["forward:graph0:neg"],
        "aot_node_names": ["neg"],
        "expected_boundary_inputs": [{
            "source_node": "x",
            "consumer_node_id": "forward:graph0:neg",
            "consumer_argument_path": ["args", 0],
        }],
        "expected_boundary_outputs": [{
            "source_node_id": "forward:graph0:neg",
            "consumer_node_id": "forward:graph0:output",
            "consumer_argument_path": ["args", 0],
        }],
        "required_extractor_schema": "forkcert.fx-reference-cut.v2",
    }
    extracted, _ = prepare_reference_cut_tasks(
        graph_module=module, phase="FORWARD", graph_index=0, tasks=[task]
    )
    observed = []
    interpreter = ReferenceCutReplayInterpreter(
        module,
        extracted,
        verify_replay=False,
        value_sink=lambda replay_task, values: observed.append(
            (replay_task.task_id, values[0].clone())
        ),
    )
    value = torch.tensor([1.0, -2.0])
    actual = interpreter.run(value)
    assert torch.equal(actual, -value)
    assert observed[0][0] == "neg-endpoint"
    assert torch.equal(observed[0][1], -value)
    assert interpreter.observations == []
    assert not interpreter.pending
