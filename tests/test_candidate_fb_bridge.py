from copy import deepcopy
import gzip
import json
from pathlib import Path

from scripts.build_candidate_fb_bridge import (
    executable_aot_graph_projection,
    match_segmented_executable_graphs,
    owner_index,
)


ROOT = Path(__file__).resolve().parents[1]


def test_executable_graph_identity_ignores_only_observation_provenance() -> None:
    graph = {
        "phase": "FORWARD",
        "graph_index": 0,
        "code_sha256": "code",
        "input_count": 1,
        "node_count": 1,
        "call_function_count": 1,
        "nodes": [{
            "phase": "FORWARD",
            "ordinal": 0,
            "name": "mul",
            "op": "call_function",
            "target": "aten.mul.Tensor",
            "arguments": ["x", "y"],
            "input_nodes": ["x", "y"],
            "input_edges": [],
            "users": ["output"],
            "tensor_meta": {"dtype": "torch.bfloat16", "shape": [2]},
            "seq_nr": 7,
            "stack_trace": "first wrapper",
            "from_node": ["first provenance"],
        }],
    }
    recompiled = deepcopy(graph)
    recompiled["nodes"][0].update({
        "seq_nr": 9001,
        "stack_trace": "proof-tagged wrapper",
        "from_node": ["proof-tagged provenance"],
    })
    assert executable_aot_graph_projection([graph]) == (
        executable_aot_graph_projection([recompiled])
    )

    recompiled["nodes"][0]["arguments"] = ["y", "x"]
    assert executable_aot_graph_projection([graph]) != (
        executable_aot_graph_projection([recompiled])
    )


def test_partition_auxiliary_and_backward_replay_nodes_receive_proof_owners() -> None:
    proof = {"proof_kind": "EXACT_REPLAY", "passed": True}
    math = {
        "units": [],
        "auxiliary_backward_program": [],
        "partition_auxiliary_forward_program": [{
            "node_id": "forward:graph0:saved_view",
            "status": "PROVED_PARTITION_AUXILIARY_FORWARD_PROGRAM",
            "composite_program_proof": proof,
        }],
        "backward_only_partition_replay_program": [{
            "node_id": "backward:graph0:replayed_view",
            "status": "PROVED_BACKWARD_PARTITION_REMATERIALIZATION",
            "composite_program_proof": proof,
        }],
    }
    owners = owner_index(math)
    assert set(owners) == {
        "forward:graph0:saved_view",
        "backward:graph0:replayed_view",
    }
    assert owners["forward:graph0:saved_view"]["owner_kind"] == (
        "PROVED_PARTITION_AUXILIARY_FORWARD_PROGRAM"
    )
    assert owners["backward:graph0:replayed_view"]["owner_kind"] == (
        "PROVED_BACKWARD_PARTITION_REPLAY_PROGRAM"
    )


def test_segmented_graph_match_allows_only_exact_device_put_insertion() -> None:
    def graph(index: int, target: str, code: str) -> dict:
        return {
            "phase": "FORWARD", "graph_index": index, "code_sha256": code,
            "input_count": 1, "node_count": 1, "call_function_count": 1,
            "nodes": [{
                "phase": "FORWARD", "ordinal": 0, "name": target.split(".")[1],
                "op": "call_function", "target": target, "arguments": ["x"],
                "input_nodes": ["x"], "input_edges": [], "users": ["output"],
                "tensor_meta": {"dtype": "torch.float32", "shape": []},
            }],
        }

    source = [graph(0, "aten.sin.default", "sin"), graph(1, "aten.cos.default", "cos")]
    proof = [
        graph(0, "aten.sin.default", "sin"),
        graph(1, "prims.device_put.default", "device"),
        graph(2, "aten.cos.default", "cos"),
    ]
    mapping, extras = match_segmented_executable_graphs(source, proof)
    assert mapping == {
        ("FORWARD", 0): ("FORWARD", 0),
        ("FORWARD", 2): ("FORWARD", 1),
    }
    assert len(extras) == 1
    assert extras[0]["theorem"]["proof_kind"] == (
        "EXACT_COMPILER_ADDED_DEVICE_TRANSFER"
    )

    proof[1] = graph(1, "aten.add.Tensor", "add")
    try:
        match_segmented_executable_graphs(source, proof)
    except RuntimeError as error:
        assert "non-DevicePut" in str(error)
    else:
        raise AssertionError("non-DevicePut graph insertion must fail closed")


def test_mamba64_actual_candidate_denominator_is_fully_bound() -> None:
    release = ROOT / "results/coverage/runtime_releases/mamba_seq64_r1"
    with gzip.open(release / "default_aot_math.json.gz", "rt") as handle:
        math = json.load(handle)
    with gzip.open(release / "candidate_fb_bridge.json.gz", "rt") as handle:
        bridge = json.load(handle)
    with gzip.open(release / "same_dtype_tasks.json.gz", "rt") as handle:
        tasks = json.load(handle)

    assert math["status"] == "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION"
    assert math["denominator"]["semantic_forward_backward_units"] == 14824
    assert math["denominator"]["units_pending_composite_vjp_proof"] == 0
    assert all(math["gates"].values())

    assert bridge["status"] == (
        "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_PROVED_FB_MATHEMATICS"
    )
    assert bridge["denominator"]["candidate_compute_regions"] == 8513
    assert bridge["denominator"]["bound_to_proved_fb_mathematics"] == 8513
    assert bridge["denominator"]["unresolved"] == 0
    assert bridge["bindings"]["proof_tagged_aot_binding_mode"] == (
        "EXACT_EXECUTABLE_AOT_GRAPH"
    )

    assert tasks["status"] == (
        "COMPLETE_ALL_CANDIDATE_PORTS_ASSIGNED_TO_EXACT_SEMANTIC_ENDPOINTS"
    )
    assert tasks["denominator"]["candidate_compute_regions"] == 8513
    assert tasks["denominator"]["stored_candidate_ports"] == 11755
    assert tasks["denominator"]["unresolved"] == 0


def test_phi64_segmented_aliases_reach_same_dtype_endpoints() -> None:
    release = ROOT / "results/coverage/runtime_releases/phi4_seq64_r1"
    with gzip.open(release / "candidate_fb_bridge.json.gz", "rt") as handle:
        bridge = json.load(handle)
    with gzip.open(release / "same_dtype_tasks.json.gz", "rt") as handle:
        tasks = json.load(handle)

    aliases = bridge["bindings"]["proof_id_aliases"]
    assert aliases
    assert any("forward_g4__" in value for value in aliases.values())
    assert bridge["gates"]["only_explicit_device_put_graphs_admitted"]
    assert tasks["status"] == (
        "COMPLETE_ALL_CANDIDATE_PORTS_ASSIGNED_TO_EXACT_SEMANTIC_ENDPOINTS"
    )
    assert tasks["denominator"]["exact_semantic_endpoints"] == 1203
    assert tasks["denominator"]["unresolved"] == 0
    segmented_cut = next(
        row for row in tasks["reference_cut_tasks"]
        if row.get("semantic_endpoint_id", "").startswith(
            "forward:graph0:forward_g0__"
        )
    )
    proof_node_id = segmented_cut["aot_node_ids"][0]
    assert "forward_g0__" not in proof_node_id
    assert all(
        edge["consumer_node_id"] == proof_node_id
        for edge in segmented_cut["expected_boundary_inputs"]
    )
    assert all(
        edge["source_node_id"] == proof_node_id
        for edge in segmented_cut["expected_boundary_outputs"]
    )
