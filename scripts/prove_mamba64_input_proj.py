#!/usr/bin/env python3
"""Close the concrete F+B proof for Mamba seq64 layer-0 input projection."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CID = "mamba_seq64_forward_1_output"
SEQ = 68994
SOURCE_LINE = 3711


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def output_leaves(node: dict) -> set[str]:
    leaves: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict) and set(value) == {"node"}:
            leaves.add(str(value["node"]))
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node["arguments"])
    return leaves


def shortest_path(nodes: dict[str, dict], start: str, outputs: set[str]) -> list[str] | None:
    queue = collections.deque([(start, [start])])
    seen = {start}
    while queue:
        name, path = queue.popleft()
        if name in outputs:
            return path
        for user in nodes[name]["users"]:
            if user != "output" and user not in seen:
                seen.add(user)
                queue.append((user, path + [user]))
    return None


def main() -> None:
    aot_path = COVERAGE / "mamba_seq64_aot.json.gz"
    with gzip.open(aot_path, "rt") as handle:
        aot = json.load(handle)
    assert aot["status"] == "COMPLETE_AOT_FB_CAPTURE"
    assert all(aot["observation_stability"].values())
    release = COVERAGE / "runtime_releases/mamba_seq64_r1"
    release_capture = json.loads((release / "capture.json").read_text())
    queue = json.loads((COVERAGE / "bias_candidate_queue.json").read_text())
    candidate = next(row for row in queue["candidates"] if row["candidate_id"] == CID)
    live = json.loads((COVERAGE / "live_contrasts/mamba_seq64.json").read_text())
    precision = next(row for row in live["results"] if row["candidate_id"] == CID
                     and row["contrast_axis"] == "PRECISION")
    optimization = next(row for row in live["results"] if row["candidate_id"] == CID
                        and row["contrast_axis"] == "OPTIMIZATION")
    decomposition_path = COVERAGE / "cases/mamba_seq64_input_proj_precision_decomposition.json"
    decomposition = json.loads(decomposition_path.read_text())
    assert decomposition["candidate_id"] == CID and all(decomposition["gates"].values())
    assert decomposition["coherent_sources"] == ["kernel", "output_rounding"]

    source_path = Path(release_capture["modules"][0]["captured_source"])
    source = source_path.read_text()
    source_call = source.splitlines()[SOURCE_LINE - 1].strip()
    source_comment = source.splitlines()[SOURCE_LINE - 2].strip()
    exact_call = candidate["exact_generated_call"]
    assert hashlib.sha256(source.encode()).hexdigest() == release_capture["modules"][0]["sha256"]
    assert hashlib.sha256(source_call.encode()).hexdigest() == exact_call["source_line_sha256"]
    assert source_call == exact_call["call_expression"]
    assert "aten.mm" in source_comment and "out=buf4" in source_call
    assert "buf3, (64, 768)" in source_call and "primals_4, (768, 3072)" in source_call

    graphs = aot["capture"]["graphs"]
    forward = next(graph for graph in graphs if graph["phase"] == "FORWARD")
    backward = next(graph for graph in graphs if graph["phase"] == "BACKWARD")
    fw = {node["name"]: node for node in forward["nodes"]}
    bw = {node["name"]: node for node in backward["nodes"]}
    root = fw["mm"]
    assert root["target"] == "aten.mm.default" and root["seq_nr"] == SEQ
    assert root["arguments"]["args"] == [{"node": "view"}, {"node": "t"}]
    assert root["tensor_meta"][0] == [64, 3072]
    assert any("layers.0.mixer.in_proj" in value[0]
               for value in root["nn_module_stack"].values())
    assert fw["view"]["tensor_meta"][0] == [64, 768]
    assert fw["t"]["input_nodes"] == ["primals_4"]
    assert fw["t"]["tensor_meta"][0] == [768, 3072]
    assert {"view", "t"}.issubset(set(fw["output"]["input_nodes"]))
    assert bw["view"]["op"] == "placeholder" and bw["t"]["op"] == "placeholder"

    cotangent = "view_9556"
    assert bw[cotangent]["tensor_meta"][0] == [64, 3072]
    ancestors: set[str] = set()
    stack = [cotangent]
    while stack:
        name = stack.pop()
        if name in ancestors:
            continue
        ancestors.add(name)
        stack.extend(bw[name]["input_nodes"])
    assert "tangents_1" in ancestors
    assert bw["mm_289"]["arguments"]["args"] == [
        {"node": "t_529"}, {"node": "view"}
    ]
    assert bw["mm_289"]["tensor_meta"][0] == [3072, 768]
    assert bw["mm_290"]["arguments"]["args"] == [
        {"node": cotangent}, {"node": "t_531"}
    ]
    assert bw["mm_290"]["tensor_meta"][0] == [64, 768]
    assert all(bw[name]["seq_nr"] == SEQ for name in
               ("t_529", "mm_289", "t_530", "t_531", "mm_290"))
    outputs = output_leaves(bw["output"])
    dw_path = shortest_path(bw, "mm_289", outputs)
    dx_path = shortest_path(bw, "mm_290", outputs)
    assert dw_path == ["mm_289", "t_530", "t_532"]
    assert dx_path == ["mm_290", "view_9557", "mul_5235", "sum_217", "view_9558"]

    derivation = {
        "symbols": {"X": "[64,768]", "W": "[3072,768]", "Q=dL/dY": "[64,3072]"},
        "forward": "Y = X W^T",
        "backward": {"dX_contribution": "Q W", "dW": "Q^T X"},
        "actual_aot_program": {
            "forward": "mm(view(X), t(primals_4=W))",
            "dW": "t_532(t(t(mm_289(t_529(Q^T), view(X))))) = Q^T X",
            "dX_contribution": "mm_290(view_9556(Q), t_531(W)) = Q W",
            "dW_output_path": dw_path,
            "dX_output_accumulation_path": dx_path,
        },
        "candidate_generated_call": source_call,
    }
    concrete = {
        "saved_tensor_origins_exact": True,
        "cotangent_edge_exact": True,
        "backward_program_matches_analytic_vjp": True,
        "non_tensor_arguments_exact": True,
        "output_edges_exact": True,
        "forward_program_sha256": forward["code_sha256"],
        "backward_program_sha256": backward["code_sha256"],
        "analytic_derivation_sha256": canonical(derivation),
    }
    payload = {
        "schema": "kernel-analyzer-concrete-fb-bias-case-v1",
        "status": "T1_COHERENT_MIXED_PRECISION_MECHANISMS_WITH_CONCRETE_FB_PROOF",
        "candidate_id": CID,
        "subject": "Mamba-130M seq64 layer-0 mixer input-projection forward MM",
        "cause_axis": "PRECISION",
        "forward_backward_unit": derivation,
        "concrete_program_proof": concrete,
        "bindings": {
            "aot_capture_sha256": aot["result_sha256"],
            "aot_file_sha256": hashlib.sha256(aot_path.read_bytes()).hexdigest(),
            "release_capture_sha256": release_capture["result_sha256"],
            "generated_source_sha256": release_capture["modules"][0]["sha256"],
            "generated_source_line": SOURCE_LINE,
            "generated_source_line_sha256": exact_call["source_line_sha256"],
            "forward_root": "mm", "sequence_nr": SEQ,
            "module": "backbone.layers.0.mixer.in_proj",
            "saved_tensor_nodes": ["view", "t"], "cotangent_node": cotangent,
            "weight_gradient_output": "t_532", "input_gradient_contribution": "mm_290",
        },
        "numerical": {
            "states": live["state_count"], "coordinates": precision["coordinates"],
            "max_abs": precision["max_abs"],
            "u_statistic": precision["direction"]["cross_state_inner_product_u"],
            "cluster_bootstrap_95": precision["direction"]["cluster_bootstrap_95"],
            "repeat_exact": precision["repeat_exact"],
            "optimization_max_abs": optimization["max_abs"],
            "live_result_sha256": live["result_sha256"],
        },
        "precision_mechanism": {
            "coherent_sources": decomposition["coherent_sources"],
            "kernel": decomposition["direction"]["kernel"],
            "output_rounding": decomposition["direction"]["output_rounding"],
            "decomposition_result_sha256": decomposition["result_sha256"],
        },
        "claim_boundary": (
            "Complete concrete local F+B proof, T1 coherent precision bias, and exact local "
            "mechanism decomposition. Both local MM kernel difference and deterministic output "
            "rounding are coherent. Causal intervention, downstream carrier, and T4 remain open."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    output = COVERAGE / "cases/mamba_seq64_input_proj.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": payload["status"]}))


if __name__ == "__main__":
    main()
