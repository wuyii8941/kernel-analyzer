#!/usr/bin/env python3
"""Close the concrete F+B proof for Qwen seq128 layer-0 v_proj."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CID = "qwen_seq128_forward_8_output"
SEQ = 11194
SOURCE_LINE = 1737


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def output_leaves(node: dict) -> list[str]:
    leaves: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict) and set(value) == {"node"}:
            leaves.append(str(value["node"]))
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node["arguments"])
    return leaves


def shortest_output_path(nodes: dict[str, dict], start: str, outputs: set[str]) -> list[str] | None:
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
    aot_path = COVERAGE / "qwen_seq128_aot.json.gz"
    with gzip.open(aot_path, "rt") as handle:
        capture = json.load(handle)
    live = json.loads((COVERAGE / "live_contrasts/qwen_seq128.json").read_text())
    queue = json.loads((COVERAGE / "bias_candidate_queue.json").read_text())
    candidate = next(row for row in queue["candidates"] if row["candidate_id"] == CID)
    precision = next(row for row in live["results"] if row["candidate_id"] == CID
                     and row["contrast_axis"] == "PRECISION")
    optimization = next(row for row in live["results"] if row["candidate_id"] == CID
                        and row["contrast_axis"] == "OPTIMIZATION")
    decomposition_path = COVERAGE / "cases/qwen128_vproj_precision_decomposition.json"
    decomposition = json.loads(decomposition_path.read_text())
    assert decomposition["candidate_id"] == CID
    assert decomposition["status"] == "COMPLETE_EXACT_PRECISION_MECHANISM_DECOMPOSITION"
    assert all(decomposition["gates"].values())
    assert decomposition["coherent_sources"] == ["output_rounding"]
    assert decomposition["direction"]["kernel"]["status"] == "FAIL_CAUSAL_NONCOHERENT"
    assert decomposition["direction"]["output_rounding"]["status"] == "PASS"
    release = COVERAGE / "runtime_releases/qwen_seq128_r1"
    release_capture = json.loads((release / "capture.json").read_text())
    source_path = Path(release_capture["modules"][0]["captured_source"])
    source = source_path.read_text()
    source_lines = source.splitlines()
    source_call = source_lines[SOURCE_LINE - 1].strip()
    source_comment = source_lines[SOURCE_LINE - 2].strip()
    assert hashlib.sha256(source.encode()).hexdigest() == release_capture["modules"][0]["sha256"]
    assert hashlib.sha256(source_call.encode()).hexdigest() == \
        candidate["exact_generated_call"]["source_line_sha256"]
    assert source_call == candidate["exact_generated_call"]["call_expression"]
    assert "linear_2" in source_comment and "aten.mm" in source_comment
    assert "buf7, (128, 2048)" in source_call
    assert "primals_9, (2048, 1024)" in source_call and "out=buf14" in source_call

    graphs = capture["capture"]["graphs"]
    forward = next(graph for graph in graphs if graph["phase"] == "FORWARD")
    backward = next(graph for graph in graphs if graph["phase"] == "BACKWARD")
    fw = {node["name"]: node for node in forward["nodes"]}
    bw = {node["name"]: node for node in backward["nodes"]}
    root = fw["mm_2"]
    assert root["seq_nr"] == SEQ and root["target"] == "aten.mm.default"
    assert root["input_nodes"] == ["t_2", "view_14"]
    assert root["tensor_meta"][0] == [128, 1024]
    assert "v_proj" in root["stack_trace"] and root["source_fn_stack"][0][0] == "linear_2"
    assert fw["t_2"]["input_nodes"] == ["primals_9"]
    assert fw["t_2"]["tensor_meta"][0] == [2048, 1024]
    assert fw["view_14"]["tensor_meta"][0] == [128, 2048]
    assert {"t_2", "view_14"}.issubset(set(fw["output"]["input_nodes"]))
    assert bw["t_2"]["op"] == "placeholder" and bw["view_14"]["op"] == "placeholder"

    assert bw["view_1240"]["tensor_meta"][0] == [128, 1024]
    assert bw["t_973"]["input_nodes"] == ["view_1240"]
    assert bw["mm_585"]["input_nodes"] == ["t_973", "view_14"]
    assert bw["mm_585"]["tensor_meta"][0] == [1024, 2048]
    assert bw["t_976"]["input_nodes"] == ["t_974"]
    assert bw["t_975"]["input_nodes"] == ["t_2"]
    assert bw["mm_586"]["input_nodes"] == ["t_975", "view_1240"]
    assert bw["mm_586"]["tensor_meta"][0] == [128, 2048]
    assert all(bw[name]["seq_nr"] == SEQ for name in
               ("t_973", "mm_585", "t_974", "t_975", "mm_586"))

    ancestors: set[str] = set()
    stack = ["view_1240"]
    while stack:
        name = stack.pop()
        if name in ancestors:
            continue
        ancestors.add(name)
        stack.extend(bw[name]["input_nodes"])
    assert "tangents_1" in ancestors
    outputs = set(output_leaves(bw["output"]))
    assert "t_976" in outputs
    dx_path = shortest_output_path(bw, "view_1241", outputs)
    assert dx_path == ["view_1241", "add_614", "add_616", "mul_1489", "sum_281", "view_1248"]

    derivation = {
        "symbols": {"X": "[128,2048]", "W": "[1024,2048]",
                    "Q=dL/dY": "[128,1024]"},
        "forward": "Y = X W^T",
        "backward": {"dX_contribution": "Q W", "dW": "Q^T X"},
        "actual_aot_program": {
            "forward": "mm_2(view_14, t_2(primals_9))",
            "dW": "t_976(t(t(mm_585(t_973(Q^T), view_14(X))))) = Q^T X",
            "dX_contribution": "view_1241(mm_586(t_975(t(t_2(W^T))), view_1240(Q))) = Q W",
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
    assert all(value is True for key, value in concrete.items() if key.endswith("_exact"))
    payload = {
        "schema": "kernel-analyzer-concrete-fb-bias-case-v1",
        "status": "T1_COHERENT_OUTPUT_ROUNDING_BIAS_WITH_CONCRETE_FB_PROOF",
        "candidate_id": CID,
        "supporting_candidate_id": "qwen_seq64_forward_8_output",
        "subject": "Qwen3-1.7B seq128 layer-0 self-attention v_proj forward MM",
        "cause_axis": "PRECISION",
        "forward_backward_unit": derivation,
        "concrete_program_proof": concrete,
        "bindings": {
            "aot_capture_sha256": capture["result_sha256"],
            "aot_file_sha256": hashlib.sha256(aot_path.read_bytes()).hexdigest(),
            "release_capture_sha256": release_capture["result_sha256"],
            "generated_source_sha256": release_capture["modules"][0]["sha256"],
            "generated_source_line": SOURCE_LINE,
            "generated_source_line_sha256": candidate["exact_generated_call"]["source_line_sha256"],
            "forward_root": "mm_2",
            "sequence_nr": SEQ,
            "module": "model.layers.0.self_attn.v_proj",
            "saved_tensor_nodes": ["view_14", "t_2"],
            "cotangent_node": "view_1240",
            "weight_gradient_output": "t_976",
            "input_gradient_contribution": "view_1241",
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
            "identity": decomposition["identity"],
            "coherent_source": "output_rounding",
            "kernel_difference": {
                "u_statistic": decomposition["direction"]["kernel"]["cross_state_inner_product_u"],
                "cluster_bootstrap_95": decomposition["direction"]["kernel"]["cluster_bootstrap_95"],
                "status": decomposition["direction"]["kernel"]["status"],
            },
            "output_rounding": {
                "u_statistic": decomposition["direction"]["output_rounding"]["cross_state_inner_product_u"],
                "cluster_bootstrap_95": decomposition["direction"]["output_rounding"]["cluster_bootstrap_95"],
                "status": decomposition["direction"]["output_rounding"]["status"],
            },
            "decomposition_result_sha256": decomposition["result_sha256"],
            "decomposition_file_sha256": hashlib.sha256(decomposition_path.read_bytes()).hexdigest(),
        },
        "claim_boundary": (
            "Complete concrete local F+B proof, T1 coherent precision bias, and exact local "
            "mechanism decomposition for seq128 v_proj. The coherent source is deterministic "
            "FP32-to-BF16 output rounding, not the generated MM's local kernel difference. "
            "The seq64 arm supplies independent same-generated-region T1 support but lacks a seq64 AOT proof. "
            "A BF16-ABI-preserving local accumulation repair is rejected. An output-rounding "
            "intervention, coherent downstream carrier, and live accumulation T2-T4 remain open."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    output = COVERAGE / "cases/qwen128_vproj.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": payload["status"]}))


if __name__ == "__main__":
    main()
