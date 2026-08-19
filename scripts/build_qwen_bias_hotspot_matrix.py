#!/usr/bin/env python3
"""Freeze Qwen bias-formation hotspots from training semantics, not verdicts.

The roster is selected before reading T1/T3/T4/SEUP outcomes.  Exact task and
module bindings are verified against the compiler-carried AOT graph and the
same-dtype task denominator.  The output is directly consumable by
capture_bound_endpoint_bias_formation_v21.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Frozen seq64 development search matrix.  Selection is by semantic bottleneck
# and layer, never by historical numerical verdict.
SPECS = (
    # Loss bottleneck: dlogits is deliberately included even though its local
    # residual need not be directional.  This is the transition old T1 gated
    # out before parameter-gradient transport was measured.
    ("loss_head", "qwen_seq64_ce_dlogits", "backward:515:in_out_ptr0", "model.norm.weight"),
    ("loss_head", "qwen_seq64_lm_head_dx", "backward:517:output_0", "model.norm.weight"),
    # Complete normalization VJP outputs, not isolated rsqrt/reduction ops.
    ("normalization", "qwen_seq64_l27_post_norm_vjp", "backward:527:in_out_ptr0", "model.layers.27.self_attn.o_proj.weight"),
    ("normalization", "qwen_seq64_l27_input_norm_vjp", "backward:550:in_out_ptr0", "model.layers.26.self_attn.o_proj.weight"),
    ("normalization", "qwen_seq64_l27_q_norm_vjp", "backward:540:out_ptr3", "model.layers.27.self_attn.q_proj.weight"),
    ("normalization", "qwen_seq64_l27_k_norm_vjp", "backward:543:in_out_ptr0", "model.layers.27.self_attn.k_proj.weight"),
    ("normalization", "qwen_seq64_l23_post_norm_vjp", "backward:659:in_out_ptr0", "model.layers.23.self_attn.o_proj.weight"),
    ("normalization", "qwen_seq64_l23_input_norm_vjp", "backward:682:in_out_ptr0", "model.layers.22.self_attn.o_proj.weight"),
    ("normalization", "qwen_seq64_l23_q_norm_vjp", "backward:672:out_ptr3", "model.layers.23.self_attn.q_proj.weight"),
    ("normalization", "qwen_seq64_l23_k_norm_vjp", "backward:675:in_out_ptr0", "model.layers.23.self_attn.k_proj.weight"),
    # Attention-state and transport bottlenecks preceding projection gradients.
    ("attention", "qwen_seq64_l23_saved_state", "backward:665:out_ptr0", "model.layers.23.self_attn.q_proj.weight"),
    ("attention", "qwen_seq64_l23_softmax_vjp", "backward:666:in_out_ptr0", "model.layers.23.self_attn.q_proj.weight"),
    ("attention", "qwen_seq64_l23_attention_dv", "backward:663:output_0", "model.layers.23.self_attn.v_proj.weight"),
    ("attention", "qwen_seq64_l23_attention_dp", "backward:664:output_0", "model.layers.23.self_attn.q_proj.weight"),
    ("attention", "qwen_seq64_l23_attention_dq", "backward:669:output_0", "model.layers.23.self_attn.q_proj.weight"),
    ("attention", "qwen_seq64_l23_attention_dk", "backward:670:output_0", "model.layers.23.self_attn.k_proj.weight"),
)


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json")
    args = parser.parse_args()
    release = ROOT / "results/coverage/runtime_releases/qwen_seq64_r1"
    plan = load(release / "same_dtype_tasks.json.gz")
    by_task = {str(row["task_id"]): row for row in plan["rows"]}
    capture = load(ROOT / "results/coverage/standard_aot/qwen_seq64_capture.json.gz")["capture"]
    nodes = {
        f'{str(node["phase"]).lower()}:graph{graph["graph_index"]}:{node["name"]}': node
        for graph in capture["graphs"] for node in graph["nodes"]
    }
    cases = []
    for family, case_id, task_id, carrier in SPECS:
        if task_id not in by_task:
            raise RuntimeError(f"frozen hotspot task is absent: {task_id}")
        task = by_task[task_id]
        endpoint = str(task["exact_aot_endpoint_id"])
        node = nodes.get(endpoint)
        if node is None or str(node["phase"]).upper() != "BACKWARD":
            raise RuntimeError(f"hotspot lacks an exact backward AOT node: {task_id}")
        stack = node.get("nn_module_stack") or node.get("fwd_nn_module_stack") or {}
        module_paths = [str(value[0]) for value in stack.values()
                        if isinstance(value, (list, tuple)) and value]
        cases.append({
            "case_id": case_id, "task_id": task_id, "carrier": carrier,
            "hotspot_family": family, "exact_aot_endpoint_id": endpoint,
            "aot_target": str(node.get("target")), "module_paths": module_paths,
            "candidate_region_id": task["candidate_region_id"],
            "implementation_kind": task["implementation_kind"],
            "selection_uses_historical_verdict": False,
        })
    result = {
        "schema": "kernel-analyzer-bias-hotspot-search-matrix-v1",
        "status": "FROZEN_BEFORE_FORMATION_MEASUREMENT",
        "model": "Qwen3-1.7B", "sequence_length": 64,
        "selection_principle": "TRAINING_SEMANTIC_BOTTLENECK_NOT_ERROR_MAGNITUDE",
        "families": {
            "loss_head": "logits/cross-entropy cotangent into hidden-state gradient",
            "normalization": "complete RMSNorm or q/k norm backward bottleneck",
            "attention": "layer-23 dP/dQ/dK/dV and projection backward bottleneck",
        },
        "case_count": len(cases), "cases": cases,
        "excluded_selection_inputs": ["T1", "T3", "T4", "SEUP", "trajectory_drift"],
        "bindings": {"task_plan_sha256": plan["result_sha256"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plan_output = args.output.with_name("qwen_seq64_capture_plan.json")
    plan_output.write_text(json.dumps({"cases": [
        {"case_id": row["case_id"], "task_id": row["task_id"], "carrier": row["carrier"]}
        for row in cases
    ]}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matrix": str(args.output), "capture_plan": str(plan_output),
                      "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
