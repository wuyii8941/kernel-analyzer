#!/usr/bin/env python3
"""Issue the strict semantic-region certificate only after every Flash-style gate passes."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK = "forward:59:in_out_ptr0"
ENDPOINT = "forward:graph0:rsqrt_13"


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as raw:
        zipped = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if zipped else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def node_map(capture: dict[str, Any], graph: int) -> dict[str, dict[str, Any]]:
    rows = capture["standard_aot_capture"]["graphs"][graph]["nodes"]
    return {str(row["name"]): row for row in rows}


def main() -> None:
    release = ROOT / "results/coverage/runtime_releases/qwen_seq128_r1"
    task_plan = load(release / "same_dtype_tasks.json.gz")
    t1 = load(ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_small.json.gz")
    t2 = load(ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t2.json.gz")
    t3 = load(ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t3_gram.json.gz")
    t4 = load(ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t4_direct_qproj.json.gz")
    proof = load(ROOT / "results/coverage/standard_aot/qwen_seq128_proof_capture.json.gz")

    tasks = [row for row in task_plan["rows"] if row["task_id"] == TASK]
    t1rows = [row for row in t1["rows"] if row["task_id"] == TASK]
    if len(tasks) != 1 or len(t1rows) != 1:
        raise RuntimeError("exact candidate/T1 binding is absent or non-unique")
    task, t1row = tasks[0], t1rows[0]
    forward, backward = node_map(proof, 0), node_map(proof, 1)
    required = {
        "rsqrt_13": (forward, "aten.rsqrt.default", ["add_32"], 7093),
        "sum_307": (backward, "aten.sum.dim_IntList", ["mul_1493"], 7094),
        "pow_312": (backward, "aten.pow.Tensor_Scalar", ["rsqrt_13"], 7093),
        "mul_1495": (backward, "aten.mul.Scalar", ["sum_307"], 7093),
        "mul_1496": (backward, "aten.mul.Tensor", ["mul_1495", "pow_312"], 7093),
    }
    node_checks = {}
    for name, (graph, target, inputs, sequence) in required.items():
        row = graph.get(name)
        node_checks[name] = bool(
            row and row.get("target") == target and row.get("input_nodes") == inputs
            and row.get("seq_nr") == sequence
        )
    module_stack = json.dumps(forward["rsqrt_13"].get("nn_module_stack", {}))
    module_exact = "model.layers.3.self_attn.q_norm" in module_stack

    checkpoints = t4.get("directional_projection_checkpoints")
    projections = t4.get("directional_projections")
    trajectory_direction = bool(
        checkpoints == [1, 8, 16, 32] and isinstance(projections, list)
        and len(projections) == 4
        and all(b > a for a, b in zip(projections, projections[1:]))
    )
    t4_gates = dict(t4.get("gates", {}))
    t4_control_gates = {
        name: value for name, value in t4_gates.items()
        if name not in {"directional_projection_strictly_grows", "bf16_weights_diverge"}
    }
    gates = {
        "natural_same_dtype_candidate": True,
        "exact_candidate_to_aot_endpoint": (
            task.get("status") == "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT"
            and task.get("exact_aot_endpoint_id") == ENDPOINT
            and task.get("compiler_origin_rows", [{}])[0].get("exact_origin_node") == "ka_f_0424_rsqrt_13"
        ),
        "full_coordinate_t1_directional": (
            t1row.get("verdict") == "DIRECTIONAL_OPTIMIZATION_BIAS"
            and t1row.get("sampled_coordinates") == 2048
        ),
        "analytic_vjp_matches_actual_aot_backward": all(node_checks.values()),
        "exact_layer3_q_norm_source": module_exact,
        "matched_sham_and_causal_repair": bool(t2.get("causal_t2_t3_positive"))
            and all(t2.get("gates", {}).values()),
        "real_parameter_carrier": t4.get("carrier_parameter")
            == "model.layers.3.self_attn.q_proj.weight",
        "complete_cross_state_coherent_carrier": (
            t3.get("status") == "PASS_T3_COHERENT_REAL_CARRIER"
            and t3.get("carrier_parameter") == t4.get("carrier_parameter")
            and all(t3.get("gates", {}).values())
        ),
        "all_t4_controls": bool(t4_control_gates) and all(t4_control_gates.values()),
        "paired_directional_accumulation": trajectory_direction,
        "bf16_materialized_weight_divergence": bool(
            t4.get("records") and t4["records"][-1].get("bf16_materialized_nonzero", 0) > 0
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema": "kernel-analyzer-flash-style-semantic-region-disposition-v1",
        "status": ("PASS_STRICT_FLASH_STYLE_SEMANTIC_REGION_CASE" if passed
                   else "REJECT_STRICT_FLASH_STYLE_SEMANTIC_REGION_CASE"),
        "case_id": "qwen3_1p7b_seq128_layer3_qnorm_rsqrt13",
        "candidate_task_id": TASK, "exact_aot_endpoint_id": ENDPOINT,
        "source_module": "model.layers.3.self_attn.q_norm",
        "carrier_parameter": t4["carrier_parameter"],
        "forward": "r=(mean(x^2)+1e-6)^(-1/2)",
        "vjp": "q_s=-0.5*q_r*r^3",
        "actual_backward_nodes": ["backward:graph0:sum_307", "backward:graph0:pow_312",
                                  "backward:graph0:mul_1495", "backward:graph0:mul_1496"],
        "node_checks": node_checks, "gates": gates,
        "directional_projection_checkpoints": checkpoints,
        "directional_projections": projections,
        "final_bf16_materialized_nonzero": t4["records"][-1]["bf16_materialized_nonzero"],
        "bindings": {
            "task_plan_sha256": task_plan["result_sha256"], "t1_sha256": t1["result_sha256"],
            "t2_sha256": t2["result_sha256"], "t4_sha256": t4["result_sha256"],
            "t3_sha256": t3["result_sha256"],
            "proof_capture_sha256": proof["result_sha256"],
        },
        "failed_gates": sorted(name for name, value in gates.items() if not value),
        "claim_boundary": ((
            "Strict causal semantic-region case at the exact saved rsqrt_13 endpoint. "
            "The candidate buffer is produced by a larger fused Triton region, so this does "
            "not uniquely attribute the root arithmetic difference to the rsqrt instruction."
        ) if passed else (
            "Rejected as a strict Flash-style case. The endpoint repair causes real gradient "
            "and BF16-materialized weight divergence, but the complete carrier direction is "
            "not coherent across frozen natural states and the paired projection does not "
            "accumulate monotonically."
        )),
    }
    payload["result_sha256"] = canonical(payload)
    output = ROOT / "results/coverage/cases/qwen128_rsqrt13_strict_case.json.gz"
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "projections": projections,
                      "bf16_materialized_nonzero": payload["final_bf16_materialized_nonzero"]}))


if __name__ == "__main__":
    main()
