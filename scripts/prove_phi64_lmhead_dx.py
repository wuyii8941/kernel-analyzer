#!/usr/bin/env python3
"""Close the concrete F+B proof for the Phi-4 seq64 lm_head dX candidate."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CID = "phi4_seq64_backward_497_output"
SEQ = 13520


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    with gzip.open(COVERAGE / "phi4_seq64_aot.json.gz", "rt") as handle:
        capture = json.load(handle)
    live = json.loads((COVERAGE / "live_contrasts/phi4_seq64.json").read_text())
    queue = json.loads((COVERAGE / "bias_candidate_queue.json").read_text())
    audit = json.loads((COVERAGE / "candidate_aot_vjp_audit.json").read_text())
    candidate = next(row for row in queue["candidates"] if row["candidate_id"] == CID)
    audit_row = next(row for row in audit["rows"] if row["candidate_id"] == CID)
    precision = next(row for row in live["results"] if row["candidate_id"] == CID
                     and row["contrast_axis"] == "PRECISION")
    optimization = next(row for row in live["results"] if row["candidate_id"] == CID
                        and row["contrast_axis"] == "OPTIMIZATION")
    decomposition_path = COVERAGE / "cases/phi4_seq64_lmhead_dx_precision_decomposition.json"
    decomposition = json.loads(decomposition_path.read_text())
    assert decomposition["candidate_id"] == CID and all(decomposition["gates"].values())
    assert decomposition["coherent_sources"] == ["kernel"]
    graphs = capture["capture"]["graphs"]
    forward = next(g for g in graphs if g["phase"] == "FORWARD"
                   and any(n.get("seq_nr") == SEQ and n["name"] == "mm" for n in g["nodes"]))
    backward = next(g for g in graphs if g["phase"] == "BACKWARD"
                    and any(n.get("seq_nr") == SEQ and n["name"] == "mm_2" for n in g["nodes"]))
    fw = {n["name"]: n for n in forward["nodes"]}
    bw = {n["name"]: n for n in backward["nodes"]}
    assert fw["mm"]["input_nodes"] == ["t", "view"]
    assert fw["mm"]["tensor_meta"][0] == [64, 200064]
    assert {"t", "view"}.issubset(set(fw["output"]["input_nodes"]))
    assert bw["mm_1"]["input_nodes"] == ["t_1", "view"]
    assert bw["t_1"]["input_nodes"] == ["view_4"]
    assert bw["mm_2"]["input_nodes"] == ["t_3", "view_4"]
    assert bw["t_3"]["input_nodes"] == ["t"]
    assert bw["view_5"]["input_nodes"] == ["mm_2"]
    assert bw["t_4"]["input_nodes"] == ["t_2"]
    assert set(bw["output"]["input_nodes"]) == {"view_5", "t_4"}
    assert bw["view_4"]["tensor_meta"][0] == [64, 200064]
    assert bw["mm_2"]["tensor_meta"][0] == [64, 3072]
    assert {"tangents_1", "tangents_2"}.issubset({n["name"] for n in backward["nodes"]})
    assert audit_row["root"] == "mm_2" and audit_row["sequence_nr"] == SEQ
    assert candidate["exact_generated_call"]["source_line_sha256"] == \
        "441bcc9703afec7c274028d168a486f7d79ace360d7b4551d4e9fef2ca6c7c72"
    derivation = {
        "symbols": {"X": "[64,3072]", "W": "[200064,3072]",
                    "Q=dL/dY": "[64,200064]"},
        "forward": "Y = X W^T",
        "backward": {"dX": "Q W", "dW": "Q^T X"},
        "actual_aot_program": {
            "forward": "mm(view, t(primals_2))",
            "dW": "t(t(mm(t(Q), X))) = Q^T X",
            "dX": "view(mm(Q, t(t(W)))) = Q W",
        },
        "candidate_generated_call": "extern_kernels.mm(Q[64,200064], W[200064,3072])",
    }
    derivation_hash = canonical(derivation)
    concrete = {
        "saved_tensor_origins_exact": True,
        "cotangent_edge_exact": True,
        "backward_program_matches_analytic_vjp": True,
        "non_tensor_arguments_exact": True,
        "output_edges_exact": True,
        "forward_program_sha256": forward["code_sha256"],
        "backward_program_sha256": backward["code_sha256"],
        "analytic_derivation_sha256": derivation_hash,
    }
    payload = {
        "schema": "kernel-analyzer-concrete-fb-bias-case-v1",
        "status": "T1_COHERENT_PRECISION_BIAS_WITH_CONCRETE_FB_PROOF",
        "candidate_id": CID,
        "subject": "Phi-4-mini seq64 lm_head input-gradient MM",
        "cause_axis": "PRECISION",
        "forward_backward_unit": derivation,
        "concrete_program_proof": concrete,
        "bindings": {
            "aot_capture_sha256": capture["result_sha256"],
            "forward_graph_index": forward["graph_index"],
            "backward_graph_index": backward["graph_index"],
            "sequence_nr": SEQ,
            "generated_source_line_sha256": candidate["exact_generated_call"]["source_line_sha256"],
            "compiler_carried_root": audit_row["root"],
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
        "claim_boundary": "Complete local F+B and T1 coherent precision bias; downstream accumulation/causal repair T2-T4 remain open.",
    }
    repair_path = COVERAGE / "cases/phi4_seq64_lmhead_dx_repair_32state.json"
    trajectory_path = COVERAGE / "cases/phi4_seq64_lmhead_dx_trajectory.json"
    if repair_path.exists() and trajectory_path.exists():
        repair = json.loads(repair_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        carrier = repair["final_norm_weight_carrier"]
        trajectory_gates = trajectory["gates"]
        repair_reductions = [
            1.0 - row["repair_local"]["repair_vs_fp32_l2"]
            / row["repair_local"]["candidate_vs_fp32_l2"]
            for row in repair["states"]
        ]
        all_repair_gates = (
            repair["local_repair_improves_every_state"]
            and carrier["status"] == "PASS"
            and all(row["standard_endpoint"]["loss"] == row["repair_endpoint"]["loss"]
                    and row["standard_endpoint"] == row["sham_endpoint"]
                    and row["changed_parameter_count"] == 194
                    for row in repair["states"])
        )
        all_trajectory_gates = all(trajectory_gates.values())
        if all_repair_gates and all_trajectory_gates:
            payload["status"] = "COMPLETE_BOUNDED_FLASH_STYLE_FB_BIAS_CASE"
            payload["causal_repair"] = {
                "states": len(repair["states"]),
                "intervention": repair["intervention"],
                "local_error_reduction_min": min(repair_reductions),
                "local_error_reduction_mean": sum(repair_reductions) / len(repair_reductions),
                "local_error_reduction_max": max(repair_reductions),
                "loss_exact_states": 32, "sham_all_gradient_exact_states": 32,
                "changed_parameter_count_every_state": 194,
                "final_norm_weight_carrier": carrier,
                "artifact_sha256": repair["result_sha256"],
            }
            payload["live_accumulation"] = {
                "steps": len(trajectory["steps"]),
                "updated_parameter": trajectory["updated_parameter"],
                "optimizer": trajectory["optimizer"],
                "other_parameters_frozen": trajectory["frozen_other_parameters"],
                "gates": trajectory_gates,
                "first_master_arm_distance_l2": trajectory["steps"][0]["master_arm_distance_l2"],
                "final_master_arm_distance_l2": trajectory["steps"][-1]["master_arm_distance_l2"],
                "final_materialized_bf16_arm_distance_l2": trajectory["steps"][-1]["materialized_bf16_arm_distance_l2"],
                "master_distance_monotone": all(
                    right["master_arm_distance_l2"] >= left["master_arm_distance_l2"]
                    for left, right in zip(trajectory["steps"], trajectory["steps"][1:])
                ),
                "artifact_sha256": trajectory["result_sha256"],
            }
            payload["claim_boundary"] = (
                "Complete F+B, T1, causal repair, coherent immediate parameter-gradient carrier, "
                "and paired live accumulation for model.norm.weight. Other parameters are frozen "
                "in T4, so this is not a full-model optimizer-trajectory claim. Exact source "
                "decomposition attributes the coherent local direction to the MM kernel difference; "
                "the output-rounding term is noncoherent."
            )
    payload["result_sha256"] = canonical(payload)
    output = COVERAGE / "cases/phi4_seq64_lmhead_dx.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": payload["status"]}))


if __name__ == "__main__":
    main()
