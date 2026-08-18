#!/usr/bin/env python3
"""Freeze exact generated-call candidates for post-denominator bias diagnosis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/bias_candidate_queue.json",
    )
    args = parser.parse_args()
    positive_denominator = {"triton": 0, "nontriton": 0, "cells": []}
    for model in ("qwen", "mamba", "phi4", "deepseek8b"):
        for seq_len in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{seq_len}_r1"
            cell = {"model": model, "sequence_length": seq_len}
            for mode in ("triton", "nontriton"):
                oracle = load_gzip(release / f"{mode}_oracle.json.gz")
                count = sum(
                    row["verdict"] == "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
                    for row in oracle["rows"]
                )
                cell[f"{mode}_positive"] = count
                cell[f"{mode}_endpoint_denominator"] = len(oracle["rows"])
                positive_denominator[mode] += count
            positive_denominator["cells"].append(cell)
    positive_denominator["total"] = (
        positive_denominator["triton"] + positive_denominator["nontriton"]
    )

    selections = (
        {
            "sequence_length": 64, "region_id": "forward:4476", "layer": 22,
            "token_index": 11, "role": "FORWARD_SCAN_OUTPUT",
        },
        {
            "sequence_length": 128, "region_id": "forward:8780", "layer": 22,
            "token_index": 40, "role": "FORWARD_SCAN_OUTPUT",
        },
        {
            "sequence_length": 256, "region_id": "forward:18444", "layer": 23,
            "token_index": 234, "role": "FORWARD_SCAN_OUTPUT",
        },
        {
            "sequence_length": 64, "region_id": "backward:5017", "layer": 23,
            "token_index": 62, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 22,
            "forward_source_node": "scan_output_1536", "c_offset": 5024,
        },
        {
            "sequence_length": 64, "region_id": "backward:8269", "layer": 1,
            "token_index": 42, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 1,
            "forward_source_node": "scan_output_109", "c_offset": 3424,
        },
        {
            "sequence_length": 64, "region_id": "backward:7057", "layer": 10,
            "token_index": 22, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 9,
            "forward_source_node": "scan_output_625", "c_offset": 1824,
        },
        {
            "sequence_length": 64, "region_id": "backward:6647", "layer": 12,
            "token_index": 37, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 11,
            "forward_source_node": "scan_output_774", "c_offset": 3024,
        },
        {
            "sequence_length": 64, "region_id": "backward:7061", "layer": 10,
            "token_index": 18, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 9,
            "forward_source_node": "scan_output_621", "c_offset": 1504,
        },
        {
            "sequence_length": 128, "region_id": "backward:16133", "layer": 1,
            "token_index": 85, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 0,
            "forward_source_node": "scan_output_85", "c_offset": 6864,
        },
        {
            "sequence_length": 128, "region_id": "backward:11111", "layer": 18,
            "token_index": 125, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 17,
            "forward_source_node": "scan_output_2352", "c_offset": 10064,
        },
        {
            "sequence_length": 128, "region_id": "backward:13297", "layer": 11,
            "token_index": 80, "role": "SCAN_OUTPUT_VJP_C",
            "layer_index_zero_based": 10,
            "forward_source_node": "scan_output_1390", "c_offset": 6464,
        },
    )
    candidates = []
    for selection in selections:
        seq_len = selection["sequence_length"]
        region_id = selection["region_id"]
        layer = selection["layer"]
        token_index = selection["token_index"]
        release = ROOT / f"results/coverage/runtime_releases/mamba_seq{seq_len}_r1"
        oracle = load_gzip(release / "nontriton_oracle.json.gz")
        inventory = load_gzip(release / "inventory.json.gz")
        observed = [
            row for row in oracle["rows"]
            if row["region_id"] == region_id and row["endpoint"] == "output"
        ]
        generated = [
            row for row in inventory["runtime_call_audit"]["rows"]
            if row.get("compute_region_id") == region_id
        ]
        if len(observed) != 1 or len(generated) != 1:
            raise RuntimeError(f"candidate identity is not unique: seq{seq_len} {region_id}")
        screen, call = observed[0], generated[0]
        if screen["verdict"] != "DIRECTIONAL_BIAS_SCREEN_POSITIVE":
            raise RuntimeError(f"selected candidate is no longer screen-positive: {region_id}")
        candidate_suffix = (
            "scan_output_vjp_c" if selection["role"] == "SCAN_OUTPUT_VJP_C"
            else "scan_output"
        )
        candidates.append({
            "candidate_id": f"mamba_seq{seq_len}_layer{layer}_token{token_index}_{candidate_suffix}",
            "architecture": "mamba_130m",
            "sequence_length": seq_len,
            "semantic_unit": {
                "layer": layer,
                "layer_index_zero_based": selection.get("layer_index_zero_based"),
                "layer_label_note": (
                    "candidate ID preserves the discovery label; layer_index_zero_based is authoritative"
                ),
                "token_index": token_index,
                "selected_edge": selection["role"],
                "forward": "y_t = S_t C_t, S_t in R^(1536x16), C_t in R^16",
                "vjp": "bar_S_t = bar_y_t C_t^T; bar_C_t = S_t^T bar_y_t",
                "carrier_hypothesis": "recurrent scan state S_t carries prior timestep rounding before a 16-term contraction",
            },
            "exact_generated_call": {
                "region_id": region_id,
                "phase": call["phase"],
                "function": call["function"],
                "implementation_kind": call["implementation_kind_or_helper_role"],
                "source_path": call["source_path"],
                "source_line": call["source_line"],
                "source_line_sha256": call["source_line_sha256"],
                "call_expression": call["call_expression"],
            },
            "sampled_t1": {
                "states": screen["states"],
                "repeats": 2,
                "sampled_coordinates": screen["sampled_coordinates"],
                "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                "rms_mean": screen["rms_mean_over_state_repeats"],
                "max_abs": screen["max_abs_over_state_repeats"],
                "runtime_unstable_states": screen["runtime_unstable_states"],
                "nonfinite_mismatch_count": screen["nonfinite_mismatch_count"],
                "verdict": screen["verdict"],
                "row_sha256": screen["row_sha256"],
            },
            "gates": {
                "sampled_t1": True,
                "full_coordinate_t1": False,
                "exact_generated_fb_binding": False,
                "causal_repair_with_sham": False,
                "complete_coherent_carrier": False,
                "accumulation": False,
            },
            "claim": "CANDIDATE_ONLY",
        })
        full_t1_path = ROOT / f"results/coverage/mamba_seq{seq_len}_scan_output_t1_full.json"
        if selection["role"] == "SCAN_OUTPUT_VJP_C":
            # This endpoint has shape 16, so the frozen all-call screen already
            # measured every coordinate for 32 states and two exact repeats.
            candidates[-1]["full_coordinate_t1"] = {
                "source": "frozen_nontriton_oracle",
                "row_sha256": screen["row_sha256"],
                "states": screen["states"],
                "coordinates": screen["sampled_coordinates"],
                "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                "passed": True,
            }
            candidates[-1]["generated_fb_binding"] = {
                "status": "BOUND_ONE_OF_TWO_VJP_EDGES",
                "forward_source_node": selection["forward_source_node"],
                "forward_call": (
                    "extern_kernels.mm(S[1536,16], C[16,1] at offset "
                    f"{selection['c_offset']}, out=y[1536,1])"
                ),
                "forward_token_evidence": (
                    f"{selection['c_offset']} = {token_index} * 80 + 64"
                ),
                "backward_source_node": "mm_default_1470",
                "backward_call": call["call_expression"],
                "derivation": "bar_C_62 = S_62^T bar_y_62",
                "missing_for_complete_unit": "bind and intervene on bar_S_62 = bar_y_62 C_62^T",
            }
            candidates[-1]["gates"]["full_coordinate_t1"] = True
            candidates[-1]["gates"]["exact_generated_fb_binding"] = "PARTIAL_ONE_VJP_EDGE"
            candidates[-1]["claim"] = "CANDIDATE_FULL_COORDINATE_PENDING_CAUSAL_T2"
            causal_path = ROOT / (
                f"results/coverage/mamba_seq{seq_len}_l{layer}_t{token_index}_vjp_c_causal_pilot.json"
            )
            if causal_path.exists():
                causal = json.loads(causal_path.read_text())
                if causal.get("candidate_id") != candidates[-1]["candidate_id"]:
                    raise RuntimeError(f"causal artifact binds another candidate: {causal_path}")
                passed = bool(causal["causal_t2_positive"])
                candidates[-1]["causal_t2"] = {
                    "path": str(causal_path.relative_to(ROOT)),
                    "result_sha256": causal["result_sha256"],
                    "states": causal["states"],
                    "repair_nonnull_at_declared_dtype_in_states": causal[
                        "repair_nonnull_at_declared_dtype_in_states"
                    ],
                    "repair_reached_parameter_gradients_in_states": causal[
                        "repair_reached_parameter_gradients_in_states"
                    ],
                    "passed": passed,
                }
                candidates[-1]["gates"]["causal_repair_with_sham"] = passed
                if not passed and causal["repair_nonnull_at_declared_dtype_in_states"] == 0:
                    candidates[-1]["claim"] = "REJECTED_PRECISION_ONLY_REFERENCE_CAST_NULL"
            continue
        if full_t1_path.exists():
            full_t1 = json.loads(full_t1_path.read_text())
            if (
                full_t1.get("candidate_id") != candidates[-1]["candidate_id"]
                or full_t1.get("status") != "COMPLETE_FULL_COORDINATE_T1"
                or full_t1.get("states") != 32
                or full_t1.get("coordinates") != 1536
            ):
                raise RuntimeError(f"invalid full-coordinate T1 artifact: {full_t1_path}")
            passed = bool(full_t1["directional_positive"])
            candidates[-1]["full_coordinate_t1"] = {
                "path": str(full_t1_path.relative_to(ROOT)),
                "result_sha256": full_t1["result_sha256"],
                "states": full_t1["states"],
                "coordinates": full_t1["coordinates"],
                "cross_state_inner_product_u": full_t1["cross_state_inner_product_u"],
                "cluster_bootstrap_95": full_t1["cluster_bootstrap_95"],
                "passed": passed,
            }
            candidates[-1]["gates"]["full_coordinate_t1_complete"] = True
            candidates[-1]["gates"]["full_coordinate_t1"] = passed
            if not passed:
                candidates[-1]["claim"] = "REJECTED_T1_SAMPLED_COORDINATE_FALSE_POSITIVE"
        else:
            pilot_path = ROOT / f"results/coverage/mamba_seq{seq_len}_scan_output_t1_pilot.json"
            if pilot_path.exists():
                pilot = json.loads(pilot_path.read_text())
                if (
                    pilot.get("candidate_id") != candidates[-1]["candidate_id"]
                    or pilot.get("status") != "COMPLETE_FULL_COORDINATE_T1_PILOT"
                    or pilot.get("coordinates") != 1536
                ):
                    raise RuntimeError(f"invalid full-coordinate pilot artifact: {pilot_path}")
                passed = bool(pilot["directional_positive"])
                candidates[-1]["full_coordinate_t1_pilot"] = {
                    "path": str(pilot_path.relative_to(ROOT)),
                    "result_sha256": pilot["result_sha256"],
                    "states": pilot["states"],
                    "coordinates": pilot["coordinates"],
                    "cross_state_inner_product_u": pilot["cross_state_inner_product_u"],
                    "cluster_bootstrap_95": pilot["cluster_bootstrap_95"],
                    "passed": passed,
                }
                candidates[-1]["gates"]["full_coordinate_t1_pilot"] = passed
                if not passed:
                    candidates[-1]["claim"] = (
                        "REJECTED_FULL_COORDINATE_PILOT_NOT_DIRECTIONAL"
                    )
    selected_regions = {
        (row["sequence_length"], row["exact_generated_call"]["region_id"])
        for row in candidates
    }
    # Preserve every remaining truly full-coordinate (16-element) Mamba
    # endpoint in the executable pre-semantic cast queue.  Mathematical F+B
    # binding is intentionally deferred until this necessary dtype gate passes.
    for seq_len in (64, 128, 256):
        release = ROOT / f"results/coverage/runtime_releases/mamba_seq{seq_len}_r1"
        oracle = load_gzip(release / "nontriton_oracle.json.gz")
        inventory = load_gzip(release / "inventory.json.gz")
        calls = {
            row.get("compute_region_id"): row
            for row in inventory["runtime_call_audit"]["rows"]
            if row.get("category") == "COMPUTE"
        }
        screens = sorted(
            (
                row for row in oracle["rows"]
                if row["verdict"] == "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
                and row["sampled_coordinates"] == 16
                and (seq_len, row["region_id"]) not in selected_regions
            ),
            key=lambda row: row["cluster_bootstrap_95"]["lower_95"],
            reverse=True,
        )
        for screen in screens:
            call = calls[screen["region_id"]]
            candidate_id = f"mamba_seq{seq_len}_{screen['region_id'].replace(':', '_')}_full16"
            candidates.append({
                "candidate_id": candidate_id,
                "architecture": "mamba_130m",
                "sequence_length": seq_len,
                "semantic_unit": {
                    "status": "PENDING_ONLY_IF_DECLARED_DTYPE_CAST_GATE_PASSES",
                    "reason": (
                        "Pre-semantic necessary-condition screen; no F+B case claim is permitted"
                    ),
                },
                "exact_generated_call": {
                    "region_id": screen["region_id"],
                    "phase": call["phase"],
                    "function": call["function"],
                    "implementation_kind": call["implementation_kind_or_helper_role"],
                    "source_path": call["source_path"],
                    "source_line": call["source_line"],
                    "source_line_sha256": call["source_line_sha256"],
                    "call_expression": call["call_expression"],
                },
                "full_coordinate_t1": {
                    "source": "frozen_nontriton_oracle",
                    "row_sha256": screen["row_sha256"],
                    "states": screen["states"],
                    "coordinates": 16,
                    "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                    "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                    "passed": True,
                },
                "gates": {
                    "sampled_t1": True,
                    "full_coordinate_t1": True,
                    "exact_generated_fb_binding": False,
                    "declared_dtype_cast_nonnull": False,
                    "causal_repair_with_sham": False,
                    "complete_coherent_carrier": False,
                    "accumulation": False,
                },
                "claim": "PRE_SEMANTIC_CAST_GATE_CANDIDATE_ONLY",
            })
            selected_regions.add((seq_len, screen["region_id"]))
    by_candidate_id = {row["candidate_id"]: row for row in candidates}
    for cast_path in sorted((ROOT / "results/coverage").glob("mamba_*cast_gate.json")):
        cast_gate = json.loads(cast_path.read_text())
        if cast_gate.get("status") != "COMPLETE_BATCH_CAST_GATE":
            continue
        for result in cast_gate["results"]:
            candidate = by_candidate_id.get(result["candidate_id"])
            if candidate is None:
                continue
            passed = bool(result["promote_to_causal_t2"])
            candidate["declared_dtype_cast_gate"] = {
                "path": str(cast_path.relative_to(ROOT)),
                "result_sha256": cast_gate["result_sha256"],
                "states": cast_gate["state_count"],
                "nonnull_states": result["nonnull_states"],
                "total_changed_coordinates": result["total_changed_coordinates"],
                "passed": passed,
            }
            candidate["gates"]["declared_dtype_cast_nonnull"] = passed
            if passed:
                candidate["claim"] = "CAST_GATE_POSITIVE_PENDING_FB_BINDING_AND_CAUSAL_T2"
            elif not candidate.get("causal_t2", {}).get("passed", False):
                candidate["claim"] = "REJECTED_PRECISION_ONLY_REFERENCE_CAST_NULL"

    # Exhaust the valid non-Triton screen-positive denominator.  Earlier
    # releases hard-coded an Mamba recurrent subset and left the remaining
    # positives only in source Oracles.  Selection priority may order work, but
    # it must never remove a positive from the executable/disposition queue.
    prefix_to_architecture = {
        "qwen": "qwen3_1p7b",
        "mamba": "mamba_130m",
        "phi4": "phi4_mini_3p8b",
        "deepseek8b": "deepseek_r1_0528_qwen3_8b",
    }
    queued_keys = {
        (
            "mamba" if row["architecture"] == "mamba_130m" else
            "qwen" if row["architecture"] == "qwen3_1p7b" else
            "phi4" if row["architecture"] == "phi4_mini_3p8b" else
            "deepseek8b",
            int(row["sequence_length"]),
            str(row["exact_generated_call"]["region_id"]),
            str(row.get("sampled_t1", row.get("full_coordinate_t1", {})).get("endpoint", "output")),
        )
        for row in candidates
    }
    # Existing Mamba rows predate the explicit endpoint field; their selected
    # target is the generated output.
    queued_regions = {(key[0], key[1], key[2]) for key in queued_keys}
    for prefix, architecture in prefix_to_architecture.items():
        for seq_len in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{prefix}_seq{seq_len}_r1"
            oracle = load_gzip(release / "nontriton_oracle.json.gz")
            inventory = load_gzip(release / "inventory.json.gz")
            calls = {
                str(row.get("compute_region_id")): row
                for row in inventory["runtime_call_audit"]["rows"]
                if row.get("category") == "COMPUTE"
            }
            positives = sorted(
                (
                    row for row in oracle["rows"]
                    if str(row["verdict"]).startswith("DIRECTIONAL_BIAS_SCREEN_POSITIVE")
                ),
                key=lambda row: (str(row["region_id"]), str(row["endpoint"])),
            )
            for screen in positives:
                region_id = str(screen["region_id"])
                if (prefix, seq_len, region_id) in queued_regions:
                    continue
                call = calls.get(region_id)
                if call is None:
                    raise RuntimeError(
                        f"screen-positive runtime region absent from inventory: "
                        f"{prefix}/seq{seq_len}/{region_id}"
                    )
                safe_region = region_id.replace(":", "_")
                endpoint = str(screen["endpoint"])
                candidates.append({
                    "candidate_id": (
                        f"{prefix}_seq{seq_len}_{safe_region}_{endpoint}"
                    ),
                    "architecture": architecture,
                    "sequence_length": seq_len,
                    "semantic_unit": {
                        "status": "PENDING_EXACT_FB_BINDING",
                        "reason": (
                            "Exhaustive valid precision-screen positive; no semantic or case "
                            "claim is permitted before full-coordinate and F+B gates."
                        ),
                    },
                    "exact_generated_call": {
                        "region_id": region_id,
                        "phase": call["phase"],
                        "function": call["function"],
                        "implementation_kind": call["implementation_kind_or_helper_role"],
                        "source_path": call["source_path"],
                        "source_line": call["source_line"],
                        "source_line_sha256": call["source_line_sha256"],
                        "call_expression": call["call_expression"],
                    },
                    "sampled_t1": {
                        "source": "frozen_nontriton_precision_oracle",
                        "endpoint": endpoint,
                        "row_sha256": screen["row_sha256"],
                        "states": screen["states"],
                        "coordinates": screen["sampled_coordinates"],
                        "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                        "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                        "precision_contrast_only": True,
                    },
                    "gates": {
                        "sampled_precision_screen": True,
                        "full_coordinate_t1": False,
                        "exact_generated_fb_binding": False,
                        "same_dtype_optimization_contrast": False,
                        "causal_repair_with_sham": False,
                        "complete_coherent_carrier": False,
                        "accumulation": False,
                    },
                    "claim": "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING",
                })
                queued_regions.add((prefix, seq_len, region_id))
    output = {
        "schema": "kernel-analyzer-bias-candidate-queue-v2",
        "status": "FROZEN_CANDIDATES_PENDING_FULL_COORDINATE_AND_CAUSAL_GATES",
        "selection_policy": (
            "Every valid non-Triton runtime endpoint with a repeat-stable positive sampled-coordinate "
            "cross-state U-statistic enters the queue. Priority changes execution order only; it never "
            "removes a positive from the denominator."
        ),
        "candidate_count": len(candidates),
        "selected_for_execution_count": len(candidates),
        "candidate_disposition": {
            "cast_gate_positive": sum(
                row["claim"] == "CAST_GATE_POSITIVE_PENDING_FB_BINDING_AND_CAUSAL_T2"
                for row in candidates
            ),
            "precision_only_cast_null": sum(
                row["claim"] == "REJECTED_PRECISION_ONLY_REFERENCE_CAST_NULL"
                for row in candidates
            ),
            "full_coordinate_direction_rejected": sum(
                row["claim"] in {
                    "REJECTED_T1_SAMPLED_COORDINATE_FALSE_POSITIVE",
                    "REJECTED_FULL_COORDINATE_PILOT_NOT_DIRECTIONAL",
                }
                for row in candidates
            ),
            "pending_cast_gate": sum(
                row["claim"] in {
                    "PRE_SEMANTIC_CAST_GATE_CANDIDATE_ONLY",
                    "CANDIDATE_FULL_COORDINATE_PENDING_CAUSAL_T2",
                }
                for row in candidates
            ),
            "pending_exhaustive_full_coordinate_and_fb_binding": sum(
                row["claim"] == "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING"
                for row in candidates
            ),
        },
        "screen_positive_denominator": positive_denominator,
        "triton_screen_validity": (
            {
                "status": "INVALID_REFERENCE_ABI",
                "path": "results/coverage/triton_reference_abi_audit.json",
                "disposition": (
                    "Counts retained as historical screen outputs; none is a candidate label."
                ),
            }
            if (ROOT / "results/coverage/triton_reference_abi_audit.json").exists()
            else {"status": "PENDING_ABI_AUDIT"}
        ),
        "unselected_screen_positives_disposition": (
            "NONE_FOR_VALID_NONTRITON_POSITIVES. All valid non-Triton positives are queued. "
            "Invalid-ABI Triton outputs are historical invalid observations, not candidates."
        ),
        "execution_order": [
            candidates[4]["candidate_id"],
            candidates[3]["candidate_id"],
            candidates[5]["candidate_id"],
            candidates[6]["candidate_id"],
            candidates[7]["candidate_id"],
            candidates[8]["candidate_id"],
            candidates[9]["candidate_id"],
            candidates[10]["candidate_id"],
            candidates[0]["candidate_id"],
            candidates[2]["candidate_id"],
            candidates[1]["candidate_id"],
        ] + [
            row["candidate_id"] for row in candidates
            if row["claim"] == "PRE_SEMANTIC_CAST_GATE_CANDIDATE_ONLY"
        ] + [
            row["candidate_id"] for row in candidates
            if row["claim"] == "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING"
        ],
        "candidates": candidates,
        "claim_boundary": (
            "No row is a bias case until full-coordinate T1, actual generated F+B binding, "
            "reference repair plus restoration sham, complete carrier, and accumulation gates pass."
        ),
    }
    if (
        output["candidate_disposition"]["pending_cast_gate"] == 0
        and output["candidate_disposition"]["cast_gate_positive"] == 0
        and output["candidate_disposition"]["pending_exhaustive_full_coordinate_and_fb_binding"] == 0
        and sum(output["candidate_disposition"].values()) == len(candidates)
    ):
        output["status"] = "RESOLVED_NO_NEW_COMPLETE_CASES"
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
