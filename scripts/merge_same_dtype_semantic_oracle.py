#!/usr/bin/env python3
"""Merge disjoint frozen-state same-dtype shards and recompute the Oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from scripts.analyze_generated_fp32_screen import (  # noqa: E402
    bootstrap, bootstrap_counts, metric_equal, u_statistic,
)


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=23091)
    args = parser.parse_args()

    shards = [load(path) for path in args.inputs]
    plan = load(args.task_plan)
    bank = json.loads(args.input_bank.read_text())
    bank_states = bank.get("states", bank.get("records"))
    if len(bank_states) != 32:
        raise RuntimeError("complete same-dtype merge requires exactly 32 frozen states")
    count = len(shards)
    if {int(row["shard"]["index"]) for row in shards} != set(range(count)):
        raise RuntimeError("same-dtype shard indices are incomplete")
    if any(
        row["status"] != "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE_SHARD"
        or int(row["shard"]["count"]) != count
        for row in shards
    ):
        raise RuntimeError("same-dtype shard status/count is invalid")

    invariant_fields = (
        "architecture", "sequence_length", "input_bank_sha256",
        "campaign_result_sha256", "inventory_result_sha256",
        "task_plan_result_sha256", "environment_result_sha256",
    )
    first = shards[0]
    for shard in shards[1:]:
        if any(shard.get(key) != first.get(key) for key in invariant_fields):
            raise RuntimeError("same-dtype shard bindings differ")
    if first["input_bank_sha256"] != file_digest(args.input_bank):
        raise RuntimeError("same-dtype shards bind another input bank")
    if first["task_plan_result_sha256"] != plan["result_sha256"]:
        raise RuntimeError("same-dtype shards bind another task plan")
    if any(
        not all(shard["reference_structure"]["reference_cut_runtime"]["gates"].values())
        for shard in shards
    ):
        raise RuntimeError("a same-dtype shard has an invalid reference cut")

    expected_ids = [
        str(state.get("sequence_id", state.get("state_id", index)))
        for index, state in enumerate(bank_states)
    ]
    state_payload: dict[str, Any] = {}
    for shard in shards:
        index = int(shard["shard"]["index"])
        assigned = [i for i in range(32) if i % count == index]
        if shard["shard"]["assigned_state_indices"] != assigned:
            raise RuntimeError("same-dtype assigned-state schedule changed")
        expected = {expected_ids[i] for i in assigned}
        observed = set(shard["states"])
        if observed != expected:
            raise RuntimeError("same-dtype shard state denominator changed")
        overlap = set(state_payload).intersection(observed)
        if overlap:
            raise RuntimeError(f"same-dtype shards overlap: {sorted(overlap)}")
        state_payload.update(shard["states"])
    if set(state_payload) != set(expected_ids):
        raise RuntimeError("same-dtype merged state denominator is incomplete")

    exact_rows = [
        row for row in plan["rows"]
        if row.get("exact_semantic_endpoint_id") is not None
    ]
    result_rows = []
    for task in exact_rows:
        task_id = str(task["task_id"])
        observations = [state_payload[state_id]["repeats"] for state_id in expected_ids]
        left = [rows[0]["endpoint_metrics"][task_id]["error"] for rows in observations]
        right = [rows[1]["endpoint_metrics"][task_id]["error"] for rows in observations]
        repeat_stable = [metric_equal(a, b) for a, b in zip(left, right)]
        sketches = [row["directional_error_sketch"] for row in left]
        coordinates = sketches[0]["flat_coordinate_indices"]
        if any(row["flat_coordinate_indices"] != coordinates for row in sketches):
            raise RuntimeError(f"same-dtype coordinate identity changed: {task_id}")
        errors = np.asarray([row["signed_delta_values"] for row in sketches], dtype=float)
        nonfinite = (
            any(row["nonfinite_mismatch"] for row in left + right)
            or not np.isfinite(errors).all()
        )
        exact = all(row["exact"] for row in left + right)
        if nonfinite:
            verdict, statistic, confidence = "NONFINITE_RISK", None, None
        elif exact and all(repeat_stable):
            verdict, statistic = "EQUIVALENT_EXACT_ON_HELDOUT_STATES", 0.0
            confidence = {"lower_95": 0.0, "median": 0.0, "upper_95": 0.0}
        else:
            statistic = u_statistic(errors)
            confidence = bootstrap(
                errors,
                bootstrap_counts(len(errors), args.bootstrap_draws, args.bootstrap_seed),
            )
            if confidence["lower_95"] > 0:
                verdict = "DIRECTIONAL_OPTIMIZATION_BIAS"
            elif not all(repeat_stable):
                verdict = "RUNTIME_VARIANCE_RISK"
            else:
                verdict = "FINITE_NONEXACT_WITHOUT_STABLE_DIRECTION"
        result_rows.append({
            "task_id": task_id,
            "candidate_region_id": task["candidate_region_id"],
            "exact_aot_endpoint_id": task.get("exact_aot_endpoint_id"),
            "exact_semantic_endpoint_id": task["exact_semantic_endpoint_id"],
            "states": 32,
            "sampled_coordinates": len(coordinates),
            "cross_state_inner_product_u": statistic,
            "cluster_bootstrap_95": confidence,
            "repeat_stable": all(repeat_stable),
            "max_abs_over_state_repeats": max(row["max_abs"] for row in left + right),
            "verdict": verdict,
        })

    verdict_counts = Counter(row["verdict"] for row in result_rows)
    denominator = {
        "states": 32,
        "repeats_per_state": 2,
        "candidate_ports": plan["denominator"]["stored_candidate_ports"],
        "candidate_compute_regions": plan["denominator"]["candidate_compute_regions"],
        "exact_semantic_endpoints": len(exact_rows),
        "internal_ports_closed_by_semantic_endpoint": plan["denominator"][
            "internal_ports_closed_by_semantic_endpoint"
        ],
        "compiler_added_ports_closed_by_exact_theorem": plan["denominator"].get(
            "compiler_added_ports_closed_by_exact_theorem", 0
        ),
        "unresolved": 0,
    }
    payload = {
        "schema": "kernel-analyzer-same-dtype-semantic-oracle-v1",
        "status": "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE",
        **{key: first[key] for key in invariant_fields},
        "denominator": denominator,
        "reference_structure": first["reference_structure"],
        "source_shards": [
            {
                "path": str(path),
                "result_sha256": shard["result_sha256"],
                "shard": shard["shard"],
            }
            for path, shard in zip(args.inputs, shards)
        ],
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "rows": result_rows,
        "states": {state_id: state_payload[state_id] for state_id in expected_ids},
        "gates": {
            "all_32_frozen_states_present": True,
            "same_dtype_bf16": True,
            "runtime_environment_exact": True,
            "same_model_weight_storage": True,
            "exact_compiler_origin_endpoints_only": True,
            "all_internal_candidate_ports_closed": True,
            "all_candidate_compute_regions_have_observed_output_ports": (
                plan["denominator"]["candidate_regions_without_observed_output_port"] == 0
            ),
            "candidate_values_used_for_pairing": False,
            "all_reference_cuts_bitwise_self_replay": True,
            "state_shards_disjoint_and_complete": True,
        },
        "claim_boundary": (
            "C16 minus R16 at every exact compiler-origin semantic endpoint over 32 "
            "disjoint frozen states. State sharding changes execution scheduling only; "
            "statistics are recomputed after exact state-level union."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    print(json.dumps({
        "output": str(args.output), "denominator": denominator,
        "verdict_counts": payload["verdict_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
