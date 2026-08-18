#!/usr/bin/env python3
"""Freeze every same-dtype directional endpoint before Flash-style follow-up.

The endpoint denominator is authoritative.  Region and proof-owner groupings are
views used to schedule experiments; they never remove endpoint candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("qwen", "mamba", "phi4", "deepseek8b")
SHAPES = (64, 128, 256)


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root", type=Path,
        default=ROOT / "results/coverage/runtime_releases",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/cases/same_dtype_case_ledger.json.gz",
    )
    args = parser.parse_args()

    endpoints: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    for prefix in PREFIXES:
        for shape in SHAPES:
            run = f"{prefix}_seq{shape}_r1"
            directory = args.runtime_root / run
            oracle_path = directory / "same_dtype_oracle.json.gz"
            plan_path = directory / "same_dtype_tasks.json.gz"
            if not oracle_path.exists() or not plan_path.exists():
                missing.append(run)
                continue
            oracle, plan = load(oracle_path), load(plan_path)
            if oracle.get("status") != "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE":
                raise RuntimeError(f"incomplete same-dtype Oracle: {run}")
            if plan.get("status") != (
                "COMPLETE_ALL_CANDIDATE_PORTS_ASSIGNED_TO_EXACT_SEMANTIC_ENDPOINTS"
            ):
                raise RuntimeError(f"incomplete same-dtype task plan: {run}")
            if oracle.get("task_plan_result_sha256") != plan.get("result_sha256"):
                raise RuntimeError(f"Oracle/task-plan binding changed: {run}")
            by_task = {str(row["task_id"]): row for row in plan["rows"]}
            if len(by_task) != len(plan["rows"]):
                raise RuntimeError(f"duplicate task id: {run}")
            directional = [
                row for row in oracle["rows"]
                if row.get("verdict") == "DIRECTIONAL_OPTIMIZATION_BIAS"
            ]
            declared = int(oracle.get("verdict_counts", {}).get(
                "DIRECTIONAL_OPTIMIZATION_BIAS", -1
            ))
            if len(directional) != declared:
                raise RuntimeError(f"directional denominator changed: {run}")
            for row in directional:
                task_id = str(row["task_id"])
                task = by_task.get(task_id)
                if task is None:
                    raise RuntimeError(f"directional task absent from plan: {run}/{task_id}")
                endpoint_id = f"{prefix}:seq{shape}:{task_id}"
                endpoints.append({
                    "candidate_id": endpoint_id,
                    "model": prefix,
                    "sequence_length": shape,
                    "task_id": task_id,
                    "candidate_region_id": row["candidate_region_id"],
                    "phase": task["phase"],
                    "symbol": task["symbol"],
                    "implementation_kind": task["implementation_kind"],
                    "exact_aot_endpoint_id": row.get("exact_aot_endpoint_id"),
                    "exact_semantic_endpoint_id": row["exact_semantic_endpoint_id"],
                    "proof_owner_ids": task.get("proof_owner_ids", []),
                    "sampled_t1": {
                        "states": row["states"],
                        "coordinates": row["sampled_coordinates"],
                        "u": row["cross_state_inner_product_u"],
                        "bootstrap_95": row["cluster_bootstrap_95"],
                        "repeat_stable": row["repeat_stable"],
                        "max_abs": row["max_abs_over_state_repeats"],
                    },
                    "required_followup": [
                        "FULL_COORDINATE_T1", "CAUSAL_REPAIR_OR_SHAM",
                        "REAL_COHERENT_CARRIER", "PAIRED_ACCUMULATION",
                    ],
                    "completed_gates": ["EXACT_FORWARD_BACKWARD_CLOSURE"],
                    "disposition": "PENDING_COMPLETE_FLASH_STYLE_AUDIT",
                })
            cells.append({
                "cell": run,
                "candidate_ports": oracle["denominator"]["candidate_ports"],
                "exact_semantic_endpoints": oracle["denominator"]["exact_semantic_endpoints"],
                "directional_endpoints": len(directional),
                "oracle_result_sha256": oracle["result_sha256"],
                "task_plan_result_sha256": plan["result_sha256"],
            })

    if missing:
        raise RuntimeError("12-cell ledger is incomplete: " + ", ".join(missing))
    candidate_ids = [row["candidate_id"] for row in endpoints]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("candidate endpoint identity is not unique")

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in endpoints:
        owners = ",".join(row["proof_owner_ids"]) or "NO_OWNER"
        key = (
            f"{row['model']}:seq{row['sequence_length']}:"
            f"{row['candidate_region_id']}:{owners}"
        )
        grouped[key].append(row["candidate_id"])
    groups = [
        {"group_id": key, "endpoint_candidate_ids": sorted(ids),
         "endpoint_count": len(ids)}
        for key, ids in sorted(grouped.items())
    ]
    output = {
        "schema": "kernel-analyzer-exhaustive-same-dtype-case-ledger-v1",
        "status": "COMPLETE_12_CELL_DIRECTIONAL_ENDPOINT_LEDGER",
        "claim_boundary": (
            "Every directional endpoint is retained. Grouping schedules shared F+B "
            "experiments and never substitutes for endpoint-level disposition."
        ),
        "denominator": {
            "cells": len(cells),
            "directional_endpoints": len(endpoints),
            "candidate_region_owner_groups": len(groups),
            "by_model": dict(Counter(row["model"] for row in endpoints)),
            "by_implementation_kind": dict(Counter(
                row["implementation_kind"] for row in endpoints
            )),
        },
        "gates": {
            "all_12_cells_present": len(cells) == 12,
            "all_oracles_complete": True,
            "all_oracles_bind_exact_task_plans": True,
            "every_directional_endpoint_retained": (
                len(endpoints) == sum(row["directional_endpoints"] for row in cells)
            ),
            "no_top_k_or_priority_filter": True,
        },
        "cells": cells,
        "endpoint_candidates": endpoints,
        "region_owner_groups": groups,
    }
    if not all(output["gates"].values()):
        raise RuntimeError("exhaustive case ledger gate failed")
    output["result_sha256"] = digest(output)
    write(args.output, output)
    print(json.dumps({
        "status": output["status"], "denominator": output["denominator"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
