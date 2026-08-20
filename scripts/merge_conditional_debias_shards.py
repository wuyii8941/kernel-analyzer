#!/usr/bin/env python3
"""Merge condition-disjoint debias shards without inventing cross-state geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kernel_analyzer.bias_formation_v22 import aggregate_conditional_debias


RANDOMIZED = {"ROUNDING_ONLY", "JOINT"}


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def merge(shards: list[dict[str, Any]], *, minimum_conditions: int = 16) -> dict[str, Any]:
    if not shards:
        raise ValueError("no conditional-debias shards")
    identity = (
        shards[0]["candidate_id"], shards[0]["architecture"],
        shards[0]["carrier_parameter"], tuple(shards[0]["arms"]),
        shards[0]["bindings"]["source_line_sha256"],
    )
    rows = []
    shard_provenance = []
    for shard in shards:
        current = (
            shard["candidate_id"], shard["architecture"],
            shard["carrier_parameter"], tuple(shard["arms"]),
            shard["bindings"]["source_line_sha256"],
        )
        if current != identity or shard.get("conditional_debias_enabled") is not True:
            raise ValueError("conditional shards have different bindings")
        if not str(shard.get("status", "")).startswith("COMPLETE_CONDITIONAL_DEBIAS_"):
            raise ValueError("conditional shard is incomplete")
        rows.extend(shard["states"])
        shard_provenance.append({
            "state_start": shard["bindings"]["state_start"],
            "state_count": shard["bindings"]["state_count"],
            "result_sha256": shard["result_sha256"],
        })
    state_ids = [str(row["state_id"]) for row in rows]
    if len(state_ids) < minimum_conditions or len(set(state_ids)) != len(state_ids):
        raise ValueError("merged conditions are insufficient or overlap")
    rows.sort(key=lambda row: str(row["state_id"]))
    summaries = {}
    for arm in shards[0]["arms"]:
        if arm not in RANDOMIZED:
            summaries[arm] = {"status": "NOT_APPLICABLE_DETERMINISTIC_ARM"}
            continue
        summaries[arm] = aggregate_conditional_debias({
            str(row["state_id"]): row["arms"][arm]["conditional_debias"]["layers"]
            for row in rows
        })
    result = {
        "schema": "kernel-analyzer-mm-conditional-debias-merged-v1",
        "status": "COMPLETE_CONDITIONAL_DEBIAS_CONFIRMATION",
        "candidate_id": identity[0],
        "architecture": identity[1],
        "carrier_parameter": identity[2],
        "arms": list(identity[3]),
        "states": rows,
        "conditional_debias_enabled": True,
        "conditional_debias_summary": summaries,
        "direction": "NOT_COMPUTED_NOT_A_CONDITIONAL_GATE",
        "bindings": {
            "source_line_sha256": identity[4],
            "condition_disjoint_shards": sorted(
                shard_provenance, key=lambda row: row["state_start"]
            ),
            "shard_merge_does_not_pool_condition_vectors": True,
        },
        "claim_boundary": (
            "Only fixed-condition certificates are merged. No cross-condition direction "
            "or bootstrap is reconstructed, because global direction is not a conditional "
            "bias requirement."
        ),
    }
    result["result_sha256"] = canonical(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-conditions", type=int, default=16)
    args = parser.parse_args()
    result = merge(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs],
        minimum_conditions=args.minimum_conditions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
