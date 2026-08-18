#!/usr/bin/env python3
"""Merge the complete Qwen external/direct same-precision census."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [load(path) for path in args.inputs]
    if {row["shard_index"] for row in shards} != set(range(len(shards))):
        raise RuntimeError("external/direct shard set is incomplete")
    if any(row["shard_count"] != len(shards) for row in shards):
        raise RuntimeError("external/direct shard count changed")
    if any(row["status"] != "COMPLETE_EXTERNAL_DIRECT_HELDOUT_SHARD" for row in shards):
        raise RuntimeError("external/direct shard is incomplete")
    inventory_hashes = {row["inventory_sha256"] for row in shards}
    if len(inventory_hashes) != 1:
        raise RuntimeError("inventory changed across shards")
    states = {}
    for shard in shards:
        if states.keys() & shard["states"].keys():
            raise RuntimeError("state repeated across shards")
        states.update(shard["states"])
    if len(states) != 32:
        raise RuntimeError("held-out state denominator changed")
    denominators = {
        (row["denominator"]["external"], row["denominator"]["direct_aten"])
        for row in shards
    }
    if len(denominators) != 1:
        raise RuntimeError("external/direct denominator changed across shards")
    expected_external, expected_direct = next(iter(denominators))

    external_exact = Counter()
    external_nonexact = Counter()
    direct_exact = 0
    direct_nonexact = 0
    for state_id, state in sorted(states.items()):
        if len(state["repeats"]) != 2:
            raise RuntimeError(f"repeat denominator changed for {state_id}")
        repeat_keys = []
        for repeat in state["repeats"]:
            if not repeat["observation_stable"]:
                raise RuntimeError(f"observer perturbed {state_id}")
            external = repeat["external"]
            if external["record_count"] != expected_external:
                raise RuntimeError(f"external denominator changed for {state_id}")
            keys = []
            for record in external["records"]:
                key = (record["phase"], record["symbol"], record["invocation_index"])
                keys.append(key)
                metric = record["candidate_vs_same_precision_reference"]
                (external_exact if metric["exact"] else external_nonexact)[key] += 1
                if not metric["candidate_finite"] or not metric["reference_finite"]:
                    raise RuntimeError(f"nonfinite external boundary: {key}")
            repeat_keys.append(keys)
            direct = repeat["direct_aten"]
            if expected_direct != 1 or direct["status"] != "COMPLETE_DIRECT_ATEN_INDEX_PUT_REFERENCE":
                raise RuntimeError(f"direct ATen incomplete for {state_id}")
            metric = direct["records"][0]["active_row_metrics"]
            if metric["exact"]:
                direct_exact += 1
            else:
                direct_nonexact += 1
        if repeat_keys[0] != repeat_keys[1] or len(set(repeat_keys[0])) != expected_external:
            raise RuntimeError(f"external invocation identity changed for {state_id}")

    payload = {
        "schema": "kernel-analyzer-qwen-external-direct-oracle-v1",
        "status": "COMPLETE_EXTERNAL_DIRECT_HELDOUT_CENSUS",
        "inventory_sha256": next(iter(inventory_hashes)),
        "denominator": {
            "heldout_states": 32, "repeats_per_state": 2,
            "external_invocations_per_repeat": expected_external,
            "direct_aten_invocations_per_repeat": expected_direct,
            "external_records": 32 * 2 * expected_external,
            "direct_aten_records": 32 * 2 * expected_direct,
        },
        "verdict": {
            "external_exact_records": sum(external_exact.values()),
            "external_nonexact_records": sum(external_nonexact.values()),
            "direct_exact_records": direct_exact,
            "direct_nonexact_records": direct_nonexact,
        },
        "gates": {
            "all_invocations_retained": True,
            "all_repeats_nonperturbing": True,
            "all_values_finite": True,
            "invocation_identity_repeat_stable": True,
        },
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), **payload["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
