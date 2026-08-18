#!/usr/bin/env python3
"""Merge candidate-blind, dtype-mapped generated-region observations.

Unlike the original BF16 changed-site merger, this input is deliberately
restricted to exact semantic mappings between a non-BF16 generated symbol
and the frozen BF16 eager semantic boundary.  The merger preserves the
per-checkpoint endpoint metrics while keeping the mapping denominator and
unresolved topology explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, choices=(64, 128, 256), required=True)
    p.add_argument("--dtype-mapping", type=Path, required=True)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.dtype_mapping.read_text())
    if int(mapping["seq_len"]) != args.seq_len:
        raise RuntimeError("mapping shape mismatch")
    if bool(mapping.get("candidate_values_used_to_select_or_classify")):
        raise RuntimeError("mapping is not candidate-blind")
    mapped_rows = {
        (str(row["symbol"]), str(row["reference_symbol"])): row
        for row in mapping["rows"]
        if row.get("status") == "MAPPED"
    }
    expected_regions = int(mapping["denominator"]["mapped_invocations"])
    if len(mapped_rows) == 0 or expected_regions <= 0:
        raise RuntimeError("mapping has no mapped invocations")

    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(args.inputs):
        raw = json.loads(path.read_text())
        if int(raw["seq_len"]) != args.seq_len:
            raise RuntimeError(f"shape mismatch in {path}")
        if str(raw.get("dtype")) != str(mapping["dtype"]):
            raise RuntimeError(f"dtype mismatch in {path}")
        if bool(raw.get("tf32", False)) != bool(mapping.get("tf32", False)):
            raise RuntimeError(f"TF32 mismatch in {path}")
        if str(raw.get("dtype_mapping_sha256")) != digest(args.dtype_mapping):
            raise RuntimeError(f"mapping digest mismatch in {path}")
        gates = raw["gates"]
        if not all(bool(gates[key]) for key in (
            "all_expected_ordinary_regions_observed_twice",
            "all_changed_region_ids_retained_twice",
            "all_observation_repeats_stable",
        )):
            raise RuntimeError(f"worker gate failed in {path}")
        if int(raw["changed_region_ids_expected"]) != expected_regions:
            raise RuntimeError(f"mapped-region denominator mismatch in {path}")
        artifacts.append((path, raw))
    if not artifacts:
        raise RuntimeError("no worker artifacts")
    steps = [int(raw["checkpoint_step"]) for _, raw in artifacts]
    if len(set(steps)) != len(steps):
        raise RuntimeError("duplicate checkpoint step")
    artifacts.sort(key=lambda pair: int(pair[1]["checkpoint_step"]))
    steps = [int(raw["checkpoint_step"]) for _, raw in artifacts]

    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, raw in artifacts:
        for repeat in raw["repeats"]:
            for record in repeat["records"]:
                key = (str(record["symbol"]), str(record["reference_symbol"]))
                if key not in mapped_rows:
                    raise RuntimeError(f"worker emitted unmapped region {key}")
                for endpoint, metric in record["endpoint_metrics"].items():
                    by_region[str(record["region_id"])].append({
                        "step": int(raw["checkpoint_step"]),
                        "repeat_id": int(repeat["repeat_id"]),
                        "phase": record["phase"],
                        "symbol": record["symbol"],
                        "reference_symbol": record["reference_symbol"],
                        "endpoint": endpoint,
                        "metric": metric,
                    })

    rows = []
    for region_id, values in sorted(by_region.items()):
        metrics = [item["metric"] for item in values]
        signed = [float(item["metric"].get("signed_mean", 0.0)) for item in values]
        expected_pairs = {(step, repeat_id) for step in steps for repeat_id in (0, 1)}
        observed_pairs = {(int(item["step"]), int(item["repeat_id"])) for item in values}
        rows.append({
            "region_id": region_id,
            "symbol": values[0]["symbol"],
            "reference_symbol": values[0]["reference_symbol"],
            "phase": values[0]["phase"],
            "endpoint_count": len({str(item["endpoint"]) for item in values}),
            "record_count": len(values),
            "expected_record_pairs": len(expected_pairs),
            "all_steps_and_repeats_present": expected_pairs <= observed_pairs,
            "all_finite": all(bool(metric.get("candidate_finite")) and bool(metric.get("reference_finite")) for metric in metrics),
            "exact_record_count": sum(bool(metric.get("exact")) for metric in metrics),
            "nonexact_record_count": sum(not bool(metric.get("exact")) for metric in metrics),
            "positive_signed_mean_count": sum(value > 0.0 for value in signed),
            "negative_signed_mean_count": sum(value < 0.0 for value in signed),
            "max_abs_max": max([float(metric.get("max_abs", 0.0)) for metric in metrics] or [0.0]),
            "rms_max": max([float(metric.get("rms", 0.0)) for metric in metrics] or [0.0]),
            "state_repeat_metrics": values,
        })

    expected_region_ids = {str(item["region_id"]) for item in rows}
    output = {
        "schema": "kernel-analyzer-dtype-semantic-matrix-v1",
        "subject": "Qwen3-1.7B exact semantic mappings across natural checkpoints",
        "seq_len": args.seq_len,
        "dtype": mapping["dtype"],
        "tf32": bool(mapping.get("tf32", False)),
        "checkpoint_steps": steps,
        "checkpoint_count": len(artifacts),
        "dtype_mapping": str(args.dtype_mapping),
        "dtype_mapping_sha256": digest(args.dtype_mapping),
        "denominator": {
            "runtime_symbols": mapping["denominator"]["runtime_symbols"],
            "runtime_invocations": mapping["denominator"]["runtime_invocations"],
            "mapped_symbols": mapping["denominator"]["mapped_symbols"],
            "mapped_invocations": mapping["denominator"]["mapped_invocations"],
            "unresolved_symbols": mapping["denominator"]["unresolved_symbols"],
            "unresolved_invocations": mapping["denominator"]["unresolved_invocations"],
            "mapped_regions_observed_at_all_steps_and_repeats": sum(bool(row["all_steps_and_repeats_present"]) for row in rows),
        },
        "gates": {
            "all_worker_gates_pass": all(all(bool(raw["gates"][key]) for key in (
                "all_expected_ordinary_regions_observed_twice",
                "all_changed_region_ids_retained_twice",
                "all_observation_repeats_stable",
            )) for _, raw in artifacts),
            "all_mapped_regions_present_at_all_steps_and_repeats": len(rows) == expected_regions and all(bool(row["all_steps_and_repeats_present"]) for row in rows),
            "candidate_values_used_to_select_or_classify": False,
            "unresolved_mapping_retained_in_denominator": mapping["denominator"]["unresolved_invocations"] > 0,
        },
        "mapping_boundary": mapping["boundary"],
        "artifacts": [
            {"path": str(path), "sha256": digest(path), "checkpoint_step": int(raw["checkpoint_step"]), "warmed_symbol_count": raw["warmed_symbol_count"]}
            for path, raw in artifacts
        ],
        "rows": rows,
        "boundary": "Only exact pointer-topology semantic mappings are measured. Unresolved generated symbols remain outside attribution but inside the denominator; no natural bias claim is made from this table alone.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "steps": steps, "mapped_regions": len(rows), "complete": output["gates"]["all_mapped_regions_present_at_all_steps_and_repeats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
