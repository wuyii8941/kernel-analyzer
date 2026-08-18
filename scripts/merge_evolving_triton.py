#!/usr/bin/env python3
"""Merge compact evolving Triton checkpoint observations.

The worker artifacts contain only changed-site metrics, while this merger
checks that every ordinary generated region was observed and that every
changed site was retained twice at every checkpoint.  No mathematical
derivation is recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, choices=(64, 128, 256), required=True)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--implementation-atlas", type=Path, default=Path("results/final/implementation_atlas.json"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    atlas = json.loads(args.implementation_atlas.read_text())
    allowed = {
        "EXPLICIT_FP32_REDUCTION_SCHEDULE_DIFFERENCE",
        "MATERIALIZATION_OR_ROUNDING_SCHEDULE_INTERVENTION",
        "SAME_PRECISION_GENERATED_SCHEDULE_DIFFERENCE",
    }
    atlas_rows = {
        str(row["region"]): row
        for row in atlas["rows"]
        if bool(row.get("implementation_changed")) and str(row.get("mechanism")) in allowed
    }
    if not atlas_rows:
        raise RuntimeError("implementation atlas contains no changed rows")
    artifacts = []
    for path in sorted(args.inputs):
        raw = json.loads(path.read_text())
        if int(raw["seq_len"]) != args.seq_len:
            raise RuntimeError(f"shape mismatch in {path}")
        if not raw["gates"]["all_expected_ordinary_regions_observed_twice"]:
            raise RuntimeError(f"ordinary region gate failed in {path}")
        if not raw["gates"]["all_changed_region_ids_retained_twice"]:
            raise RuntimeError(f"changed-site gate failed in {path}")
        if not raw["gates"]["all_observation_repeats_stable"]:
            raise RuntimeError(f"observation stability gate failed in {path}")
        artifacts.append((path, raw))
    observed_steps = [int(raw["checkpoint_step"]) for _, raw in artifacts]
    if len(set(observed_steps)) != len(observed_steps):
        raise RuntimeError("duplicate checkpoint step")
    dtypes = {str(raw.get("dtype", "bf16")) for _, raw in artifacts}
    tf32_modes = {bool(raw.get("tf32", False)) for _, raw in artifacts}
    if len(dtypes) != 1 or len(tf32_modes) != 1:
        raise RuntimeError("mixed dtype or TF32 mode in one evolving merge")
    artifacts.sort(key=lambda item: int(item[1]["checkpoint_step"]))
    observed_steps = [int(raw["checkpoint_step"]) for _, raw in artifacts]

    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, raw in artifacts:
        for repeat in raw["repeats"]:
            for record in repeat["records"]:
                rid = str(record["region_id"])
                if rid not in atlas_rows:
                    continue
                for endpoint, metric in record["endpoint_metrics"].items():
                    by_site[rid].append({
                        "step": int(raw["checkpoint_step"]),
                        "repeat_id": int(repeat["repeat_id"]),
                        "phase": record["phase"],
                        "symbol": record["symbol"],
                        "reference_symbol": record.get("reference_symbol"),
                        "endpoint": endpoint,
                        "metric": metric,
                    })

    site_rows = []
    for rid, meta in sorted(atlas_rows.items()):
        values = by_site.get(rid, [])
        steps = sorted({int(row["step"]) for row in values})
        repeats = {(int(row["step"]), int(row["repeat_id"])) for row in values}
        expected_repeats = {(int(raw["checkpoint_step"]), repeat_id) for _, raw in artifacts for repeat_id in (0, 1)}
        metrics = [row["metric"] for row in values]
        signed = [float(metric.get("signed_mean", 0.0)) for metric in metrics]
        rms = [float(metric.get("rms", 0.0)) for metric in metrics]
        site_rows.append({
            "region_id": rid,
            "phase": meta["phase"],
            "kind": meta["kind"],
            "symbol": meta["symbol"],
            "mechanism": meta["mechanism"],
            "steps_observed": steps,
            "endpoint_count": len({str(row["endpoint"]) for row in values}),
            "record_count": len(values),
            "expected_record_count": len(expected_repeats),
            "all_steps_and_repeats_present": expected_repeats <= repeats,
            "all_finite": all(bool(metric.get("candidate_finite")) and bool(metric.get("reference_finite")) for metric in metrics),
            "exact_record_count": sum(bool(metric.get("exact")) for metric in metrics),
            "nonexact_record_count": sum(not bool(metric.get("exact")) for metric in metrics),
            "positive_signed_mean_count": sum(value > 0.0 for value in signed),
            "negative_signed_mean_count": sum(value < 0.0 for value in signed),
            "max_abs_max": max([float(metric.get("max_abs", 0.0)) for metric in metrics] or [0.0]),
            "rms_max": max(rms or [0.0]),
            "state_repeat_metrics": values,
        })

    expected_site_ids = set(atlas_rows)
    observed_site_ids = {row["region_id"] for row in site_rows if row["all_steps_and_repeats_present"]}
    output = {
        "schema": "kernel-analyzer-evolving-triton-matrix-v1",
        "subject": "Qwen3-1.7B changed generated Triton regions on natural checkpoints",
        "seq_len": args.seq_len,
        "dtype": next(iter(dtypes)),
        "tf32": next(iter(tf32_modes)),
        "checkpoint_steps": observed_steps,
        "checkpoint_count": len(artifacts),
        "implementation_atlas": str(args.implementation_atlas),
        "implementation_atlas_sha256": digest(args.implementation_atlas),
        "denominator": {
            "changed_generated_region_sites": len(expected_site_ids),
            "changed_sites_observed_at_all_steps_and_repeats": len(observed_site_ids),
            "ordinary_region_observation_artifacts": len(artifacts),
        },
        "gates": {
            "all_worker_gates_pass": all(raw["gates"]["all_changed_region_ids_retained_twice"] for _, raw in artifacts),
            "all_changed_sites_present_at_all_steps_and_repeats": observed_site_ids == expected_site_ids,
            "all_worker_observations_stable": all(raw["gates"]["all_observation_repeats_stable"] for _, raw in artifacts),
            "candidate_values_used_to_select_regions": False,
        },
        "artifacts": [
            {"path": str(path), "sha256": digest(path), "checkpoint_step": int(raw["checkpoint_step"]), "warmed_symbol_count": raw["warmed_symbol_count"], "unmatched_warmed_symbols": raw["unmatched_warmed_symbols"]}
            for path, raw in artifacts
        ],
        "rows": site_rows,
        "boundary": "Changed generated Triton sites are directly observed on every listed natural checkpoint with same-input reference metrics. Unmatched newly generated symbols, external exact replays, and other dtype/backend inventories remain separate boundaries.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "steps": observed_steps, "changed_sites": len(expected_site_ids), "complete": output["gates"]["all_changed_sites_present_at_all_steps_and_repeats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
