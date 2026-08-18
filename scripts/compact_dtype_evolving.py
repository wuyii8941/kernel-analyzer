#!/usr/bin/env python3
"""Compact an expanded dtype-specific endpoint campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _metric(m: dict[str, Any]) -> dict[str, Any]:
    keep = {"exact", "candidate_finite", "reference_finite", "rms", "max_abs", "signed_mean", "nonzero_elements", "nonzero_fraction"}
    return {k: m[k] for k in keep if k in m}


def _record_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": record["region_id"],
        "phase": record["phase"],
        "symbol": record["symbol"],
        "reference_symbol": record.get("reference_symbol"),
        "reference_role": record.get("reference_role"),
        "endpoint_metrics": {k: _metric(v) for k, v in record.get("endpoint_metrics", {}).items()},
        "semantic_closure_present": record.get("semantic_closure") is not None,
    }


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


def compact(paths: list[Path]) -> dict[str, Any]:
    data = sorted((json.loads(p.read_text()) for p in paths), key=lambda d: int(d["checkpoint_step"]))
    if [int(d["checkpoint_step"]) for d in data] != [0, 1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("expanded endpoint campaign must contain the full natural bank")
    first_ids = [r["region_id"] for r in data[0]["repeats"][0]["records"]]
    if len(first_ids) != len(set(first_ids)):
        raise ValueError("mapped invocation rows must have unique region ids")
    checkpoints = []
    repeat_gates = []
    by_region: dict[str, list[dict[str, Any]]] = {rid: [] for rid in first_ids}
    for d in data:
        if not all(d["gates"].get(k) is True for k in ("all_expected_ordinary_regions_observed_twice", "all_changed_region_ids_retained_twice")) or d["gates"].get("candidate_values_used_to_select_regions") is not False:
            raise ValueError(f"observation gate failed at step {d['checkpoint_step']}")
        reps = d["repeats"]
        if len(reps) != 2:
            raise ValueError("each checkpoint must have two repeats")
        views = [_record_view(r) for r in reps[0]["records"]]
        views2 = [_record_view(r) for r in reps[1]["records"]]
        if [r["region_id"] for r in views] != first_ids or [r["region_id"] for r in views2] != first_ids:
            raise ValueError("mapped invocation set changed across checkpoints")
        repeat_match = all(_same(a, b) for a, b in zip(views, views2))
        repeat_gates.append(repeat_match)
        for row in views:
            by_region[row["region_id"]].append({
                "step": int(d["checkpoint_step"]),
                "symbol": row["symbol"],
                "reference_symbol": row["reference_symbol"],
                "metrics": row["endpoint_metrics"],
                "reference_role": row["reference_role"],
            })
        checkpoints.append({"step": int(d["checkpoint_step"]), "repeat_match": repeat_match, "observed_rows": len(views), "observer_status": reps[0]["triton_summary"]["status"]})
    rows = []
    persistent = 0
    for rid in first_ids:
        samples = by_region[rid]
        endpoints = sorted({name for s in samples for name in s["metrics"]})
        endpoint_summary = []
        for endpoint in endpoints:
            values = [float(s["metrics"][endpoint].get("signed_mean", 0.0)) for s in samples if endpoint in s["metrics"]]
            positive = sum(v > 0 for v in values)
            negative = sum(v < 0 for v in values)
            sign_persistent = bool(values) and (positive == len(values) or negative == len(values))
            persistent += int(sign_persistent)
            endpoint_summary.append({"endpoint": endpoint, "signed_mean": values, "positive": positive, "negative": negative, "sign_persistent": sign_persistent})
        rows.append({"region_id": rid, "phase": samples[0]["metrics"] and next(r["phase"] for r in data[0]["repeats"][0]["records"] if r["region_id"] == rid), "samples": samples, "endpoint_summary": endpoint_summary})
    mapping = json.loads(Path(args_mapping).read_text()) if args_mapping else None
    out: dict[str, Any] = {
        "schema": "kernel-analyzer-dtype-evolving-endpoints-v1",
        "subject": args_subject or "Qwen3-1.7B strict-FP32 evolving mapped endpoint campaign",
        "dtype": "fp32",
        "tf32": False,
        "candidate_blind": True,
        "checkpoint_steps": [c["step"] for c in checkpoints],
        "mapped_invocations": len(first_ids),
        "mapped_symbols": mapping["denominator"]["mapped_symbols"] if mapping else None,
        "unresolved_symbols": mapping["denominator"]["unresolved_symbols"] if mapping else None,
        "unresolved_invocations": mapping["denominator"]["unresolved_invocations"] if mapping else None,
        "checkpoints": checkpoints,
        "rows": rows,
        "gates": {
            "all_mapped_invocations_observed_at_all_checkpoints_and_repeats": True,
            "all_repeats_match": all(repeat_gates),
            "all_tensor_values_retained": False,
            "f_b_closure_complete": False,
            "natural_bias_case_added": False,
            "property_claim": False,
        },
        "interpretation": "complete endpoint observation for the mapped subset; unresolved symbols and unclosed internal reductions remain explicit boundaries",
    }
    out["result_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()
    global args_mapping
    global args_subject
    args_mapping = args.mapping
    args_subject = args.subject
    paths = sorted(Path("/").glob(args.input_glob.lstrip("/")))
    out = compact(paths)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": args.output, "checkpoints": out["checkpoint_steps"], "mapped_invocations": out["mapped_invocations"]}))


if __name__ == "__main__":
    args_mapping = None
    args_subject = None
    main()
