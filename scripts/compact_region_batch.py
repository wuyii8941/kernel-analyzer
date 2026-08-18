#!/usr/bin/env python3
"""Compact the multi-region intervention pilot without retaining tensor data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


THRESHOLD = 0.05


def _num(x: Any) -> Any:
    return float(x) if isinstance(x, (int, float)) else x


def _repeat_view(rep: dict[str, Any]) -> dict[str, Any]:
    carrier = next(iter(rep.get("carrier", {}).values()), {})
    sketch = next(iter(rep.get("carrier_sketch", {}).values()), {})
    rec = rep.get("record", {})
    endpoints = {
        str(name): {
            k: _num(value)
            for k, value in metrics.items()
            if k in {"exact", "candidate_finite", "reference_finite", "rms", "max_abs", "signed_mean", "nonzero_elements", "nonzero_fraction"}
        }
        for name, metrics in rec.get("endpoint_metrics", {}).items()
    }
    return {
        "repeat_id": int(rep.get("repeat_id", -1)),
        "loss": _num(rep.get("loss")),
        "endpoint_metrics": endpoints,
        "intervened_endpoints": list(rec.get("intervened_endpoints", [])),
        "carrier": {
            "l2": _num(carrier.get("l2")),
            "rms": _num(carrier.get("rms")),
            "max_abs": _num(carrier.get("max_abs")),
            "delta_baseline_cosine": _num(carrier.get("delta_baseline_cosine")),
            "nonzero": int(carrier.get("nonzero", 0)),
            "status": carrier.get("status"),
        },
        "fixed_projection": {
            "pilot_step": sketch.get("pilot_step"),
            "pilot_cosine": _num(sketch.get("pilot_cosine")),
            "pilot_dot": _num(sketch.get("pilot_dot")),
            "sample_size": int(sketch.get("sample_size", 0)),
            "status": sketch.get("status"),
        },
        "gates": dict(rep.get("gates", {})),
    }


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # Ignore repeat id and loss/elapsed bookkeeping; the intervention metrics
    # must be bitwise-identical at the compact JSON precision.
    aa, bb = dict(a), dict(b)
    aa.pop("repeat_id", None)
    bb.pop("repeat_id", None)
    return aa == bb


def compact(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(p.read_text()) for p in paths]
    rows.sort(key=lambda d: int(d["checkpoint_step"]))
    if not rows:
        raise ValueError("no rows")
    first = rows[0]
    arm_ids = [a["region_id"] for a in first["arms"]]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("duplicate arm ids")
    checkpoints: list[dict[str, Any]] = []
    repeat_checks: list[bool] = []
    for data in rows:
        if data.get("candidate_blind") is not True:
            raise ValueError("candidate-blind gate failed")
        ids = [a["region_id"] for a in data["arms"]]
        if ids != arm_ids:
            raise ValueError("arm set changed across checkpoints")
        gates = data.get("gates", {})
        required = ("all_mapped_regions_census_complete_per_repeat", "every_arm_has_two_repeats", "every_arm_observed_at_exact_boundary")
        if not all(gates.get(k) is True for k in required) or gates.get("natural_bias_case_added") is not False:
            raise ValueError(f"batch gate failed at step {data.get('checkpoint_step')}")
        arm_rows = []
        for arm in data["arms"]:
            reps = arm.get("repeats", [])
            if len(reps) != 2:
                raise ValueError("each arm must have two repeats")
            compact_reps = [_repeat_view(r) for r in reps]
            identical = _same(compact_reps[0], compact_reps[1])
            repeat_checks.append(identical)
            arm_rows.append({
                "arm_index": int(arm["arm_index"]),
                "region_id": arm["region_id"],
                "phase": reps[0].get("record", {}).get("phase"),
                "symbol": reps[0].get("record", {}).get("symbol"),
                "repeat_metrics": compact_reps[0],
                "all_repeats_match": identical,
            })
        checkpoints.append({"step": int(data["checkpoint_step"]), "arms": arm_rows})

    by_arm: dict[str, list[float | None]] = {rid: [] for rid in arm_ids}
    for ckpt in checkpoints:
        for arm in ckpt["arms"]:
            by_arm[arm["region_id"]].append(arm["repeat_metrics"]["fixed_projection"].get("pilot_cosine"))
    arm_summary = []
    for idx, rid in enumerate(arm_ids):
        vals = by_arm[rid]
        signs = ["positive" if v is not None and v >= THRESHOLD else "negative" if v is not None and v <= -THRESHOLD else "near_zero" for v in vals]
        arm_summary.append({
            "arm_index": idx,
            "region_id": rid,
            "projection_values": vals,
            "projection_signs": signs,
            "positive_count": sum(s == "positive" for s in signs),
            "negative_count": sum(s == "negative" for s in signs),
            "near_zero_count": sum(s == "near_zero" for s in signs),
            "persistent_direction": (len(set(s for s in signs if s != "near_zero")) == 1 and "near_zero" not in signs and bool(signs)),
        })
    payload: dict[str, Any] = {
        "schema": "kernel-analyzer-evolving-region-intervention-batch-v1",
        "subject": "Qwen3-1.7B full-step strict-FP32 seq128",
        "candidate_blind": True,
        "intervention": "exact region output replacement with candidate-independent reference output",
        "carrier": {"parameter": "model.embed_tokens.weight", "sketch": "fixed coordinate sample", "threshold": THRESHOLD},
        "checkpoint_steps": [c["step"] for c in checkpoints],
        "arm_count": len(arm_ids),
        "arms": arm_summary,
        "checkpoints": checkpoints,
        "gates": {
            "all_checkpoints_present": [c["step"] for c in checkpoints] == [0, 1, 2, 4, 8, 16, 32, 64],
            "all_repeats_match": all(repeat_checks),
            "fixed_sketch_candidate_blind": True,
            "natural_bias_case_added": False,
            "property_claim": False,
        },
        "interpretation": "screening evidence only; no natural Flash-style bias case is certified",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["result_sha256"] = digest
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    paths = sorted(Path("/").glob(args.input_glob.lstrip("/")))
    out = compact(paths)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": args.output, "checkpoints": out["checkpoint_steps"], "arms": out["arm_count"]}))


if __name__ == "__main__":
    main()
