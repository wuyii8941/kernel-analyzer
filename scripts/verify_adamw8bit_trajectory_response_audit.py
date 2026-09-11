#!/usr/bin/env python3
"""Independently verify the AdamW8bit trajectory-response audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds


CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    protocol = load(root / "protocol.json")
    summary = load(root / "summary.json")
    errors = []
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("frozen source or input changed: " + name)
    rows = []
    for index in range(int(protocol["streams"])):
        path = root / "streams" / f"stream-{index:02d}.json"
        if not path.is_file():
            errors.append(f"missing stream {index}")
            continue
        row = load(path)
        if row.get("status") != "COMPLETE" or row.get("stream_index") != index:
            errors.append(f"invalid stream {index}")
        if set(map(int, row.get("checkpoints", {}))) != set(protocol["checkpoints"]):
            errors.append(f"checkpoint set changed for stream {index}")
        rows.append(row)
    recomputed = None
    if len(rows) == int(protocol["streams"]):
        final_key = str(protocol["checkpoints"][-1])
        parameter_flags = []
        loss_flags = []
        for row in rows:
            final = row["checkpoints"][final_key]
            distances = final["parameter_distance_to_fp32"]
            losses = final["evaluation_loss"]
            parameter_flag = (
                distances["ADAMW8BIT_BLOCK64"]["relative_l2"]
                < distances["ADAMW8BIT_BLOCK256"]["relative_l2"]
            )
            loss_flag = (
                abs(losses["ADAMW8BIT_BLOCK64"] - losses["FP32_ADAMW"])
                < abs(losses["ADAMW8BIT_BLOCK256"] - losses["FP32_ADAMW"])
            )
            if parameter_flag != row["final_block64_parameter_closer"]:
                errors.append(f"parameter flag changed for stream {row['stream_index']}")
            if loss_flag != row["final_block64_loss_closer"]:
                errors.append(f"loss flag changed for stream {row['stream_index']}")
            parameter_flags.append(parameter_flag)
            loss_flags.append(loss_flag)
        successes = sum(parameter_flags)
        bounds = exact_binomial_one_sided_bounds(successes, len(rows), alpha=0.05)
        recomputed = {
            "block64_final_parameter_closer_count": successes,
            "block64_final_parameter_closer_probability_bounds": list(bounds),
            "block64_parameter_proximity_result": (
                "CONFIRMED_MORE_OFTEN_THAN_HALF" if bounds[0] > 0.5 else "NOT_CONFIRMED"
            ),
            "block64_final_loss_closer_count": sum(loss_flags),
            "mean_final_relative_parameter_distance": {
                condition: math.fsum(
                    row["checkpoints"][final_key]["parameter_distance_to_fp32"][condition]["relative_l2"]
                    for row in rows
                ) / len(rows)
                for condition in ("ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
            },
            "mean_final_evaluation_loss": {
                condition: math.fsum(
                    row["checkpoints"][final_key]["evaluation_loss"][condition]
                    for row in rows
                ) / len(rows)
                for condition in CONDITIONS
            },
        }
        for key, value in recomputed.items():
            if summary.get(key) != value:
                errors.append("summary field does not recompute: " + key)
    return {
        "schema": "adamw8bit-trajectory-response-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "stream_count": len(rows),
        "recomputed": recomputed,
        "protocol_sha256": sha(root / "protocol.json"),
        "summary_sha256": sha(root / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("choose a new output under /data1/tzh")
    result = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
