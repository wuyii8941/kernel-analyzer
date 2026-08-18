#!/usr/bin/env python3
"""Merge disjoint evolving-checkpoint result extensions after digest checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("matrix", "carrier"), required=True)
    args = parser.parse_args()
    base, extension, bank = read(args.base), read(args.extension), read(args.bank)
    bank_by_step = {row["step"]: row for row in bank["checkpoints"]}
    base_by_step = {row["checkpoint_step"]: row for row in base["rows"]}
    ext_by_step = {row["checkpoint_step"]: row for row in extension["rows"]}
    for step, row in {**base_by_step, **ext_by_step}.items():
        expected = bank_by_step[step]["parameter_sha256"]
        if row["checkpoint_parameter_sha256"] != expected:
            raise ValueError(f"checkpoint digest mismatch at step {step}")
    if args.kind == "carrier" and 0 in base_by_step and 0 in ext_by_step:
        # The extension reruns the pilot. Keep the base row but require its
        # numerical pilot values to agree exactly for each candidate.
        if base_by_step[0]["variants"] != ext_by_step[0]["variants"]:
            raise ValueError("step-0 carrier pilot differs between runs")
    rows = [base_by_step[step] for step in sorted(base_by_step)]
    rows_by_step = {row["checkpoint_step"]: row for row in rows}
    for step, row in ext_by_step.items():
        rows_by_step[step] = row
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    result = dict(extension)
    result["rows"] = rows
    result["bank_manifest"] = str(args.bank)
    result["bank_protocol_sha256"] = bank["protocol_sha256"]
    if args.kind == "matrix":
        result["variants"] = {
            "eager": {"role": "reference", "changed": False},
            "sdpa_math": {"role": "candidate", "changed": True, "mechanisms": ["backend", "reduction_schedule", "materialization"]},
            "sdpa_flash": {"role": "candidate", "changed": True, "mechanisms": ["backend", "online_reduction", "layout", "materialization"], "accumulator": "not directly observable from this interface"},
        }
    else:
        result["variants"] = {
            "sdpa_math": {"changed": True, "mechanisms": ["backend", "reduction_schedule", "materialization"]},
            "sdpa_flash": {"changed": True, "mechanisms": ["backend", "online_reduction", "layout", "materialization"], "accumulator": "not directly observable from this interface"},
        }
    if args.kind == "carrier":
        summaries = {}
        variants = sorted({variant for row in rows for variant in row["variants"]})
        for variant in variants:
            valid = [row["variants"].get(variant, {}) for row in rows]
            valid = [value for value in valid if value.get("status") == "OK"]
            heldout = [value for row in rows if row["checkpoint_step"] != 0 for value in [row["variants"].get(variant, {})] if value.get("status") == "OK"]
            summaries[variant] = {
                "valid_states": len(valid),
                "heldout_states": len(heldout),
                "heldout_positive": sum(value["carrier_positive"] for value in heldout),
                "heldout_positive_fraction": sum(value["carrier_positive"] for value in heldout) / len(heldout) if heldout else None,
                "heldout_mean_projection": sum(value["carrier_projection"] for value in heldout) / len(heldout) if heldout else None,
                "heldout_min_projection": min((value["carrier_projection"] for value in heldout), default=None),
                "heldout_mean_cosine": sum(value["carrier_cosine"] for value in heldout) / len(heldout) if heldout else None,
            }
        result["summary"] = summaries
    result.pop("result_sha256", None)
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "kind": args.kind}, sort_keys=True))


if __name__ == "__main__":
    main()
