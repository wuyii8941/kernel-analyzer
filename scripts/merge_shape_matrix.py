#!/usr/bin/env python3
"""Merge evolving attention matrices across sequence-length strata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq128", type=Path, default=Path("results/final/checkpoint_matrix.json"))
    parser.add_argument("--seq64", type=Path, default=Path("results/final/checkpoint_matrix_seq64.json"))
    parser.add_argument("--seq256", type=Path, default=Path("results/final/checkpoint_matrix_seq256.json"))
    parser.add_argument("--output", type=Path, default=Path("results/final/checkpoint_matrix_shapes.json"))
    args = parser.parse_args()
    inputs = [(64, args.seq64), (128, args.seq128), (256, args.seq256)]
    merged_rows = []
    shape_summary = {}
    source_digests = {}
    for seq_len, path in inputs:
        data = json.loads(path.read_text())
        source_digests[str(seq_len)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for row in data["rows"]:
            item = dict(row)
            item["seq_len"] = seq_len
            merged_rows.append(item)
            rows.append(item)
        candidate_count = sum(
            1
            for row in rows
            for variant, value in row["variants"].items()
            if variant != "eager" and value.get("status", "OK") == "OK"
        )
        shape_summary[str(seq_len)] = {
            "checkpoint_count": len(rows),
            "reference_regions_per_direction": rows[0]["reference_forward_regions"] if rows else 0,
            "candidate_variant_rows": candidate_count,
            "candidate_forward_backward_region_comparisons": candidate_count * (rows[0]["reference_forward_regions"] if rows else 0) * 2,
            "all_variant_rows_ok": all(
                value.get("status", "OK") == "OK"
                for row in rows
                for value in row["variants"].values()
            ),
            "evaluation": data["evaluation"],
        }
    output = {
        "schema": "kernel-analyzer-evolving-shape-matrix-v1",
        "subject": "Qwen3-1.7B eager/SDPA attention F+B on natural checkpoints",
        "bank_manifest": "results/final/natural_bank.json",
        "bank_protocol_sha256": json.loads(args.seq128.read_text())["bank_protocol_sha256"],
        "shapes": [64, 128, 256],
        "variants": json.loads(args.seq128.read_text())["variants"],
        "source_digests": source_digests,
        "shape_summary": shape_summary,
        "rows": merged_rows,
        "denominator": {
            "shape_strata": 3,
            "checkpoints_per_shape": 8,
            "reference_forward_backward_region_instances": 3 * 8 * 28 * 2,
            "candidate_forward_backward_region_comparisons": sum(
                value["candidate_forward_backward_region_comparisons"]
                for value in shape_summary.values()
            ),
        },
        "boundary": (
            "Rows compare already closed attention F+B semantic regions; no mathematical derivations are "
            "recomputed. This shape matrix does not replace the invocation-level generated-region atlas."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(merged_rows), "denominator": output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
