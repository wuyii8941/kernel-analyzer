#!/usr/bin/env python3
"""Explain a rejected source binding without selecting a replacement endpoint."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.run_training_numerical_v2 import save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rejection", type=Path, required=True)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rejection = json.loads(args.rejection.read_text())
    with gzip.open(args.proof, "rt") as handle:
        proof = json.load(handle)
    origins = proof["inductor_buffer_origins"]["buffer_origins"]
    rows = []
    for task in rejection["matched_tasks"]:
        phase, symbol, buffer = task["compiler_origin_key"].split("\0")
        observed = [row for row in origins.get(buffer, []) if row["phase"] == phase]
        exact = [row for row in observed if row["kernel_name"] == symbol and row.get("origin_node_exact")]
        rows.append({"task_id": task["task_id"], "expected_symbol": symbol, "buffer": buffer,
            "observed_same_name": [{key: row.get(key) for key in
                ("phase", "kernel_name", "exact_origin_node", "origin_node_exact")} for row in observed],
            "diagnosis": "EXACT_ORIGIN_PRESENT_REQUIRES_ADAPTER_REVIEW" if exact else
                "BUFFER_NAME_REUSED_BY_DIFFERENT_COMPILED_KERNEL" if observed and all(row["kernel_name"] != symbol for row in observed)
                else "ORIGINAL_BUFFER_ORIGIN_NOT_EXACT",
            "automatic_replacement_allowed": False})
    save_new(args.output, {"status": "DIAGNOSED_NOT_REPAIRED", "rows": rows,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.rejection, args.proof)},
        "claim_boundary": "Names identify buffers only within their own compiled artifact. This diagnosis neither proves semantic correspondence nor authorizes a larger repair boundary."})


if __name__ == "__main__":
    main()
