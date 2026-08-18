#!/usr/bin/env python3
"""Reduce a metadata-only dtype topology census to stable signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    signatures: dict[str, dict[str, object]] = {}
    for row in source["rows"]:
        symbol = str(row["symbol"])
        signature = signatures.setdefault(
            symbol,
            {
                "invocations": 0,
                "pointer_names": row["pointer_names"],
                "pointer_signatures": [],
            },
        )
        signature["invocations"] = int(signature["invocations"]) + 1
        pointer_signature = {
            name: {
                key: value
                for key, value in metadata.items()
                if key in {"dtype", "shape", "stride", "numel", "kind"}
            }
            for name, metadata in row["pointer_metadata"].items()
        }
        signatures[symbol]["pointer_signatures"].append(pointer_signature)
    for signature in signatures.values():
        unique = {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in signature["pointer_signatures"]
        }
        signature["unique_pointer_signatures"] = len(unique)
        del signature["pointer_signatures"]
    payload = {
        "schema": "kernel-analyzer-dtype-generated-topology-compact-v1",
        "subject": source["subject"],
        "dtype": source["dtype"],
        "tf32": source["tf32"],
        "seq_len": source["seq_len"],
        "checkpoint_step": source["checkpoint_step"],
        "checkpoint_parameter_sha256": source["checkpoint_parameter_sha256"],
        "warmed_symbol_count": source["warmed_symbol_count"],
        "runtime_symbol_count": source["runtime_symbol_count"],
        "runtime_invocation_count": source["runtime_invocation_count"],
        "symbol_signatures": signatures,
        "candidate_values_used_to_select_or_classify": source["candidate_values_used_to_select_or_classify"],
        "boundary": source["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    print(json.dumps({"output": str(args.output), "symbols": len(signatures), "invocations": source["runtime_invocation_count"], "result_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
