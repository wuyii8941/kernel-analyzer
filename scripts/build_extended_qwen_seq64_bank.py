#!/usr/bin/env python3
"""Build a deterministic 64-state Qwen seq64 bank for unresolved follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    extension = json.loads(args.extension.read_text())
    base_states = base["states"]
    extra = extension["records"][: max(0, 64 - len(base_states))]
    if len(base_states) + len(extra) != 64:
        raise SystemExit("base plus extension must provide 64 states")
    states = []
    for row in base_states:
        states.append({
            "state_id": str(row.get("state_id", row.get("cluster_id"))),
            "token_ids": list(row.get("token_ids", row.get("input_ids"))),
            "role": "BASE_HELDOUT",
        })
    for row in extra:
        states.append({
            "state_id": str(row.get("sequence_id", row.get("cluster_id"))),
            "token_ids": list(row["input_ids"]),
            "role": "EXTENDED_CONFIRMATION",
        })
    ids = [row["state_id"] for row in states]
    if len(ids) != len(set(ids)):
        raise SystemExit("state IDs are not unique")
    lengths = {len(row["token_ids"]) for row in states}
    if lengths != {64}:
        raise SystemExit(f"unexpected sequence lengths: {sorted(lengths)}")
    payload = {
        "schema": "kernel-analyzer-qwen-extended-input-bank-v1",
        "model": base["model"],
        "sequence_length": 64,
        "state_count": 64,
        "base_bank_sha256": digest(base),
        "extension_design_sha256": extension["design_sha256"],
        "selection": "first 32 frozen seq64 states plus first 32 mechanically ordered qwen145 confirmation records",
        "states": states,
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(states),
                      "result_sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
