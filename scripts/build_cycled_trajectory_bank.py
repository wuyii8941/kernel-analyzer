#!/usr/bin/env python3
"""Build an explicit repeated-input trajectory bank for a long consequence probe.

This is only used when the original 4096-step natural trajectory bank is absent.
The repeated stream is recorded as a separate protocol, never merged with a
natural trajectory label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4224)
    args = parser.parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_rows = payload.get("states", payload.get("records", []))
    if not source_rows:
        raise ValueError("source bank has no states")
    states = []
    for index in range(args.steps):
        source = source_rows[index % len(source_rows)]
        row = dict(source)
        row["state_id"] = f"cycled-{index:04d}-source-{source.get('state_id', index % len(source_rows))}"
        row["role"] = "TRAJECTORY"
        row["order_within_role"] = index
        states.append(row)
    result = {
        "schema": "kernel-analyzer-cycled-trajectory-bank-v1",
        "source_bank": str(args.source),
        "source_state_count": len(source_rows),
        "steps": args.steps,
        "stream_rule": "cycle the frozen source states in their declared order",
        "claim_boundary": "Synthetic repeated-input stream; results are not interchangeable with a natural training trajectory.",
        "states": states,
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "steps": args.steps, "source_state_count": len(source_rows)}))


if __name__ == "__main__":
    main()
