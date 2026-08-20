#!/usr/bin/env python3
"""Convert an execution-derived F+B inventory into the TCMP denominator ledger."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def read(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = read(args.inventory)
    events = source.get("trace", {}).get("events", [])
    if not events:
        raise RuntimeError("inventory has no executed invocation denominator")
    rows = []
    for event in events:
        local_id = str(event["invocation_id"])
        invocation_id = f"{args.cell_id}::{local_id}"
        rows.append({
            "invocation_id": invocation_id,
            "cell_id": args.cell_id,
            "source_invocation_id": local_id,
            "phase": event["phase"],
            "overload": event["overload"],
            "category": event["category"],
            "sequence_binding_status": event.get("sequence_binding_status"),
            "enclosing_dispatch_invocation_id": event.get("enclosing_dispatch_invocation_id"),
            "proof_unit_id": None,
            "causal_credit_unit_id": None,
            "proof_status": "PENDING_NEW_MODEL_FB_BINDING",
            "capability": "UNRESOLVED_BOUNDARY",
            "disposition": "UNRESOLVED_PROOF",
        })
    payload = {
        "schema": "kernel-analyzer-tcmp-invocation-ledger-v1",
        "status": "COMPLETE_INVOCATION_DENOMINATOR_PROOF_PENDING",
        "cell_id": args.cell_id,
        "source_inventory": str(args.inventory.resolve()),
        "invocation_ids": [row["invocation_id"] for row in rows],
        "rows": rows,
        "counts": {
            "invocations": len(rows),
            "forward": sum(row["phase"] == "FORWARD" for row in rows),
            "backward": sum(row["phase"] == "BACKWARD" for row in rows),
            "unique_overloads": len({row["overload"] for row in rows}),
        },
        "claim_boundary": (
            "Every executed invocation is present. Proof, orbit capability, and causal "
            "credit remain fail-closed until the new-model F+B bridge is built."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({"output": str(args.output), **payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
