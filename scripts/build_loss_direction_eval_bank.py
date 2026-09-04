#!/usr/bin/env python3
"""Build a deterministic evaluation set excluding states used by a trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory-protocol", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    protocol = json.loads(args.trajectory_protocol.read_text())
    excluded = set(protocol["trajectory"]["state_order"])
    records = source.get("records", source.get("states", []))
    selected = [
        row for row in records
        if str(row.get("sequence_id", row.get("state_id"))) not in excluded
    ][: args.count]
    if len(selected) != args.count:
        raise RuntimeError("not enough unused evaluation states")
    payload = {
        "schema": "kernel-analyzer-loss-direction-evaluation-bank-v1",
        "status": "FROZEN_WITHOUT_READING_LOSS_DIRECTION_RESULTS",
        "records": selected,
        "excluded_training_state_ids": sorted(excluded),
        "selection": "first source records not used by the paired trajectory",
        "source": str(args.source),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "records": len(selected)}))


if __name__ == "__main__":
    main()
