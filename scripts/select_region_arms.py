#!/usr/bin/env python3
"""Select candidate-blind exact-mapped region arms from semantic observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.semantic.read_text())
    candidates = []
    for row in data["rows"]:
        pilot = [
            item
            for item in row["state_repeat_metrics"]
            if int(item["step"]) == 0 and not bool(item["metric"].get("exact"))
        ]
        if not pilot:
            continue
        best = max(pilot, key=lambda item: float(item["metric"].get("rms", 0.0)))
        candidates.append({
            "region_id": row["region_id"],
            "symbol": row["symbol"],
            "phase": row["phase"],
            "pilot_rms": float(best["metric"].get("rms", 0.0)),
            "pilot_signed_mean": float(best["metric"].get("signed_mean", 0.0)),
        })
    candidates.sort(key=lambda row: row["pilot_rms"], reverse=True)
    selected = []
    seen_symbols = set()
    for row in candidates:
        if row["symbol"] in seen_symbols:
            continue
        selected.append(row)
        seen_symbols.add(row["symbol"])
        if len(selected) == args.limit:
            break
    if len(selected) < args.limit:
        selected_ids = {row["region_id"] for row in selected}
        for row in candidates:
            if row["region_id"] in selected_ids:
                continue
            selected.append(row)
            if len(selected) == args.limit:
                break
    if not selected:
        raise RuntimeError("semantic artifact has no nonexact step-0 region")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
