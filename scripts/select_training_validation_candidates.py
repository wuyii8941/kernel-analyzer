#!/usr/bin/env python3
"""Recompute the fail-closed shortlist for human training review."""

import argparse
import hashlib
import json
from pathlib import Path

from kernel_analyzer.training_candidate_selection import select_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.manifest.read_bytes()
    result = select_candidates(json.loads(raw))
    result["manifest"] = str(args.manifest.resolve())
    result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    result["selector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"eligible_cases": result["eligible_cases"],
                      "total_cases": len(result["rows"])}))


if __name__ == "__main__":
    main()
