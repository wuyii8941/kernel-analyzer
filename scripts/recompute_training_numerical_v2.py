#!/usr/bin/env python3
"""Recompute v2 from original statistics, without reading cached verdicts."""
import argparse
import hashlib
import json
from pathlib import Path
from kernel_analyzer.training_numerical_analysis import analyze_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing result; choose a new output.")
    result = analyze_artifact(json.loads(args.raw.read_text()), json.loads(args.protocol.read_text()))
    result["provenance"].update({
        "raw_artifact": str(args.raw),
        "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
