#!/usr/bin/env python3
"""Independently rebuild and compare the bias-proof plan audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_bias_proof_audit import ROOT, build_payload
except ModuleNotFoundError:  # Direct execution places scripts/, not the repository root, on sys.path.
    from build_bias_proof_audit import ROOT, build_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    expected = build_payload()
    errors = []
    if result != expected:
        errors.append("stored result differs from recomputation")
    if result.get("status") != "COMPLETE" or not all(result.get("gates", {}).values()):
        errors.append("one or more declared completion gates are not satisfied")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("choose a new output path inside kernel-analyzer")
    payload = {
        "schema": "bias-proof-plan-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "verified_gate_count": sum(bool(value) for value in result.get("gates", {}).values()),
        "scope": "recomputation of saved sufficient statistics and declared gates; not independent GPU reproduction",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
