#!/usr/bin/env python
"""Audit a valid fail-closed invalidation of SDPA decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    result = json.loads(Path(args.result).read_text())
    artifacts = {}
    for name, record in manifest["artifacts"].items():
        observed = digest(root / record["path"])
        artifacts[name] = {"expected": record["sha256"], "observed": observed, "pass": observed == record["sha256"]}
    gates = {
        "manifest_frozen_revision": manifest["status"].startswith("FROZEN_PRE_EXECUTION"),
        "artifact_hashes_exact": all(row["pass"] for row in artifacts.values()),
        "invalidation_status_exact": result["status"] == "VALID_INVALIDATION_REFERENCE_RECONSTRUCTION_CHANGED",
        "original_eager_exact_and_stable": result["original_eager"]["repeat_exact"] and len(set(result["original_eager"]["sha256"])) == 1,
        "reconstruction_stable_but_different": result["reconstructed_eager"]["repeat_exact"] and result["reconstructed_eager"]["sha256"][0] != result["original_eager"]["sha256"][0],
        "nonzero_reconstruction_delta": result["reconstruction_delta"]["l2"] > 0,
        "coverage_credit_zero": result["coverage_credit"] == 0,
        "nine_targets_attempted": len(result["targets_attempted"]) == 9,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-sdpa-decomposition-barrier-audit.v0.1",
        "status": "VALID_INVALIDATION_AUDIT" if passed else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifacts,
        "coverage_credit": 0,
        "interpretation": "AOT-visible decomposition was not an exact eager replacement under the frozen protocol",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
