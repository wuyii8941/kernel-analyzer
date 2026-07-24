#!/usr/bin/env python
"""Audit the fail-closed Qwen3 causal-mask treatment result."""

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
    eager_types = set(result["target_output_types"]["eager"])
    compiled_types = set(result["target_output_types"]["compiled"])
    gates = {
        "manifest_frozen": manifest["status"] == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifacts.values()),
        "invalid_treatment_status_exact": result["status"] == "INVALID_TREATMENT",
        "eager_anchor_and_repeats_valid": result["gates"]["instrumented_eager_anchor_exact"] and result["gates"]["all_repeats_exact"],
        "target_modes_executed": result["gates"]["target_modes_executed_exactly"],
        "outer_recompile_gate_failed": result["gates"]["no_outer_recompile_across_mask_modes"] is False,
        "output_representation_changed": eager_types == {"None"} and compiled_types == {"Tensor"},
        "coverage_credit_invalid": result["coverage_credit"] == "INVALID_TREATMENT",
        "transport_refused": result["transport_to_original_candidate"] == "NOT_ESTABLISHED",
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-causal-mask-barrier-audit.v0.1",
        "status": "VALID_INVALID_TREATMENT_AUDIT" if passed else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifacts,
        "coverage_credit": 0,
        "interpretation": "eager None and compiled Tensor outputs changed the continuation graph, confounding repair attribution",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
