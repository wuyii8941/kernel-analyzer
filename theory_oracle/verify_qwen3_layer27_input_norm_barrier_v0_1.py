#!/usr/bin/env python
"""Fail-closed verifier for the layer-27 input-RMSNorm barrier experiment."""

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
    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text())

    artifact_checks = {}
    for name, record in manifest["artifacts"].items():
        observed = digest(root / record["path"])
        artifact_checks[name] = {
            "expected": record["sha256"],
            "observed": observed,
            "pass": observed == record["sha256"],
        }
    gates = {
        "manifest_frozen": manifest["status"] == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "result_status_valid": result["status"] == "VALID_BARRIER_CONDITIONED_OPERATOR_EFFECT",
        "all_result_gates_true": all(result["gates"].values()),
        "original_candidate_transport_refused": (
            result["candidate_anchor_exact"] is False
            and result["transport_to_original_candidate"] == "NOT_ESTABLISHED"
        ),
        "injection_nonzero": result["contrasts"]["injection_Eb_to_EbI"]["l2"] > 0,
        "repair_nonzero": result["contrasts"]["repair_Cb_to_CbR"]["l2"] > 0,
        "target_modes_executed_exactly": all(
            sum(row.values()) == 2 for row in result["target_call_deltas"].values()
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-layer27-input-norm-barrier-audit.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_EVIDENCE" if passed else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifact_checks,
        "result": str(result_path),
        "coverage_credit": "BARRIER_CONDITIONED",
        "original_candidate_root_cause_credit": False,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
