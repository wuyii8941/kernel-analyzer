#!/usr/bin/env python
"""Audit the first valid candidate-preserving Qwen fused-kernel repairs."""

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
        "result_status_valid": result["status"] == "VALID_ORIGINAL_CANDIDATE_KERNEL_REPAIR",
        "all_result_gates_true": all(result["gates"].values()),
        "selected_calls_exact": result["selected_call_indices"] == [0, 13, 26],
        "all_effects_nonzero": all(row["effect"]["l2"] > 0 for row in result["repairs"].values()),
        "each_run_27_calls_one_repair": all(
            len([entry for entry in record.values() if entry["calls"] > 0]) == 1
            and next(entry for entry in record.values() if entry["calls"] > 0) == {"calls": 27, "repairs": 1}
            for row in result["repairs"].values()
            for record in row["call_records"]
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-candidate-kernel15-repair-audit.v0.4",
        "status": "VALID_ORIGINAL_CANDIDATE_KERNEL_REPAIR_AUDIT" if passed else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifacts,
        "coverage_credit": {
            "original_candidate_generated_kernel_invocations": 3,
            "generated_kernel_family_calls_total": 27,
            "constituent_operator_invocations": 0
        },
        "claim": "three selected fused-kernel invocation repairs causally change the selected-state scorer observable",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
