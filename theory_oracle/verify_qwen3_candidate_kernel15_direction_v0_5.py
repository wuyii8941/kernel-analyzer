#!/usr/bin/env python
"""Fail-closed audit for Qwen3 candidate-kernel-15 direction revision v0.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CALLS = [0, 13, 26]
EXPECTED_STATUS = "VALID_ORIGINAL_CANDIDATE_KERNEL_DIRECTION"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    result_path = Path(args.result).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)

    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    artifact_checks: dict[str, Any] = {}
    for name, record in manifest["artifacts"].items():
        path = root / record["path"]
        observed = digest(path)
        artifact_checks[name] = {
            "expected": record["sha256"],
            "observed": observed,
            "pass": observed == record["sha256"],
        }

    repairs = result.get("repairs", {})
    directional_rows = [repairs.get(str(index), {}).get("direction", {}) for index in EXPECTED_CALLS]
    finite_directions = all(
        all(math.isfinite(float(row[key])) for key in (
            "eager_candidate_l2",
            "eager_repair_l2",
            "l2_distance_change",
            "fractional_l2_reduction",
            "cosine_repair_with_candidate_to_eager",
        ))
        for row in directional_rows
    )
    direction_identities = all(
        math.isclose(
            float(row["l2_distance_change"]),
            float(row["eager_repair_l2"]) - float(row["eager_candidate_l2"]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(row["fractional_l2_reduction"]),
            -float(row["l2_distance_change"]) / float(row["eager_candidate_l2"]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
        for row in directional_rows
    ) if finite_directions else False

    call_gate = True
    for index in EXPECTED_CALLS:
        row = repairs.get(str(index), {})
        if not row.get("repeat_exact") or row.get("candidate_to_repair", {}).get("l2", 0.0) <= 0.0:
            call_gate = False
        for record in row.get("call_records", []):
            active = [entry for entry in record.values() if entry.get("calls", 0) > 0]
            if len(active) != 1 or active[0] != {"calls": 27, "repairs": 1}:
                call_gate = False

    gates = {
        "manifest_frozen": manifest.get("status") == "FROZEN_PRE_EXECUTION_DIRECTIONAL_EXTENSION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "result_status_valid": result.get("status") == EXPECTED_STATUS,
        "selected_calls_exact": result.get("selected_call_indices") == EXPECTED_CALLS,
        "all_executor_gates_true": bool(result.get("gates")) and all(result["gates"].values()),
        "each_repair_executed_and_nonzero": call_gate,
        "direction_values_finite": finite_directions,
        "direction_identities_hold": direction_identities,
    }
    payload = {
        "schema_version": "forkcert.qwen3-candidate-kernel15-direction-audit.v0.5",
        "status": "VALID_ORIGINAL_CANDIDATE_KERNEL_DIRECTION_AUDIT" if all(gates.values()) else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifact_checks,
        "directions": {str(index): repairs.get(str(index), {}).get("direction") for index in EXPECTED_CALLS},
        "claim": "direction of three selected original-candidate fused-kernel repairs relative to eager at one frozen state",
        "claim_limits": result.get("claim_limits", []),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "directions": payload["directions"]}, indent=2, sort_keys=True))
    if payload["status"] == "INVALID_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
