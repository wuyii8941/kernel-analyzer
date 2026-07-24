#!/usr/bin/env python
"""Generic fail-closed audit for an original-candidate kernel-repair result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-status", required=True)
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
        observed = digest(root / record["path"])
        artifact_checks[name] = {
            "expected": record["sha256"],
            "observed": observed,
            "pass": observed == record["sha256"],
        }

    selected = manifest["selection"]["call_indices"]
    expected_calls = int(result.get("expected_family_calls", -1))
    treatment_integrity = True
    arithmetic_integrity = True
    for index in selected:
        row = result.get("repairs", {}).get(str(index), {})
        if not row.get("repeat_exact") or len(row.get("call_records", [])) != 2:
            treatment_integrity = False
        for record in row.get("call_records", []):
            active = [entry for entry in record.values() if entry.get("calls", 0) > 0]
            if len(active) != 1 or active[0] != {"calls": expected_calls, "repairs": 1}:
                treatment_integrity = False
        direction = row.get("direction", {})
        try:
            base = float(direction["eager_candidate_l2"])
            repaired = float(direction["eager_repair_l2"])
            change = float(direction["l2_distance_change"])
            fraction = float(direction["fractional_l2_reduction"])
            effect = float(row["candidate_to_repair"]["l2"])
            if not all(math.isfinite(value) for value in (base, repaired, change, fraction, effect)):
                arithmetic_integrity = False
            if not math.isclose(change, repaired - base, rel_tol=1e-7, abs_tol=1e-9):
                arithmetic_integrity = False
            if not math.isclose(fraction, -change / base, rel_tol=1e-7, abs_tol=1e-9):
                arithmetic_integrity = False
            cosine = direction.get("cosine_repair_with_candidate_to_eager")
            if effect == 0.0:
                if cosine is not None:
                    arithmetic_integrity = False
            elif cosine is None or not math.isfinite(float(cosine)) or abs(float(cosine)) > 1.000001:
                arithmetic_integrity = False
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            arithmetic_integrity = False

    gates = {
        "manifest_frozen": manifest.get("status") == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "result_status_valid": result.get("status") == args.expected_status,
        "kernel_exact": result.get("kernel_family") == manifest["selection"]["kernel_family"],
        "selected_calls_exact": result.get("selected_call_indices") == selected,
        "all_executor_gates_true": bool(result.get("gates")) and all(result["gates"].values()),
        "treatment_integrity": treatment_integrity,
        "direction_arithmetic_integrity": arithmetic_integrity,
    }
    payload = {
        "schema_version": "forkcert.qwen3-candidate-kernel-repair-audit.v0.1",
        "status": "VALID_ORIGINAL_CANDIDATE_KERNEL_REPAIR_AUDIT" if all(gates.values()) else "INVALID_AUDIT",
        "kernel_family": result.get("kernel_family"),
        "gates": gates,
        "artifact_checks": artifact_checks,
        "effects": {
            str(index): {
                "candidate_to_repair": result.get("repairs", {}).get(str(index), {}).get("candidate_to_repair"),
                "direction": result.get("repairs", {}).get(str(index), {}).get("direction"),
            }
            for index in selected
        },
        "claim_limits": result.get("claim_limits", []),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "effects": payload["effects"]}, indent=2, sort_keys=True))
    if payload["status"] == "INVALID_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
