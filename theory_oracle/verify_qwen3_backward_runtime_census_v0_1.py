#!/usr/bin/env python
"""Independently audit the frozen Qwen3 backward runtime census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--transition-result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    census_path = Path(args.census).resolve()
    transition_path = Path(args.transition_result).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text())
    census = json.loads(census_path.read_text())
    transition = json.loads(transition_path.read_text())
    static_path = root / manifest["artifacts"]["static_backward_summary"]["path"]
    static = json.loads(static_path.read_text())

    artifact_checks: dict[str, Any] = {}
    for name, record in manifest["artifacts"].items():
        observed = digest(root / record["path"])
        artifact_checks[name] = {
            "expected": record["sha256"],
            "observed": observed,
            "pass": observed == record["sha256"],
        }

    expected_family_counts = {
        row["name"]: int(row["call_count"]) for row in static["kernel_families"]
    }
    expected_external = {
        name: int(count) for name, count in static["extern_kernel_call_site_counts"].items()
    }
    observed_family_counts = {
        name: int(count) for name, count in census.get("family_call_counts", {}).items()
    }
    observed_external = {
        name: int(count)
        for name, count in census.get("external_call_counts_during_backward", {}).items()
    }
    gates = {
        "manifest_frozen": manifest.get("status") == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "census_status_valid": census.get("status") == "VALID_BACKWARD_RUNTIME_CENSUS",
        "all_executor_gates_true": bool(census.get("gates")) and all(census["gates"].values()),
        "transition_valid": transition.get("valid") is True
        and transition.get("verdict") == "VALID",
        "candidate_identity_valid": transition.get("compiler", {}).get("candidate_identity_valid")
        is True,
        "scorer_anchor_exact": transition.get("anchors", {}).get("scorer_anchor_exact") is True,
        "family_set_and_counts_exact": observed_family_counts == expected_family_counts,
        "external_counts_exact": observed_external == expected_external,
        "all_static_families_active": all(count > 0 for count in observed_family_counts.values()),
        "embedded_census_exact": transition.get("backward_runtime_census") == census,
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-runtime-census-audit.v0.1",
        "status": "VALID_BACKWARD_RUNTIME_CENSUS_AUDIT" if all(gates.values()) else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifact_checks,
        "counts": {
            "triton_families": len(observed_family_counts),
            "triton_calls": sum(observed_family_counts.values()),
            "external_calls": observed_external,
        },
        "claim_limits": census.get("claim_limits", []),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "counts": payload["counts"]}, indent=2, sort_keys=True))
    if payload["status"] != "VALID_BACKWARD_RUNTIME_CENSUS_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
