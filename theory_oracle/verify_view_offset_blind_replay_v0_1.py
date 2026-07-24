#!/usr/bin/env python
"""Fail-closed audit for the anonymous case_002 blind replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    errors: list[str] = []
    for report in (first, second):
        if report.get("schema_version") != "forkcert.view_offset_blind_replay.v0.1":
            errors.append("schema mismatch")
        if report.get("case_id") != "case_002":
            errors.append("case id is not opaque case_002")
        gates = report.get("gates", {})
        for key in ("complete_witness", "first_boundary_control_exact", "same_input_local_production"):
            if not gates.get(key):
                errors.append(f"required gate failed: {key}")
        if gates.get("kernel_provenance_present"):
            errors.append("unexpected kernel provenance claim")
    if first["environment"] != second["environment"]:
        errors.append("environment differs across repeats")
    if first["rows"] != second["rows"]:
        errors.append("row-level replay evidence differs across repeats")
    report = {
        "schema_version": "forkcert.view_offset_blind_replay_audit.v0.1",
        "valid": not errors,
        "errors": errors,
        "case_id": first.get("case_id"),
        "evidence": {
            "complete_witness": first["gates"].get("complete_witness", False),
            "same_input_local_production": first["gates"].get("same_input_local_production", False),
            "boundary_mediation_observed": first["gates"].get("boundary_mediation_observed", False),
            "kernel_provenance_present": first["gates"].get("kernel_provenance_present", False),
            "independent_repeats_identical": first.get("rows") == second.get("rows"),
        },
        "allowed_claim_level": "LOCAL_INJECTION_WITH_WRAPPER_STOP" if not errors else "INVALID",
        "limitations": [
            "anonymous package has no external patch score yet",
            "wrapper-level local production does not prove a unique compiler root cause",
            "no generated kernel provenance was available in the target release",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
