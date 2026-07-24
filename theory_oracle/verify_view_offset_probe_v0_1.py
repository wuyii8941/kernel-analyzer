#!/usr/bin/env python
"""Independent audit for the view/storage-offset region probe."""

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
    if first["schema_version"] != "forkcert.view_offset_probe.v0.1":
        errors.append("first schema mismatch")
    if second["schema_version"] != first["schema_version"]:
        errors.append("repeat schema mismatch")
    if first["environment"] != second["environment"]:
        errors.append("environment changed between repeats")
    if len(first["rows"]) != len(second["rows"]):
        errors.append("repeat row count mismatch")
    for left, right in zip(first["rows"], second["rows"]):
        for key in ("input", "chain_exact", "first_exact", "second_local_exact", "chain_delta", "second_local_delta"):
            if left[key] != right[key]:
                errors.append(f"repeat mismatch in row {left['index']} field {key}")
    rows = first["rows"]
    production = any(not row["second_local_exact"] for row in rows)
    first_isolated_exact = all(row["first_exact"] for row in rows)
    full_failure = any(not row["chain_exact"] for row in rows)
    provenance = first["provenance"]
    # Torch 2.7's wrapper contains no ATen-origin comments or Triton kernel;
    # this is an explicit evidence gap, not permission to infer a kernel.
    no_kernel_provenance = not any(item["kernel_paths"] for group in provenance.values() for item in group["artifacts"])
    report = {
        "schema_version": "forkcert.view_offset_probe_audit.v0.1",
        "valid": not errors and full_failure and production and first_isolated_exact,
        "errors": errors,
        "repeatability": {"independent_runs_match": not errors},
        "evidence": {
            "full_case_failure_observed": full_failure,
            "view_first_isolated_exact": first_isolated_exact,
            "view_second_same_input_production_observed": production,
            "generated_kernel_provenance_present": not no_kernel_provenance,
        },
        "allowed_claim_level": "LOCAL_INJECTION_WITH_WRAPPER_STOP" if not errors and production else "OBSERVATION",
        "stop_reason": (
            "The discrepancy is reproducible in the second-view same-input replay, "
            "but the target release emits a wrapper-only reinterpret operation with "
            "no auditable generated-kernel or ATen-origin mapping; stop before kernel/root-cause claims."
        ),
        "inputs": {"first": str(Path(args.first).resolve()), "second": str(Path(args.second).resolve())},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
