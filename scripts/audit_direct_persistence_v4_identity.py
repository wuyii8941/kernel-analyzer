#!/usr/bin/env python3
"""Check that every v4 row has a reproducible experiment identity.

This audit never fills a missing digest from a related run.  Historical v3
rows are therefore allowed to remain ``PARTIAL_IDENTITY`` while fresh heldout
rows can be complete.  The result is intended to make the boundary visible
before a tolerance or persistence number is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/property/direct_persistence_v4"

REQUIRED = (
    "model",
    "exact_endpoint",
    "sequence_length",
    "state_order",
    "parameter_coordinates",
    "optimizer",
    "moment_state",
    "horizon",
    "repair",
    "runner_version",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative(path: str | None) -> str | None:
    if not path:
        return None
    candidate = (ROOT / path).resolve()
    try:
        return str(candidate.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def historical_identity(row: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / str(row.get("source", ""))
    if not source.is_file():
        return {"case_id": row.get("case_id"), "status": "ABSTAIN_MISSING_SOURCE", "missing": list(REQUIRED)}
    document = load(source)
    plan_path = ROOT / str(document.get("case_plan", ""))
    plan = load(plan_path) if plan_path.is_file() else {}
    plan_case = next((item for item in plan.get("cases", []) if item.get("case_id") == row.get("case_id")), {})
    runner = str(document.get("runner", ""))
    identity = {
        "model": document.get("model"),
        "exact_endpoint": {
            "task_id": plan_case.get("task_id"),
            "case_plan": relative(document.get("case_plan")),
            "release": relative(document.get("release")),
        },
        "sequence_length": plan.get("sequence_length"),
        "state_order": document.get("step_ids") or document.get("state_ids"),
        "parameter_coordinates": {
            "carrier": document.get("carrier"),
            "coordinate_count": document.get("carrier_coordinates"),
            "digest": None,
        },
        "optimizer": document.get("optimizer"),
        "moment_state": None,
        "horizon": document.get("steps"),
        "repair": {
            "case_plan": relative(document.get("case_plan")),
            "four_counterfactual_arms": document.get("four_counterfactual_arms_required"),
        },
        "runner_version": {
            "path": relative(runner),
            "sha256": digest(ROOT / runner),
        },
    }
    missing = [key for key, value in identity.items() if value in (None, "", [], {})]
    return {
        "case_id": row.get("case_id"),
        "role": row.get("confirmation_role"),
        "status": "COMPLETE" if not missing else "PARTIAL_IDENTITY",
        "missing": missing,
        "identity": identity,
        "source": relative(str(source.relative_to(ROOT))),
    }


def fresh_identity(row: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "model": row.get("model"),
        "exact_endpoint": row.get("target_region"),
        "sequence_length": row.get("sequence_length"),
        "state_order": row.get("state_order"),
        "parameter_coordinates": row.get("carrier"),
        "optimizer": row.get("optimizer"),
        "moment_state": row.get("moment_state"),
        "horizon": row.get("consequence", {}).get("steps"),
        "repair": row.get("repair"),
        "runner_version": row.get("runtime_capture_sha256"),
    }
    missing = [key for key, value in identity.items() if value in (None, "", [], {})]
    return {
        "case_id": row.get("case_id"),
        "role": row.get("status"),
        "status": "COMPLETE" if not missing else "PARTIAL_IDENTITY",
        "missing": missing,
        "identity": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    cohort = load(args.base / "cohort.json")
    rows = [historical_identity(row) for row in cohort.get("rows", [])]
    fresh_path = args.base / "heldout_confirmation_v2.json"
    if fresh_path.is_file():
        rows.extend(fresh_identity(row) for row in load(fresh_path).get("rows", []))
    result = {
        "schema": "kernel-analyzer-direct-persistence-v4-identity-audit-v1",
        "status": "COMPLETE" if rows and all(row["status"] == "COMPLETE" for row in rows) else "PARTIAL_FAIL_CLOSED",
        "required_fields": list(REQUIRED),
        "rows": rows,
        "claim_boundary": "Missing state-bank, parameter-coordinate, moment-state or runner identity is reported explicitly; no related run is used to fill it.",
    }
    output = args.base / "identity_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
