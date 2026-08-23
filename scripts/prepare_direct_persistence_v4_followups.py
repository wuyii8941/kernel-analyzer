#!/usr/bin/env python3
"""Prepare fail-closed v4 follow-up manifests.

This command does not claim that GPU measurements exist.  It validates a
mechanically frozen held-out pool when supplied and otherwise writes explicit
NOT_STARTED manifests for optimizer-state, held-out and catch-and-fix work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/direct_persistence_v4"
REQUIRED_IDENTITY = {
    "case_id", "model", "implementation_class", "endpoint", "sequence_length",
    "state_order", "state_bank_digest", "parameter_coordinate_digest", "repair",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pool(pool: Any) -> dict[str, Any]:
    if not isinstance(pool, dict) or pool.get("schema") != "kernel-analyzer-direct-persistence-v4-heldout-pool-v1":
        return {"status": "ABSTAIN_INVALID_POOL_SCHEMA", "rows": [], "errors": ["wrong schema"]}
    if pool.get("status") != "FROZEN_BEFORE_REVEAL":
        return {"status": "ABSTAIN_POOL_NOT_FROZEN", "rows": [], "errors": ["pool must be frozen before reveal"]}
    rows = pool.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"status": "ABSTAIN_EMPTY_POOL", "rows": [], "errors": ["rows must be nonempty"]}
    seen: set[str] = set()
    errors: list[str] = []
    checked: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: not an object")
            continue
        missing = sorted(
            key for key in REQUIRED_IDENTITY
            if key not in row or row.get(key) in (None, "", [])
        )
        case_id = row.get("case_id")
        if missing:
            errors.append(f"{case_id or index}: missing {','.join(missing)}")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"row {index}: invalid case_id")
        elif case_id in seen:
            errors.append(f"{case_id}: duplicate case_id")
        else:
            seen.add(case_id)
        forbidden = {"label", "full32_label", "p_value", "screen_result", "confirmation_result"} & set(row)
        if forbidden:
            errors.append(f"{case_id or index}: reveal fields present: {sorted(forbidden)}")
        checked.append({"case_id": case_id, "status": "READY" if not missing else "ABSTAIN_MISSING_IDENTITY"})
    status = "READY" if not errors else "ABSTAIN_INVALID_POOL"
    return {"status": status, "rows": checked, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-pool", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.heldout_pool is None:
        pool_result = {
            "status": "NOT_STARTED",
            "rows": [],
            "errors": ["no pool supplied; selection must be frozen mechanically before reveal"],
        }
    else:
        pool_result = validate_pool(load(args.heldout_pool))
    (args.output / "heldout_pool_validation.json").write_text(
        json.dumps(pool_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    optimizer = {
        "schema": "kernel-analyzer-direct-persistence-v4-optimizer-state-run-manifest-v1",
        "status": "NOT_STARTED",
        "required_arms": ["captured_moments", "moment_reset", "stateless_sgd"],
        "required_cases": ["liger_fused_ce_t128", "phi4_seq64_lmhead_dx", "qwen_seq128_lmhead_dx", "feedback_dominated_control", "multishape-backward-cell-0543"],
        "missing_artifact_action": "ABSTAIN",
    }
    heldout = {
        "schema": "kernel-analyzer-direct-persistence-v4-heldout-run-manifest-v1",
        "status": "READY" if pool_result["status"] == "READY" else pool_result["status"],
        "pool_validation": "heldout_pool_validation.json",
        "required_runs": ["16-step short screen for every row", "32-step confirmation for every eligible row"],
        "reveal_after": ["protocol freeze", "score freeze", "severity freeze", "tolerance freeze"],
        "no_recall_if_all_negative": True,
    }
    catch_fix = {
        "schema": "kernel-analyzer-direct-persistence-v4-catch-and-fix-run-manifest-v1",
        "status": "WAITING_FOR_HELDOUT_ESCALATION",
        "required_sequence": ["screen", "confirm", "localize", "executable repair", "repeat", "loss", "runtime"],
        "no_pool_replacement_after_reveal": True,
    }
    (args.output / "optimizer_state_run_manifest.json").write_text(json.dumps(optimizer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "heldout_run_manifest.json").write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "catch_and_fix_run_manifest.json").write_text(json.dumps(catch_fix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"heldout": heldout["status"], "optimizer": optimizer["status"], "catch_and_fix": catch_fix["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
