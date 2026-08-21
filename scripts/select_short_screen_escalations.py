#!/usr/bin/env python3
"""Turn shared short-screen certificates into an exact-run escalation manifest.

This selector never emits SAFE.  Null-like, incomplete, and malformed screens
remain in the denominator as abstentions; only a complete frozen risk witness
is escalated to the expensive exact trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: payload must be an object")
    return payload


def classify(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    if payload.get("status") != "COMPLETE" or not isinstance(payload.get("cases"), list):
        return [{"source": source, "case_id": source, "decision": "ABSTAIN_INVALID"}]
    protocol = payload.get("protocol", {})
    rows = []
    for row in payload["cases"]:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            rows.append({"source": source, "case_id": source, "decision": "ABSTAIN_INVALID"})
            continue
        rule = row.get("screen_rule", {})
        projection = row.get("projection", {})
        complete = (
            row.get("status") in {"RISK_CANDIDATE", "NULL_LIKE_OR_UNRESOLVED"}
            and int(row.get("steps", 0)) >= 4
            and int(projection.get("dimension", 0)) >= 4
            and rule.get("requires_observed_above_sign_flip_95") is True
            and rule.get("requires_lag1_and_at_least_two_positive_lags") is True
        )
        if not complete:
            decision = "ABSTAIN_INVALID"
        elif row["status"] == "RISK_CANDIDATE":
            decision = "ESCALATE_EXACT_TRAJECTORY"
        else:
            # This is not a safety verdict.  It only avoids the expensive arm.
            decision = "ABSTAIN_NO_ESCALATION"
        rows.append({
            "source": source,
            "case_id": case_id,
            "decision": decision,
            "short_status": row.get("status"),
            "observed_amplification": row.get("observed_amplification"),
            "sign_flip_null_upper_95": row.get("sign_flip_null", {}).get("upper_95"),
            "projection_dimension": projection.get("dimension"),
            "steps": row.get("steps"),
            "protocol": {
                "projection_seed": protocol.get("projection_seed"),
                "null_draws": protocol.get("null_draws"),
                "prefix_growth_mode": protocol.get("prefix_growth_mode"),
            },
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(classify(load(path), str(path)))
    summary = {
        "total": len(rows),
        "escalate": sum(row["decision"] == "ESCALATE_EXACT_TRAJECTORY" for row in rows),
        "abstain_no_escalation": sum(row["decision"] == "ABSTAIN_NO_ESCALATION" for row in rows),
        "abstain_invalid": sum(row["decision"] == "ABSTAIN_INVALID" for row in rows),
    }
    payload = {
        "schema": "kernel-analyzer-short-screen-escalation-manifest-v1",
        "status": "COMPLETE",
        "summary": summary,
        "rows": rows,
        "claim_boundary": "The selector chooses expensive follow-up; abstention is not a SAFE verdict and every input remains in the denominator.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
