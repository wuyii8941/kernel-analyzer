#!/usr/bin/env python3
"""Convert a completed short screen into the v4 fail-closed decision format.

The input is the existing shared short-screen certificate.  This wrapper adds
the explicit cold-start AdamW identity and never turns a short negative into a
safety result.  Missing protocol identity or malformed cases become ABSTAIN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PROTOCOL = {
    "name": "Cold-start AdamW Direct Persistence Screen",
    "optimizer": "AdamW",
    "moment_initialization": "zero at the start, then evolved normally",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def classify(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    protocol = payload.get("direct_persistence_protocol")
    if protocol != REQUIRED_PROTOCOL:
        return [{
            "source": source,
            "case_id": source,
            "decision": "ABSTAIN",
            "reason": "missing or incompatible cold-start AdamW protocol identity",
        }]
    if payload.get("status") != "COMPLETE" or not isinstance(payload.get("cases"), list):
        return [{
            "source": source,
            "case_id": source,
            "decision": "ABSTAIN",
            "reason": "short screen is incomplete or malformed",
        }]

    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        case_id = row.get("case_id")
        projection = row.get("projection", {})
        null = row.get("sign_flip_null", {})
        rule = row.get("screen_rule", {})
        valid = (
            isinstance(case_id, str)
            and bool(case_id)
            and int(row.get("steps", 0)) >= 16
            and int(projection.get("dimension", 0)) >= 4
            and int(null.get("draws", 0)) > 0
            and rule.get("requires_observed_above_sign_flip_95") is True
            and rule.get("requires_lag1_and_at_least_two_positive_lags") is True
            and rule.get("requires_late_prefix_growth") is True
        )
        if not valid:
            decision = "ABSTAIN"
            reason = "missing frozen screen fields"
        elif row.get("status") == "RISK_CANDIDATE":
            decision = "ESCALATE"
            reason = "short screen met the frozen escalation rule"
        elif row.get("status") == "NULL_LIKE_OR_UNRESOLVED":
            decision = "NO_ESCALATION_UNDER_SHORT_SCREEN"
            reason = "short screen did not escalate; this is not a safety verdict"
        else:
            decision = "ABSTAIN"
            reason = "unknown short-screen status"
        rows.append({
            "source": source,
            "case_id": case_id,
            "decision": decision,
            "reason": reason,
            "steps": row.get("steps"),
            "observed_amplification": row.get("observed_amplification"),
            "sign_flip_null_upper_95": null.get("upper_95"),
            "projection": projection,
            "protocol": protocol,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(classify(load(path), str(path)))
    summary = {
        "total": len(rows),
        "escalate": sum(row["decision"] == "ESCALATE" for row in rows),
        "no_escalation_under_short_screen": sum(
            row["decision"] == "NO_ESCALATION_UNDER_SHORT_SCREEN" for row in rows
        ),
        "abstain": sum(row["decision"] == "ABSTAIN" for row in rows),
    }
    result = {
        "schema": "kernel-analyzer-direct-persistence-v4-screen-result-v1",
        "status": "COMPLETE",
        "summary": summary,
        "rows": rows,
        "claim_boundary": "ESCALATE selects a full confirmation. NO_ESCALATION_UNDER_SHORT_SCREEN is not a safety verdict.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
