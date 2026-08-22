#!/usr/bin/env python3
"""Summarize the frozen 32-step screen-negative consequence population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def regime(levels: dict) -> str:
    local = bool(levels["local"]["above_sign_flip_95"])
    feedback = bool(levels["feedback"]["above_sign_flip_95"])
    actual = bool(levels["actual"]["above_sign_flip_95"])
    if not actual:
        return "NO_DETECTABLE_PERSISTENT_ACTUAL_DRIFT"
    if local and feedback:
        return "MIXED_LOCAL_AND_FEEDBACK_PERSISTENCE"
    if local:
        return "LOCAL_SOURCE_PERSISTENT"
    if feedback:
        return "FEEDBACK_SUSTAINED"
    return "ACTUAL_PERSISTENT_ORIGIN_UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    invalid = []
    for path in sorted(args.input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("trajectory_status") != "COMPLETE" or data.get("step_count") != 32:
            invalid.append({"path": str(path), "reason": "NOT_A_COMPLETE_32_STEP_RESULT"})
            continue
        if data.get("status") != "COMPLETE":
            invalid.append({"path": str(path), "reason": str(data.get("status"))})
            continue
        levels = data["statistics"]["levels"]
        rows.append({
            "case_id": data["case_id"],
            "architecture": data["architecture"],
            "carrier": data["carrier"],
            "regime": regime(levels),
            "max_recurrence_relative": data["max_recurrence_relative"],
            "levels": {
                name: {
                    "coherence_amplification": value["coherence_amplification"],
                    "null_upper_95": value["sign_flip_null"]["upper_95"],
                    "one_sided_p": value["sign_flip_null"]["one_sided_p"],
                    "above_sign_flip_95": value["above_sign_flip_95"],
                }
                for name, value in levels.items()
            },
            "prefix16": {
                name: value["prefix"]["16"]
                for name, value in levels.items()
            },
            "resultant_cosines": data["statistics"]["resultant_cosines"],
            "source": str(path),
        })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["regime"]] = counts.get(row["regime"], 0) + 1
    complete = len(rows) == args.expected and not invalid
    prefix_backtest = {}
    for level in ("local", "feedback", "actual"):
        prefix = np.asarray([
            row["prefix16"][level]["coherence_amplification"] for row in rows
        ])
        full = np.asarray([
            row["levels"][level]["coherence_amplification"] for row in rows
        ])
        correlation = (
            float(np.corrcoef(prefix, full)[0, 1]) if len(rows) > 1 else None
        )
        prefix_backtest[level] = {
            "evaluated": len(rows),
            "pearson_a16_a32": correlation,
            "mean_absolute_a_error": float(np.mean(np.abs(prefix - full))),
            "same_side_of_diffusive_one": int(np.count_nonzero(
                (prefix > 1.0) == (full > 1.0)
            )),
            "claim_boundary": (
                "Continuous retrospective backtest only. The 16-step Gram was not "
                "retained, so a horizon-matched sign-flip verdict is not imputed."
            ),
        }
    result = {
        "schema": "kernel-analyzer-bound-consequence-population-summary-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE_FAIL_CLOSED",
        "expected_cases": args.expected,
        "completed_cases": len(rows),
        "invalid_results": invalid,
        "regime_counts": counts,
        "prefix16_to_full32_backtest": prefix_backtest,
        "rows": rows,
        "claim_boundary": (
            "Regimes use the frozen per-level sign-flip null on complete 32-step "
            "four-arm trajectories. FEEDBACK_SUSTAINED is real closed-loop drift "
            "under the tested protocol, but is not local source persistence. A null "
            "result is no detectable persistence, not a proof of universal safety."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "completed": len(rows), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
