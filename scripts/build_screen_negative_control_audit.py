#!/usr/bin/env python3
"""Build a deterministic residual-nonzero screen-negative audit.

This consumes the already completed backward rescreen.  It does not relabel
formation or persistence; it only quantifies how many reachable, nonzero
screen negatives are available for a bounded recall audit and records a
mechanical sample without looking at historical case names or final drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/backward_rescreen_atlas.json"
OUT = ROOT / "results/property/joint_bias_formation_v1/screen_negative_control_audit.json"
MD = ROOT / "results/property/joint_bias_formation_v1/screen_negative_control_audit.md"


def key(row: dict) -> str:
    return hashlib.sha256(str(row["task_id"]).encode()).hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = list(source["rows"])
    eligible = [
        row for row in rows
        if float(row.get("confirmation_local_ratio") or 0.0) > 0.0
        and bool(row.get("confirmation_available"))
        and str(row.get("outcome")) in {
            "NOT_REPRODUCED", "DIRECTION_REVERSAL", "SAME_SIGN_UNRESOLVED_OR_CENTERED"
        }
    ]
    eligible.sort(key=key)
    sample = eligible[:12]
    payload = {
        "schema": "kernel-analyzer-screen-negative-control-audit-v1",
        "status": "COMPLETE_BOUNDED_SCREEN_AUDIT",
        "source_artifact": str(SOURCE.relative_to(ROOT)),
        "selection": {
            "rule": "confirmation_local_ratio>0, confirmation_available=true, non-promoted outcome; sort by SHA256(task_id); take first 12",
            "uses_t4_or_seup": False,
            "uses_case_names": False,
            "sample_size": len(sample),
        },
        "denominator": {
            "backward_rescreen_rows": len(rows),
            "reachable_nonzero_screen_negative_pool": len(eligible),
            "sampled_rows": len(sample),
            "outcome_counts_in_pool": dict(Counter(str(row.get("outcome")) for row in eligible)),
            "outcome_counts_in_sample": dict(Counter(str(row.get("outcome")) for row in sample)),
        },
        "sample": [
            {
                "task_id": row["task_id"],
                "model": row["model"],
                "family": row["family"],
                "sequence_length": row["sequence_length"],
                "confirmation_local_ratio": row["confirmation_local_ratio"],
                "confirmation_gradient_ratio": row["confirmation_gradient_ratio"],
                "confirmation_gradient_status": row["confirmation_gradient_status"],
                "outcome": row["outcome"],
            }
            for row in sample
        ],
        "claim_boundary": "This is a deterministic screen-level reachable-negative audit. It is not a new 32-step consequence campaign and cannot estimate end-to-end persistence recall until these sampled rows receive the full trajectory protocol.",
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Screen-negative control audit",
        "",
        "This audit is mechanical and consumes existing backward rescreen artifacts.",
        "It is not a new persistence verdict.",
        "",
        f"- reachable nonzero pool: **{len(eligible)}**",
        f"- deterministic sample: **{len(sample)}**",
        f"- outcomes in pool: `{dict(Counter(str(row.get('outcome')) for row in eligible))}`",
        "",
        "| task | model | family | local ratio | gradient status | outcome |",
        "|---|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| `{row['task_id']}` | `{row['model']}` | `{row['family']}` | {row['confirmation_local_ratio']:.4g} | `{row['confirmation_gradient_status']}` | `{row['outcome']}` |"
        for row in sample
    )
    lines.extend(["", payload["claim_boundary"]])
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "pool": len(eligible), "sample": len(sample)}, sort_keys=True))


if __name__ == "__main__":
    main()
