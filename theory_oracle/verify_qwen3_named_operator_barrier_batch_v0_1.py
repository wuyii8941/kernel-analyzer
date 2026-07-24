#!/usr/bin/env python
"""Verify and summarize the Qwen3 named-operator barrier batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text())

    artifact_checks = {}
    for name, record in manifest["artifacts"].items():
        observed = digest(root / record["path"])
        artifact_checks[name] = {"expected": record["sha256"], "observed": observed, "pass": observed == record["sha256"]}

    patterns = Counter()
    rows = []
    for target_id, record in sorted(result["targets"].items()):
        injection_nonzero = record["injection_effect"]["l2"] > 0
        repair_nonzero = record["repair_effect"]["l2"] > 0
        pattern = (
            "BOTH_NONZERO" if injection_nonzero and repair_nonzero
            else "INJECTION_ONLY" if injection_nonzero
            else "REPAIR_ONLY" if repair_nonzero
            else "BOTH_ZERO"
        )
        patterns[pattern] += 1
        rows.append({
            "target": target_id,
            "pattern": pattern,
            "injection_l2": record["injection_effect"]["l2"],
            "repair_l2": record["repair_effect"]["l2"],
            "injection_mean_signed": record["injection_effect"]["mean_signed"],
            "repair_mean_signed": record["repair_effect"]["mean_signed"],
        })

    gates = {
        "manifest_frozen": manifest["status"] == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "result_status_valid": result["status"] == "VALID_BARRIER_CONDITIONED_BATCH",
        "all_result_gates_true": all(result["gates"].values()),
        "target_count_exact": len(rows) == 35,
        "all_target_records_barrier_conditioned": all(row["status"] == "BARRIER_CONDITIONED" for row in result["targets"].values()),
        "summary_counts_recomputed": (
            result["summary"]["nonzero_injection_effects"] == patterns["BOTH_NONZERO"] + patterns["INJECTION_ONLY"]
            and result["summary"]["nonzero_repair_effects"] == patterns["BOTH_NONZERO"] + patterns["REPAIR_ONLY"]
        ),
        "original_candidate_transport_refused": (
            result["candidate_anchor_exact"] is False
            and result["transport_to_original_candidate"] == "NOT_ESTABLISHED"
        ),
    }
    passed = all(gates.values())
    audit = {
        "schema_version": "forkcert.qwen3-named-operator-barrier-batch-audit.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_BATCH_EVIDENCE" if passed else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifact_checks,
        "effect_patterns": dict(sorted(patterns.items())),
        "coverage_credit": {"barrier_conditioned_invocations": 35, "original_candidate_invocations": 0},
        "original_candidate_root_cause_credit": False,
        "rows": rows,
    }
    out_audit = Path(args.out_audit).resolve()
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3 named-operator barrier batch findings v0.1",
        "",
        "## Verdict",
        "",
        audit["status"],
        "",
        f"All 35 named-module representatives passed the fixed-boundary integrity gates. Effect patterns: `{dict(sorted(patterns.items()))}`.",
        "",
        "The barrier candidate did not reproduce the original compiled anchor. Every row below is therefore intervention-dependent `BARRIER_CONDITIONED` evidence and receives no original-candidate root-cause credit.",
        "",
        "## Per-target effects",
        "",
        "| Target | Pattern | Injection L2 | Repair L2 | Injection signed mean | Repair signed mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target']}` | {row['pattern']} | {row['injection_l2']:.9g} | {row['repair_l2']:.9g} | {row['injection_mean_signed']:.9g} | {row['repair_mean_signed']:.9g} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A zero row means only that isolated compilation of that invocation had no selected-token effect in this fixed-boundary program and state. A nonzero row establishes an effect in the same program, not in the original fused candidate. Early/middle/late rows are separate observations; they do not yet validate transport across layers or states.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": audit["status"], "gates": gates, "effect_patterns": audit["effect_patterns"]}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
