#!/usr/bin/env python3
"""Audit the Direct Persistence v4 package without inventing missing evidence.

The audit is deliberately fail-closed.  A missing natural optimizer phase,
held-out positive, tolerance measurement, or executable repair is reported as
``ABSTAIN`` rather than being inferred from a related experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/property/direct_persistence_v4"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def status_for(path: Path, expected: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "ABSTAIN_MISSING_ARTIFACT", "artifact": str(path.relative_to(ROOT))}
    value = load(path)
    status = value.get("status")
    return {
        "status": status if status in expected else "ABSTAIN_UNEXPECTED_STATUS",
        "artifact": str(path.relative_to(ROOT)),
        "observed_status": status,
    }


def verify_sha256(out: Path) -> dict[str, Any]:
    manifest = out / "SHA256SUMS"
    if not manifest.exists():
        return {"status": "ABSTAIN_MISSING_SHA256SUMS"}
    checked = 0
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = out / rel
        checked += 1
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(rel)
    return {
        "status": "COMPLETE" if not failures else "INVALID_DIGESTS",
        "files_checked": checked,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output

    retrospective = load(out / "retrospective_metrics.json")
    contribution = load(out / "contribution_table.json")
    multiplicity = load(out / "multiplicity.json")
    optimizer = load(out / "optimizer_state/manifest.json")
    optimizer_run = load(out / "optimizer_state_run_manifest.json")
    heldout = load(out / "heldout_gemma_confirmation.json")
    severity = load(out / "severity.json")
    tolerance = load(out / "tolerance_comparison.json")
    catch_fix = load(out / "catch_and_fix/manifest.json")

    rows = contribution.get("rows", [])
    audit = {
        "schema": "kernel-analyzer-direct-persistence-v4-completion-audit-v1",
        "status": "PARTIAL_COMPLETE_FAIL_CLOSED",
        "completed": {
            "retrospective_15_row_reanalysis": retrospective.get("status") == "COMPLETE_OFFLINE_REANALYSIS_WITH_EXPLICIT_UNRESOLVED_FIELDS",
            "three_case_signed_contribution_table": len(rows) == 3 and all(row.get("status", "").startswith("COMPLETE") for row in rows),
            "holm_families": multiplicity.get("primary_method") == "Holm family-wise correction",
            "same_state_optimizer_ablation_cases": {
                "count": len(optimizer.get("same_state_ablation", {}).get("completed_cases", [])),
                "cases": optimizer.get("same_state_ablation", {}).get("completed_cases", []),
                "status": "COMPLETE_FOUR_CASES" if len(optimizer.get("same_state_ablation", {}).get("completed_cases", [])) == 4 else "PARTIAL",
            },
            "one_new_impl_heldout_negative": heldout.get("metrics", {}).get("eligible_rows") == 1 and heldout.get("metrics", {}).get("confirmed_negative") == 1,
            "result_digest_manifest": verify_sha256(out),
        },
        "abstained": {
            "natural_early_middle_late_optimizer_phases": {
                "status": optimizer_run.get("phase_conditioned_natural", {}).get("status", "ABSTAIN_MISSING_PHASE_CAPTURE"),
                "reason": "No genuine early, middle and late weights/input/gradient/moment captures are available; no moments were mixed across phases.",
            },
            "0543_optimizer_ablation": {
                "status": "ABSTAIN_MISSING_RAW_GRADIENT_AND_MOMENT_CAPTURE",
                "reason": "The archived 0543 file has norms/digests but not vectors or optimizer moments needed for a same-state response replay.",
            },
            "heldout_recall_and_auroc": {
                "status": "ABSTAIN_ONE_NEW_IMPL_NEGATIVE_ONLY",
                "reason": "One valid NEW_IMPL row cannot define recall or AUROC, and no positive is inferred.",
            },
            "complete_tolerance_family": {
                "status": tolerance.get("status", "ABSTAIN"),
                "missing": tolerance.get("missing_baselines", []),
            },
            "severity_and_catch_fix": {
                "severity_status": severity.get("status", "ABSTAIN"),
                "catch_fix_status": catch_fix.get("status", "ABSTAIN"),
                "reason": "No prospective held-out escalation exists, so an executable prospective repair cannot be claimed.",
            },
        },
        "scientific_boundary": (
            "The package supports a cold-start AdamW Direct Persistence Screen and "
            "shows that optimizer response changes the mapping from gradient error "
            "to effective update. It does not support a universal all-operator "
            "Oracle, natural phase invariance, or an AdamW root-cause claim."
        ),
    }
    (out / "completion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output": str(out / "completion_audit.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
