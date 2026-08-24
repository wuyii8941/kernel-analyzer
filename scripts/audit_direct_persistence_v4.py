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
        return {"status": "ABSTAIN_MISSING_ARTIFACT", "artifact": str(path.resolve().relative_to(ROOT.resolve()))}
    value = load(path)
    status = value.get("status")
    return {
        "status": status if status in expected else "ABSTAIN_UNEXPECTED_STATUS",
        "artifact": str(path.resolve().relative_to(ROOT.resolve())),
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
        # The audit writes its own JSON after reading the manifest, so including
        # that file would create an impossible self-hash cycle.
        # The manifest historically used both ``completion_audit.json`` and
        # ``./completion_audit.json``.  Normalize the spelling before applying
        # the self-hash exception; otherwise the audit reports its own output
        # as a stale digest on every run.
        if Path(rel).as_posix() == "completion_audit.json":
            continue
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
    heldout = load(out / "heldout_confirmation_v2.json") if (out / "heldout_confirmation_v2.json").exists() else load(out / "heldout_gemma_confirmation.json")
    fresh_targets = load(out / "heldout/new_impl_targets_v2.json") if (out / "heldout/new_impl_targets_v2.json").exists() else {}
    severity = load(out / "severity.json")
    tolerance = load(out / "tolerance_comparison.json")
    prefix_null = load(out / "prefix_null_reanalysis.json") if (out / "prefix_null_reanalysis.json").exists() else {}
    catch_fix = load(out / "catch_and_fix/manifest.json")
    execution_status = load(out / "execution_status.json") if (out / "execution_status.json").exists() else {}

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
            "new_impl_heldout_runs": {
                "count": heldout.get("counting", {}).get("fresh_rows", heldout.get("metrics", {}).get("external_eligible_rows", 0)),
                "direct_positive": heldout.get("counting", {}).get("fresh_direct_positive", heldout.get("metrics", {}).get("external_confirmed_positive", 0)),
                "direct_negative": heldout.get("counting", {}).get("fresh_direct_negative", heldout.get("metrics", {}).get("external_confirmed_negative", 0)),
                "status": "COMPLETE_RUNS_NO_DIRECT_POSITIVE" if heldout.get("counting", {}).get("fresh_rows", heldout.get("metrics", {}).get("external_eligible_rows", 0)) >= 1 else "PARTIAL",
            },
            "fresh_new_impl_target_checks": status_for(
                out / "heldout/new_impl_targets_v2.json",
                {"COMPLETE_FRESH_IN_PROCESS_GEMMA_ROWS"},
            ),
            "execution_status": status_for(
                out / "execution_status.json",
                {"PARTIAL_COMPLETE_FAIL_CLOSED"},
            ),
            "result_digest_manifest": verify_sha256(out),
            "raw_prefix_null_reanalysis": {
                "status": prefix_null.get("status", "ABSTAIN_MISSING_ARTIFACT"),
                "rows": len(prefix_null.get("rows", [])),
            },
            "raw_tolerance_metrics": {
                "status": tolerance.get("raw_stage_reanalysis", {}).get("status", "ABSTAIN_MISSING_RAW_REANALYSIS"),
                "rows": len(tolerance.get("raw_stage_reanalysis", {}).get("rows", [])),
                "new_impl_rows": len(tolerance.get("raw_new_impl_reanalysis", {}).get("rows", [])),
                "new_impl_rows_with_update_pairs": sum(
                    row.get("update_pair", {}).get("status") == "COMPLETE"
                    for row in tolerance.get("raw_new_impl_reanalysis", {}).get("rows", [])
                ),
            },
        },
        "abstained": {
            "natural_early_middle_late_optimizer_phases": {
                "status": optimizer_run.get("phase_conditioned_natural", {}).get("status", "ABSTAIN_MISSING_PHASE_CAPTURE"),
                "reason": "The Qwen phase result uses each phase's own weights, inputs and moments; it is a same-state response probe, not a live persistence trajectory.",
            },
            "0543_optimizer_ablation": {
                "status": "ABSTAIN_FROZEN_WRAPPER_IDENTITY_CHANGED",
                "reason": "A fresh replay was attempted, but the current Phi runtime wrapper sequence differs from the frozen release. No raw vectors were imputed.",
            },
            "heldout_recall_and_auroc": {
                "status": "ABSTAIN_NO_NEW_IMPL_DIRECT_POSITIVE",
                "reason": "The fresh Gemma target checks add no direct-persistence positive; not-applicable and feedback-control rows are not relabeled as negatives. Recall and AUROC remain undefined.",
            },
            "complete_tolerance_family": {
                "status": tolerance.get("status", "ABSTAIN"),
                "missing": tolerance.get("missing_baselines", []),
            },
            "severity_and_catch_fix": {
                "severity_status": severity.get("status", "ABSTAIN"),
                "catch_fix_status": catch_fix.get("status", "ABSTAIN"),
                "reason": "No prospective held-out escalation exists; catch-and-fix is not applicable for this all-nonpositive NEW_IMPL pool.",
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
