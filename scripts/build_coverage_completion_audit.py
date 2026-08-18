#!/usr/bin/env python3
"""Audit completion of the requested fail-closed coverage infrastructure."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
REQUIRED_ROW_FIELDS = {
    "mathematical_fb",
    "eager_aot_binding",
    "candidate_region_binding",
    "numerical_measurement",
    "condition_unit_id",
    "bias_verdict",
}


def read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def current_claim_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "scripts").glob("*.py"))]


def main() -> None:
    qwen = read(COVERAGE / "qwen_invocation_ledger.json.gz")
    gaps = read(COVERAGE / "qwen_gap_audit.json")
    mamba = read(COVERAGE / "mamba_invocation_ledger.json.gz")
    moe = read(COVERAGE / "moe_invocation_ledger.json.gz")
    ambiguous_terms = ("matrix_" + "exhausted", "current_model_matrix_" + "exhaustion")
    current_ambiguous_claims = [
        str(path.relative_to(ROOT))
        for path in current_claim_files()
        if any(term in path.read_text() for term in ambiguous_terms)
    ]

    requirements = [
        {
            "id": "single_qwen_fail_closed_per_invocation_ledger",
            "complete": (
                len(qwen["rows"]) == 9269
                and len({row["row_id"] for row in qwen["rows"]}) == 9269
                and all(REQUIRED_ROW_FIELDS <= row.keys() for row in qwen["rows"])
                and len(qwen["mathematical_templates"]) == 54
            ),
            "evidence": "results/coverage/qwen_invocation_ledger.json.gz",
            "counts": qwen["summary"],
        },
        {
            "id": "remove_ambiguous_matrix_completion_claim",
            "complete": not current_ambiguous_claims,
            "evidence": "README.md, docs/*.md, scripts/*.py",
            "files_still_containing_claim": current_ambiguous_claims,
        },
        {
            "id": "close_1335_legacy_eager_aot_gaps",
            "complete": (
                gaps["eager_aot"]["unresolved"] == 0
                and gaps["eager_aot"]["legacy_unresolved"] == 1335
                and len(gaps["eager_aot"]["rows"]) == 1335
                and all(row["closure_status"] for row in gaps["eager_aot"]["rows"])
            ),
            "evidence": "results/coverage/qwen_gap_audit.json",
            "counts": {
                "exact_or_closed_semantic_region": gaps["eager_aot"]["exact_or_closed_semantic_region"],
                "unresolved": gaps["eager_aot"]["unresolved"],
                "legacy_classes": gaps["eager_aot"]["legacy_unresolved_by_alignment_status"],
                "closure_status_counts": gaps["eager_aot"]["closure_status_counts"],
            },
        },
        {
            "id": "audit_1476_candidate_binding_gaps",
            "complete": (
                gaps["candidate_binding"]["original_unresolved_static_source_node_audit"]["denominator"] == 1476
                and gaps["candidate_binding"]["original_unresolved_static_source_node_audit"]["phase_qualified_generated_source_node_hits"] == 0
                and gaps["candidate_binding"]["original_unresolved_static_source_node_audit"]["supplemental_exact_recoveries_from_original_unresolved"] == 29
                and gaps["candidate_binding"]["original_unresolved_static_source_node_audit"]["remaining_unresolved"] == 1447
            ),
            "evidence": "results/coverage/qwen_gap_audit.json",
            "counts": gaps["candidate_binding"]["original_unresolved_static_source_node_audit"],
        },
        {
            "id": "reclassify_122_changed_nonclosed_units",
            "complete": (
                gaps["changed_nonclosed_reclassification"]["legacy_denominator"] == 122
                and gaps["changed_nonclosed_reclassification"]["reclassified_complete"] == 122
                and gaps["changed_nonclosed_reclassification"]["remaining_unresolved"] == 0
            ),
            "evidence": "results/coverage/qwen_gap_audit.json",
            "counts": gaps["changed_nonclosed_reclassification"]["semantics_counts"],
        },
    ]
    for architecture, ledger, total in (("mamba", mamba, 56411), ("moe", moe, 25582)):
        requirements.append({
            "id": f"{architecture}_full_model_forward_backward_invocation_atlas",
            "complete": (
                len(ledger["rows"]) == total
                and len({row["row_id"] for row in ledger["rows"]}) == total
                and all(REQUIRED_ROW_FIELDS <= row.keys() for row in ledger["rows"])
                and ledger["gates"]["all_local_maps_and_adjoints_declared"]
                and ledger["gates"]["all_fb_origins_or_explicit_auxiliaries_complete"]
                and ledger["instrumentation_audit"]["all_nonextra_invocations_exactly_aligned"]
                and ledger["instrumentation_audit"]["baseline_vs_weak_loss_and_gradient_exact"]
                and ledger["instrumentation_audit"]["baseline_vs_strong_loss_and_gradient_exact"]
            ),
            "evidence": f"results/coverage/{architecture}_invocation_ledger.json.gz",
            "counts": ledger["summary"],
            "scientific_gates_still_false": {
                key: value for key, value in ledger["gates"].items() if not value
            },
        })

    payload = {
        "schema": "kernel-analyzer-requested-coverage-completion-audit-v1",
        "status": "COMPLETE_REQUESTED_INFRASTRUCTURE" if all(row["complete"] for row in requirements) else "INCOMPLETE",
        "requirements": requirements,
        "all_requirements_complete": all(row["complete"] for row in requirements),
        "scientific_coverage_status": "PARTIAL_FAIL_CLOSED",
        "claim_boundary": (
            "Completion means the requested ledgers, gap audits, and full-model eager F+B atlases exist and pass their structural gates. "
            "It does not mean Qwen gaps were fabricated closed or that candidate correctness/property induction is complete."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "completion_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
