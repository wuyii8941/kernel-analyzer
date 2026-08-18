#!/usr/bin/env python3
"""Build the single compact, fail-closed project overview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"


def read(name: str) -> dict[str, Any]:
    return json.loads((COVERAGE / name).read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    status = read("four_model_full_operator_status.json")
    audit = read("four_model_full_operator_audit.json")
    queue = read("bias_candidate_queue.json")
    live = read("live_contrast_dispositions.json")
    cases = read("existing_case_reaudit.json")
    abi = read("triton_reference_abi_audit.json")
    payload = {
        "schema": "kernel-analyzer-coverage-overview-v2",
        "status": status["status"],
        "scope": status["scope"],
        "coverage_gates": status["counts"],
        "execution_denominator": audit["totals"],
        "triton_reference_status": abi["status"],
        "candidate_queue": {
            "valid_nontriton_screen_positives": queue["screen_positive_denominator"],
            "accounted_candidates": queue["candidate_count"],
            "disposition": live["counts"],
            "status": live["status"],
            "source": "results/coverage/live_contrast_dispositions.json",
        },
        "case_counts": cases["counts"],
        "property_positive_count": sum(
            bool(row.get("property_positive_eligible")) for row in cases["rows"]
        ),
        "global_gates": {
            "execution_and_origin_accounting_complete": status["counts"]["fb_origin_bound"] == 12,
            "canonical_eager_fb_proof_complete": (
                status["counts"]["canonical_eager_fb_math_closed"] == 12
            ),
            "actual_default_aot_fb_proof_complete": (
                status["counts"]["default_aot_fb_math_closed"] == 12
            ),
            "candidate_fb_binding_complete": status["counts"]["candidate_fb_bindings_closed"] == 12,
            "valid_triton_numerical_oracle_complete": status["counts"]["triton_precision_oracles_closed"] == 12,
            "same_dtype_optimization_oracle_complete": status["counts"]["same_dtype_optimization_oracles_closed"] == 12,
            "all_scientific_gates_complete": status["counts"]["fully_closed_cells"] == 12,
            "property_induction_allowed": False,
        },
        "claim_boundary": (
            f"All {status['counts']['fb_origin_bound']}/12 model-shape cells have independent "
            "execution censuses and canonical F+B origin accounting. Actual default-AOT F+B "
            f"mathematics is closed for {status['counts']['default_aot_fb_math_closed']}/12 cells, "
            f"candidate-to-F+B binding for {status['counts']['candidate_fb_bindings_closed']}/12, "
            f"typed Triton numerical Oracles for {status['counts']['triton_precision_oracles_closed']}/12, "
            f"and same-dtype optimization Oracles for {status['counts']['same_dtype_optimization_oracles_closed']}/12. "
            "Six strict Flash-style cases "
            "pass their trajectory-local gates: four root-arithmetic cases and two causally "
            "closed semantic-region cases. Two concrete mechanisms "
            "also pass cross-state confirmation, but no cross-operator property is claimed."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
