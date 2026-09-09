#!/usr/bin/env python3
"""Audit completion of the bounded training numerical analysis plan."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/training_numerical_analysis_v1"


def _load(relative: str) -> dict:
    return json.loads((BASE / relative).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-v1", action="store_true",
                        help="Reproduce only the old bounded bookkeeping audit, not completion of the research plan.")
    args = parser.parse_args()
    if not args.historical_v1:
        from scripts.summarize_training_numerical_v2 import build_progress
        progress = build_progress()
        # Re-read current evidence rather than treating an earlier COMPLETE
        # bookkeeping label as proof of the new research plan's completion.
        print(json.dumps(progress, indent=2))
        raise SystemExit(0 if progress["status"] == "COMPLETE_FULL_RESEARCH_PLAN" else 1)
    checks = {
        "method_validation": _load("synthetic_validation.json").get("status") == "PASS",
        "conventional_candidate_actual_write": (
            _load("recomputed/phi_parameter_write.json").get("measurement_status") == "VALID"
        ),
        "triton_normalization_actual_write": (
            _load("recomputed/deepseek_norm_parameter_write.json").get("measurement_status") == "VALID"
        ),
        "triton_attention_actual_write": (
            _load("recomputed/deepseek_attn_parameter_write.json").get("measurement_status") == "VALID"
        ),
        "mixed_liger_actual_write": (
            _load("recomputed/liger_parameter_write.json").get("measurement_status") == "VALID"
        ),
        "paired_training_validation": _load("training_utility_summary.json").get("status") == "COMPLETE",
        "selected_gemma_case_retained": (
            _load("status/gemma4_text128_scan_0037.json").get("replacement_used") is False
        ),
        "selected_llama_case_retained": (
            _load("status/llama32_text128_scan_0000.json").get("replacement_used") is False
        ),
    }
    unresolved = [
        _load("status/gemma4_text128_scan_0037.json"),
        _load("status/llama32_text128_scan_0000.json"),
    ]
    payload = {
        "schema": "kernel-analyzer-training-numerical-analysis-completion-v1",
        "status": (
            "COMPLETE_WITH_DECLARED_ABSTENTIONS" if all(checks.values()) else "INCOMPLETE"
        ),
        "checks": checks,
        "declared_abstentions": unresolved,
        "claim_boundary": (
            "Completion means every task in the bounded protocol has a valid result or "
            "a retained execution reason. It does not convert the two unavailable "
            "held-out Triton cases into negative results or prove cross-model universality."
        ),
    }
    output = BASE / "completion_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit("training numerical analysis plan is incomplete")


if __name__ == "__main__":
    main()
