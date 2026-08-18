#!/usr/bin/env python3
"""Build the bounded conclusion for the declared current-model matrix."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
COVERAGE = ROOT / "results/coverage"


def read(name: str) -> dict:
    return json.loads((FINAL / name).read_text())


def main() -> None:
    matrix = read("implementation_matrix.json")
    invocation = read("invocation_atlas.json")
    carrier_census = read("carrier_census.json")
    carrier = read("priority_carrier_replay.json")
    flash = read("flash_control.json")
    bank = read("natural_bank.json")
    structured = read("structured_carrier_trigger.json")
    confirmation = read("structured_carrier_confirmation.json")
    with gzip.open(COVERAGE / "qwen_invocation_ledger.json.gz", "rt") as handle:
        coverage_ledger = json.load(handle)
    case_audit = json.loads((COVERAGE / "existing_case_reaudit.json").read_text())
    output = {
        "schema": "kernel-analyzer-bounded-search-conclusion-v1",
        "scope": {
            "model": "Qwen3-1.7B",
            "checkpoint_steps": [row["step"] for row in bank["checkpoints"]],
            "declared_matrix_cells_complete": bool(matrix["declared_matrix_cells_complete"]),
            "invocation_coverage_status": coverage_ledger["status"],
            "candidate_configs": [
                "SDPA math",
                "CUDA flash-SDPA",
                "Inductor/Triton BF16, FP16, strict FP32, TF32 at seq64/128/256",
                "Liger fused-linear cross-entropy",
            ],
        },
        "positive_controls": {
            "flash_pipeline": flash["positive_control"]["v_only_live_weight"]["verdict"],
            "project_natural_cases": [
                "seq128 lm_head input-gradient MM",
                "Liger fused-linear cross-entropy dW accumulation",
                "Phi-4 seq64 lm_head input-gradient MM",
                "layer-23 q_proj closed S_bwd attention-state semantic region",
                "Mamba seq64 layer-0 input-projection local MM accumulation",
            ],
            "strict_flash_style_case_count": case_audit["counts"]["strict_flash_style_cases"],
            "strict_root_arithmetic_op_case_count": case_audit["counts"]["strict_root_arithmetic_op_pass"],
            "strict_semantic_region_case_count": case_audit["counts"]["strict_semantic_region_pass"],
            "unresolved_composite_carrier_case_count": case_audit["counts"]["composite_carrier_pass"],
            "rejected_historical_case_count": case_audit["counts"]["rejected_by_direction_gate"],
            "cross_state_concrete_mechanism_passes": case_audit["counts"]
            ["cross_state_concrete_mechanism_passes"],
        },
        "difference_evidence": {
            "closed_changed_fbv_units": invocation["denominator"]["changed_fbv_units"],
            "real_changed_sites": invocation["denominator"]["real_changed_sites"],
            "excluded_nonclosed_changed_units": invocation["denominator"]["excluded_nonclosed_changed_units"],
            "priority_chains": carrier["chain_count"],
            "priority_screen_verdict": carrier["verdict"],
            "all_parameter_carrier_census": carrier_census["denominator"],
            "complete_structured_screen": structured["coverage"],
            "structured_discovery_triggers": structured["trigger_count"],
            "independently_confirmed_structured_carriers": confirmation["confirmed_count"],
            "confirmed_structured_parameters": confirmation["confirmed_parameters"],
        },
        "bounded_conclusion": (
            "Within the measured Qwen3-1.7B implementation/state cells, finite "
            "operator-level numerical differences are common, while a coherent "
            "weight-gradient carrier that persists across evolving checkpoints is "
            "rare. A complete coordinate-partition screen and subsequent exact "
            "interventions close one head-local q_proj tile at the actual attention-state "
            "S_bwd semantic boundary (the trajectory uses a conservative S_bwd/K repair). It is "
            "a strict semantic-region case, retained separately "
            "from strict root-arithmetic operator cases and from single-kernel property labels."
        ),
        "not_supported": [
            "a universal all-operator safety claim",
            "cross-model or cross-architecture generalization",
            "a property predictor trained from four concrete cases when one is not a single-kernel label",
            "a claim that unresolved/nonfinite boundary rows are numerically safe",
        ],
        "candidate_blind": True,
        "natural_bias_case_added_by_priority_screen": carrier["natural_bias_case_added"],
        "property_claim": False,
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "search_conclusion.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "coverage_status": output["scope"]["invocation_coverage_status"]}))


if __name__ == "__main__":
    main()
