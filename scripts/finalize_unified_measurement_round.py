#!/usr/bin/env python3
"""Collect the frozen anchor, intervention, validation, and held-out results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/unified_measurement_round_v1"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def stage_rows(anchor: dict) -> list[dict]:
    rows = []
    for stage, profile in anchor["stages"].items():
        rows.append({
            "stage": stage,
            "additive_effect": profile["additive_heldout_effect"],
            "repair_aligned_effect": profile["aligned_effect"],
            "residual_direction_effect": profile["orthogonal_heldout_effect"],
        })
    return rows


def main() -> None:
    anchors = load("results/property/anchor_unified_profiles_v1/summary.json")
    intervention = load(
        "results/property/anchor_unified_profiles_v1/phi_sr_repair_path.json"
    )
    synthetic = load(
        "results/property/anchor_unified_profiles_v1/synthetic_validation.json"
    )
    heldout = load(
        "results/property/bias_oracle_recovery/confirmation/result.json"
    )

    deepseek = next(
        row for row in heldout["rows"]
        if row["case_id"] == "deepseek8b_seq64_l35_attention_dv"
    )
    natural = intervention["metrics"]["natural"]
    sr = [intervention["metrics"][f"sr_{i}"] for i in range(4)]
    result = {
        "schema": "kernel-analyzer-unified-measurement-round-v1",
        "status": "COMPLETE",
        "protocol_frozen_before_anchor_results": True,
        "anchor_family_holm_tests": 18,
        "anchors": {
            case_id: stage_rows(case)
            for case_id, case in anchors["cases"].items()
        },
        "interventions": {
            "phi_stochastic_rounding_same_adamw_protocol": {
                "natural_A": natural["coherence_amplification"],
                "natural_null_upper_95": natural["sign_flip_null"]["upper_95"],
                "natural_p": natural["sign_flip_null"]["one_sided_p"],
                "sham_exactly_reproduces_natural": (
                    intervention["metrics"]["sham"]["coherence_amplification"]
                    == natural["coherence_amplification"]
                ),
                "stochastic_rounding_A": [
                    row["coherence_amplification"] for row in sr
                ],
                "stochastic_rounding_null_upper_95": [
                    row["sign_flip_null"]["upper_95"] for row in sr
                ],
                "stochastic_rounding_energy_over_natural": [
                    row["energy"] / natural["energy"] for row in sr
                ],
                "claim": (
                    "All four stochastic-rounding repeats return to their own "
                    "random-cancellation range. Three repeats preserve natural-arm "
                    "update-error energy within 1.6%; the fourth has half the energy."
                ),
            },
            "liger_accumulator": {
                "candidate": "BF16_CHUNK_ACCUMULATION",
                "repair": "FP32_ACCUMULATION_WITH_MATCHED_BF16_ABI",
                "measured_in_anchor_profile": True,
                "claim": (
                    "The exact candidate-repair contrast identifies accumulator "
                    "precision as the changed source boundary; it does not by itself "
                    "estimate a general reduction-orbit predictor."
                ),
            },
        },
        "synthetic_validation": {
            "status": synthetic["status"],
            "scenario_count": len(synthetic["rows"]),
            "results": synthetic["rows"],
            "claim_boundary": synthetic["claim_boundary"],
        },
        "prospective_heldout": {
            "case_id": deepseek["case_id"],
            "stage": "PARAMETER_GRADIENT",
            "optimizer_mapping": "STATELESS_SGD_FP32_MASTER",
            "mean_relative_effect": deepseek["certificate"]["mean_coefficient"],
            "confidence_interval_95": deepseek["certificate"]["bootstrap_interval"],
            "confirmed_candidates": heldout["candidate_confirmed"],
            "candidate_count": heldout["candidate_count"],
            "controls_not_flagged": heldout["controls_not_flagged"],
            "control_count": heldout["control_count"],
            "protocol_was_frozen": not heldout["threshold_changed_after_measurement"],
            "claim_boundary": (
                "This is a frozen unseen-state gradient-stage confirmation. It is not "
                "an AdamW update result or a long-run consequence result."
            ),
        },
        "overall_boundary": (
            "The round validates one shared measurement protocol for two anchors, one "
            "same-AdamW causal intervention, synthetic behavior, and an existing frozen "
            "held-out gradient result. It is not a universal kernel safety certificate."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(OUT / "summary.json")}))


if __name__ == "__main__":
    main()
