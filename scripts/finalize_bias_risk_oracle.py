#!/usr/bin/env python3
"""Freeze the validated fail-closed bias-risk oracle and its evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_oracle_recovery"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def render(result: dict[str, Any]) -> str:
    return f"""# Bias-risk oracle v1

## Decision

The validated object is a **fail-closed multi-witness risk oracle**, not a
single scalar and not a safety certificate. It reports `DIRECTIONAL_RISK` when
any independently defined sufficient witness fires; otherwise it reports
`ABSTAIN`.

The development regression recovered {result['validation']['development_strict_recovered']}/{result['validation']['development_strict_count']} strict formation positives
with zero false-safe decisions. A secondary retrospective lm-head positive was
detected while a real RMSNorm sign-changing control was not. Most importantly,
the frozen prospective moving-frame screen discovered one new case:
`{result['validation']['prospective_new_case']}`. One of three promoted
candidates confirmed on 32 disjoint natural states; neither of the two
sign-changing controls fired.

## Witnesses

1. **Conditional event/source asymmetry.** At a fixed training condition,
   randomized or semantic-orbit repair samples have a non-centered complete
   local/effective response.
2. **Transported directional component.** Either the complete parameter-vector
   population has positive cross-state U-statistic energy, or the same-state
   coefficient `<delta_g, g_repair>/||g_repair||^2` has a nonzero confirmed
   mean. A calibration-frozen analytic/cross-fit projection is an optional
   instance of this channel.
3. **Response rectification.** Exact `+delta_g/-delta_g` perturbations are not
   mapped to opposite effective optimizer updates.

These are three operational witnesses for the two exact antithetic-symmetry
defects: event/pair-mass asymmetry and non-odd downstream response. They do not
use trajectory drift, T4, or SEUP as formation labels.

## New case

For DeepSeek layer-35 attention,

`O = P V`, `dV = P^T dO`, and `dW_v = dV^T H`.

Replacing only the compiled BF16 `dV` BMM output with the FP32-recomputed,
BF16-ABI reference changes the complete `v_proj.weight` gradient. On unseen
states, candidate minus repair has mean relative coefficient
{result['validation']['prospective_new_case_mean']:.6g}, with 95% CI
[{result['validation']['prospective_new_case_interval'][0]:.6g},
{result['validation']['prospective_new_case_interval'][1]:.6g}]. The absolute
gradient direction may rotate with the input; in the repair-gradient moving
frame, the candidate consistently contracts the update component.

This is a new **conditional bias-formation case**. It is not yet a complete
SEUP/trajectory accumulation case and does not imply that all attention BMMs
share the effect.

## Cost and generality

The numerical witnesses add no additional model F+B after a candidate/repair
trace exists: they reduce retained vectors to dot products, Grams, and an
offline optimizer response. For a grouped capture of `N` endpoints, the exact
current runner costs `N + 3` F+B arms per state (`N` repairs plus shared
candidate, reference, and sham), rather than `4N` isolated arms. Four states
are an inexpensive first screen; 16+16 is reserved for promoted cases.

The oracle is implementation-agnostic over any exact F+B boundary with a
matched repair and reachable parameter gradient. Missing repair, zero carrier
reach, discontinuous unsupported paths, or non-replicating evidence produces
`ABSTAIN`, never `SAFE`.

## Claim boundary

This evidence is sufficient to freeze the indicator as a research
risk-discovery oracle. It is not sufficient for certifying arbitrary unseen
kernels as safe or for claiming long-horizon training failure. Those require
coverage-specific repair availability and, for persistence claims, the
separate SEUP/consequence analysis.
"""


def main() -> None:
    recovery = load(BASE / "recovery.json")
    confirmation = load(BASE / "confirmation/result.json")
    audit = recovery["recovery"]
    secondary = recovery["secondary_regression"]
    if not (
        audit["strict_direct_recall"] == 1.0
        and audit["false_safe_count"] == 0
        and secondary["passed"]
        and confirmation["decision"].startswith(
            "MOVING_FRAME_IS_A_VALIDATED_SUFFICIENT_RISK_WITNESS"
        )
    ):
        raise RuntimeError("oracle evidence gates are incomplete")
    new_case = next(
        row for row in confirmation["rows"]
        if row["result"] == "CONFIRMED_DIRECTIONAL_RISK"
    )
    result = {
        "schema": "kernel-analyzer-bias-risk-oracle-v1",
        "status": "FROZEN_VALIDATED_SUFFICIENT_RISK_ORACLE",
        "verdicts": {
            "any_witness_hit": "DIRECTIONAL_RISK",
            "no_witness_or_missing_capability": "ABSTAIN",
            "safe_verdict_supported": False,
        },
        "witness_families": [
            "CONDITIONAL_EVENT_SOURCE_ASYMMETRY",
            "TRANSPORTED_DIRECTIONAL_COMPONENT",
            "EXACT_ANTITHETIC_RESPONSE_RECTIFICATION",
        ],
        "validation": {
            "development_strict_count": audit["strict_positive_count"],
            "development_strict_recovered": int(
                audit["strict_positive_count"] * audit["strict_direct_recall"]
            ),
            "development_false_safe": audit["false_safe_count"],
            "secondary_retrospective_regression_passed": secondary["passed"],
            "prospective_promoted_candidates": confirmation["candidate_count"],
            "prospective_confirmed_candidates": confirmation["candidate_confirmed"],
            "prospective_controls_not_flagged": confirmation["controls_not_flagged"],
            "prospective_new_case": new_case["case_id"],
            "prospective_new_case_mean": new_case["certificate"]["mean_coefficient"],
            "prospective_new_case_interval": new_case["certificate"]["bootstrap_interval"],
        },
        "cost": {
            "extra_model_fb_after_candidate_repair_trace": 0,
            "grouped_exact_capture_per_state": "N_PLUS_3_FB_ARMS",
            "screen_states": 4,
            "confirmation_states": "16_CALIBRATION_PLUS_16_CONFIRMATION",
            "temporary_full_vectors_deleted_after_reduction": True,
        },
        "generality": {
            "requires": [
                "EXACT_FB_BOUNDARY",
                "MATCHED_REPAIR",
                "REACHABLE_DECLARED_PARAMETER_GRADIENT",
            ],
            "operator_name_used": False,
            "model_name_used_by_statistic": False,
            "trajectory_label_used": False,
            "unsupported_behavior": "ABSTAIN",
        },
    }
    (BASE / "oracle_v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (BASE / "oracle.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "new_case": new_case["case_id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
