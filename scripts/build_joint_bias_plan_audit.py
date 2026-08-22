#!/usr/bin/env python3
"""Build a claim-safe audit of the current bias/operator/Oracle work plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present(relative: str) -> bool:
    return (ROOT / relative).is_file()


def check(relative: str, status: str, note: str) -> dict:
    path = ROOT / relative
    row = {"artifact": relative, "status": status, "note": note}
    if path.is_file():
        row["sha256"] = sha256(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = [
        check(
            "results/property/joint_bias_formation_v1/phi_three_stage_reference.json",
            "COMPLETE",
            "Phi: output error, parameter-gradient error, and effective-update error on one ordered 32-state reference trajectory.",
        ),
        check(
            "results/property/joint_bias_formation_v1/liger_three_stage_reference.json",
            "COMPLETE",
            "Liger: the same three stages on one tied-embedding carrier.",
        ),
        check(
            "results/property/joint_bias_formation_v1/qwen_three_stage_reference.json",
            "COMPLETE",
            "Qwen: local output error is near diffusive while directionality appears after backward transport.",
        ),
        check(
            "results/property/joint_bias_formation_v1/three_stage_summary.json",
            "COMPLETE",
            "Unified table for the three-stage operator -> gradient -> update comparison.",
        ),
        check(
            "results/property/joint_bias_formation_v1/carrier_distribution/merged_v2/distribution.json",
            "COMPLETE",
            "Phi parameter-carrier distribution: one declared anchor plus eleven outcome-blind positions.",
        ),
        check(
            "results/property/joint_bias_formation_v1/carrier_distribution/merged_v2/loss_unseen_fp32.json",
            "COMPLETE",
            "Unseen-data common FP32 loss for carrier candidate/repair pairs.",
        ),
        check(
            "results/property/joint_bias_formation_v1/four_scale_arms/phi_lmhead_with_masters.json",
            "COMPLETE",
            "Four-arm 32-step reference: operator, seed, data order, and precision.",
        ),
        check(
            "results/property/joint_bias_formation_v1/four_scale_arms/loss_unseen_fp32.json",
            "COMPLETE",
            "Common unseen-data FP32 loss for the four-arm final weights.",
        ),
        check(
            "results/property/joint_bias_formation_v1/four_scale_arms/phi_layer26_summary.json",
            "COMPLETE",
            "Second declared Phi parameter carrier; it is reported separately and does not reproduce the final-norm result.",
        ),
        check(
            "results/property/joint_bias_formation_v1/four_scale_arms/phi_repeated_random_null.json",
            "COMPLETE",
            "Five-seed repeated RMS/support-matched random injection used as the empirical diffusion scale.",
        ),
        check(
            "results/property/joint_bias_formation_v1/oracle_baselines/comparison.json",
            "COMPLETE_RETROSPECTIVE",
            "Frozen 12-row Oracle versus RMS, dtype, and reduction baselines; one local positive.",
        ),
        check(
            "results/property/joint_bias_formation_v1/oracle_baselines/efficiency.json",
            "COMPLETE_RETROSPECTIVE_RUNTIME_UNRECORDED",
            "Measured flag rate, recall, false-positive rate, and AUROC; runtime savings remain explicitly unresolved.",
        ),
        check(
            "results/property/joint_bias_formation_v1/mainline_closure.json",
            "COMPLETE_BOUNDED",
            "Bounded conclusion tying the operator, random-null, second-carrier, and RMS baseline results together.",
        ),
        check(
            "results/property/joint_bias_formation_v1/consequence_summary.json",
            "COMPLETE",
            "Twelve screen-negative live consequence runs; this separates common feedback drift from local source persistence.",
        ),
        check(
            "results/property/joint_bias_formation_v1/heldout_confirmation.json",
            "PARTIAL_BOUNDED",
            "Held-out source-negative confirmation exists, but the full joint predictor remains unresolved.",
        ),
        check(
            "results/property/joint_bias_formation_v1/mu_parity_decomposition.json",
            "PARTIAL_BOUNDED",
            "Saved-P and SiLU response replay exists; generic raw-vector even/odd coverage is not available for every case.",
        ),
        check(
            "results/property/joint_bias_formation_v1/execution_status.json",
            "STALE_LEGACY_STATUS",
            "Legacy status still lists older open items; this audit does not silently relabel those items as complete.",
        ),
    ]

    complete = sum(row["status"] == "COMPLETE" for row in items)
    payload = {
        "schema": "kernel-analyzer-joint-bias-plan-audit-v2",
        "status": "COMPLETE_WITH_EXPLICIT_BOUNDARIES",
        "scope": "bias formation, operator-level error transport, persistence screening, and Oracle efficiency; no new property search",
        "items": items,
        "complete_artifact_count": complete,
        "remaining_boundaries": [
            "The generic raw epsilon/+epsilon/-epsilon replay is not complete for every historical case; missing replay inputs remain UNRESOLVED.",
            "The held-out full joint predictor is unresolved; held-out source-negative evidence is bounded, not universal confirmation.",
            "GPU-hour savings are not claimed because wall-clock accounting was not frozen for every screen row.",
            "All three-stage and four-arm results are carrier-scale or declared-parameter results, not full-parameter training.",
        ],
        "decision": "The current plan covers every experiment needed for the bounded bias+operator+Oracle mainline. It must report the four boundaries above instead of promoting them to a universal property.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "items": len(items), "complete": complete}))


if __name__ == "__main__":
    main()
