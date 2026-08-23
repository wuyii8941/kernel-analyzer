#!/usr/bin/env python3
"""Verify the executed priority follow-ups without trusting prose summaries.

The verifier reads the frozen evaluation rows and the saved experiment JSONs.
It fails closed when a row, carrier, random null, loss arm, or optimizer arm is
missing.  It deliberately reports bounded carrier-scale evidence; it does not
promote it to a universal property.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing evidence file: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_masters(masters: dict[str, Any], label: str) -> None:
    for name, metadata in masters.items():
        path = Path(metadata["path"])
        require(path.is_file(), f"{label}: missing saved master {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == metadata["sha256"], f"{label}: digest mismatch for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "priority_followup_audit_v3.json")
    parser.add_argument(
        "--timed-audit",
        type=Path,
        default=Path("/data1/tzh/cache/joint_consequence_timed/full_v4/timed_audit.json"),
    )
    parser.add_argument(
        "--mamba-timed",
        type=Path,
        default=Path("/data1/tzh/cache/joint_consequence_timed/mamba_v1/timed_audit.json"),
    )
    parser.add_argument(
        "--mamba-block",
        type=Path,
        default=BASE / "mamba_timing_blocked_v1.json",
    )
    args = parser.parse_args()

    oracle = load("results/property/joint_bias_formation_v1/oracle_baselines/frozen_evaluation_v2/comparison_v2.json")
    cohort = oracle["cohort"]
    require(cohort["rows"] == 14 and cohort["positive_rows"] == 3, "frozen Oracle cohort is not 14 rows with 3 positives")
    positive_ids = {
        row["case_id"] for row in oracle["rows"] if row.get("label_32step_local_persistent") is True
    }
    require(len(positive_ids) == 3, "positive labels are not three distinct frozen cases")

    manifest = load("results/property/joint_bias_formation_v1/carrier_distribution/manifest_v2.json")
    distribution = load("results/property/joint_bias_formation_v1/carrier_distribution/merged_v2/distribution.json")
    random_distribution = load("results/property/joint_bias_formation_v1/carrier_distribution/random_null_v2/distribution.json")
    require(manifest["carrier_count"] == 12, "carrier manifest does not contain 12 positions")
    require(distribution["status"] == "COMPLETE_FROZEN_12_CARRIER_DISTRIBUTION", "natural carrier distribution incomplete")
    require(len(distribution["rows"]) == 12, "natural carrier distribution row count mismatch")
    require(random_distribution["carrier_count"] == 12, "random null does not cover 12 carriers")
    require(len(random_distribution["random_null_summary"]) == 5, "random null does not contain five seeds")

    final_norm = load("results/property/joint_bias_formation_v1/four_scale_arms/phi_lmhead_with_masters.json")
    second_carrier = load("results/property/joint_bias_formation_v1/four_scale_arms/phi_layer26_post_attention_norm.json")
    require(final_norm["status"] == "COMPLETE" and final_norm["steps"] == 32, "final-norm four-arm run incomplete")
    require(second_carrier["status"] == "COMPLETE" and second_carrier["steps"] == 32, "second-carrier four-arm run incomplete")
    verify_masters(final_norm["final_masters"], "final-norm four-arm")
    require(second_carrier.get("only_declared_parameter_updated") is True, "second-carrier boundary missing")
    require(len(second_carrier.get("records", [])) == 32, "second-carrier four-arm row count mismatch")

    loss = load("results/property/joint_bias_formation_v1/four_scale_arms/loss_unseen_fp32.json")
    random_loss = load("results/property/joint_bias_formation_v1/four_scale_arms/random_null_loss.json")
    require(loss["status"] == "COMPLETE_UNSEEN_FP32_EVALUATION" and len(loss["rows"]) == 4, "four-arm unseen loss incomplete")
    require(random_loss["status"] == "COMPLETE_UNSEEN_FP32_EVALUATION", "repeated-random loss arm incomplete")

    adamw = load("results/property/joint_bias_formation_v1/phi_three_stage_adamw.json")
    require(adamw["status"] == "COMPLETE_ORDERED_32_STATE_COMMON_STATE_ADAMW", "AdamW mapping incomplete")

    consequence = load("results/property/joint_bias_formation_v1/consequence_summary.json")
    consequence_rows = consequence.get("cases", consequence.get("rows", []))
    require(consequence.get("status") == "COMPLETE" and consequence.get("completed_cases") == 12, "scientific consequence summary is not complete for the frozen 12 rows")
    complete_consequence_ids = {str(row.get("case_id")) for row in consequence_rows}
    require(len(complete_consequence_ids) == 12, "scientific consequence summary has duplicate or missing case IDs")

    timed = None
    timed_status = "UNRESOLVED_MISSING_TIMED_12_CASE_AUDIT"
    if args.timed_audit.is_file() and args.mamba_timed.is_file():
        timed = json.loads(args.timed_audit.read_text(encoding="utf-8"))
        mamba = json.loads(args.mamba_timed.read_text(encoding="utf-8"))
        timed_case_ids = {row["case_id"] for row in timed.get("cases", []) if row.get("result_status") == "COMPLETE"}
        mamba_rows = [row for row in mamba.get("cases", []) if row.get("result_status") == "COMPLETE"]
        require(len(mamba_rows) == 1 and mamba_rows[0].get("case_id") == "multishape-backward-cell-0450", "Mamba timed audit is not a single complete case")
        timed_case_ids.add(mamba_rows[0]["case_id"])
        require(len(timed_case_ids) == 12, f"timed consequence coverage is {len(timed_case_ids)}, expected 12")
        timed_status = "COMPLETE_12_CASES_WITH_REAL_OUTPUTS_AND_MAMBA_CERTIFICATE"
    elif args.mamba_block.is_file():
        blocked = json.loads(args.mamba_block.read_text(encoding="utf-8"))
        require(blocked.get("status") == "BLOCKED_AOT_WARMUP", "Mamba block evidence is malformed")
        require(blocked.get("scientific_certificate_status") == "COMPLETE" and blocked.get("scientific_certificate_steps") == 32, "Mamba scientific certificate is not complete")
        timed_status = "BLOCKED_TIMED_MAMBA_AOT_WARMUP_SCIENTIFIC_12_OF_12_COMPLETE"

    payload = {
        "schema": "kernel-analyzer-priority-followup-audit-v3",
        "status": timed_status,
        "executed_checks": {
            "frozen_oracle_rows": cohort["rows"],
            "frozen_oracle_positive_rows": cohort["positive_rows"],
            "positive_case_ids": sorted(positive_ids),
            "carrier_positions": 12,
            "random_null_seeds": 5,
            "four_arm_carriers": [final_norm["updated_parameter"], second_carrier["updated_parameter"]],
            "unseen_fp32_loss_rows": len(loss["rows"]),
            "repeated_random_loss_status": random_loss["status"],
            "adamw_status": adamw["status"],
            "scientific_consequence_rows": len(complete_consequence_ids),
            "timed_consequence_status": timed_status,
        },
        "key_measurements": {
            "final_norm_operator_A": final_norm["arms"]["A_operator"]["coherence_amplification"],
            "second_carrier_operator_A": second_carrier["arms"]["A_operator"]["coherence_amplification"],
            "natural_final_norm_max_carrier_A": distribution["summary"]["maximum_A"],
            "random_null_max_A": max(row["maximum_A"] for row in random_distribution["random_null_summary"].values()),
            "adamw_gradient_A": adamw["stages"]["parameter_gradient_error"]["coherence_curve"][-1]["coherence_amplification"],
            "adamw_update_A": adamw["stages"]["effective_update_error"]["coherence_curve"][-1]["coherence_amplification"],
            "sgd_update_A": load("results/property/joint_bias_formation_v1/phi_three_stage_reference.json")["stages"]["effective_update_error"]["coherence_curve"][-1]["coherence_amplification"],
            "random_loss_gap": random_loss["absolute_loss_gap_random_minus_repair"],
        },
        "claim_boundary": (
            "The three positives are in the same frozen evaluation cohort, but they are bounded headline cases. "
            "The carrier and four-arm measurements are one-parameter or declared-carrier experiments, not full-model training. "
            "A universal all-operator property is not claimed; missing raw response replay and a full joint held-out predictor remain fail-closed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["executed_checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
