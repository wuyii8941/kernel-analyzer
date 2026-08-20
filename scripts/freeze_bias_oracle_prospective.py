#!/usr/bin/env python3
"""Freeze a prospective moving-frame regression before measuring its values."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/property/bias_oracle_recovery/prospective"

SELECTIONS = {
    "qwen": (
        "qwen_seq64_lm_head_dx",
        "qwen_seq64_l27_q_norm_vjp",
        "qwen_seq64_l23_softmax_vjp",
        "qwen_seq64_l23_attention_dq",
    ),
    "deepseek8b": (
        "deepseek8b_seq64_ce_dlogits",
        "deepseek8b_seq64_final_norm_vjp",
        "deepseek8b_seq64_l35_softmax_vjp",
        "deepseek8b_seq64_l35_attention_dv",
    ),
}


def load_cases(name: str) -> list[dict[str, object]]:
    source = (
        ROOT
        / "results/property/bias_formation/hotspot_search"
        / f"{name}_seq64_capture_plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    by_id = {str(row["case_id"]): row for row in payload["cases"]}
    missing = set(SELECTIONS[name]) - set(by_id)
    if missing:
        raise RuntimeError(f"frozen semantic cases are missing: {sorted(missing)}")
    return [by_id[case_id] for case_id in SELECTIONS[name]]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model in SELECTIONS:
        (OUTPUT / f"{model}_plan.json").write_text(
            json.dumps({"cases": load_cases(model)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    protocol = {
        "schema": "kernel-analyzer-prospective-bias-witness-regression-v1",
        "status": "FROZEN_BEFORE_REFERENCE_RELATIVE_VALUES",
        "selection": {
            "uses_historical_bias_verdict": False,
            "principle": "TRAINING_SEMANTIC_BOTTLENECK",
            "models": list(SELECTIONS),
            "case_ids": [case for rows in SELECTIONS.values() for case in rows],
            "semantic_families": [
                "LOSS_HEAD_OR_CE_BACKWARD",
                "NORMALIZATION_BACKWARD",
                "ATTENTION_SOFTMAX_BACKWARD",
                "ATTENTION_MATRIX_BACKWARD",
            ],
        },
        "measurement": {
            "unit": "COMPLETE_FORWARD_PLUS_ACTUAL_BACKWARD_WITH_EXACT_ENDPOINT_REPAIR",
            "calibration_common_states": 16,
            "confirmation_common_states": 16,
            "open_loop": True,
            "candidate_and_repair_share_pre_state": True,
            "trajectory_labels_used": False,
        },
        "frozen_witnesses": {
            "population": {
                "statistic": "COMPLETE_VECTOR_CROSS_STATE_U_STATISTIC",
                "strict_hit": "BIASED_IN_CALIBRATION_AND_CONFIRMATION",
            },
            "moving_frame": {
                "statistic": "MEAN_<DELTA_G,G_REPAIR>/||G_REPAIR||^2",
                "bootstrap_draws": 4000,
                "minimum_absolute_mean_coefficient": 1e-5,
                "strict_hit": (
                    "DIRECTIONAL_RISK_IN_BOTH_PARTITIONS_WITH_SAME_MEAN_SIGN"
                ),
            },
        },
        "decision": {
            "risk": "EITHER_FROZEN_WITNESS_STRICT_HIT",
            "miss": "UNRESOLVED_NOT_SAFE",
            "threshold_changes_after_measurement": "FORBIDDEN",
            "scientific_case_requires": (
                "EXACT_FB_BOUNDARY_PLUS_REPLICATED_WITNESS_PLUS_MATCHED_REPAIR"
            ),
        },
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "cases": 8}, sort_keys=True))


if __name__ == "__main__":
    main()
