#!/usr/bin/env python3
"""Build the 15-row Oracle comparison after removing optimizer mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def row_from_level(*, case_id: str, architecture: str, family: str,
                   level: dict[str, Any], source: str,
                   historical_sgd_positive: bool) -> dict[str, Any]:
    prefix = level["prefix"]["16"]
    return {
        "case_id": case_id,
        "architecture": architecture,
        "family": family,
        "optimizer": "AdamW(beta1=0.9,beta2=0.95,eps=1e-8,weight_decay=0)",
        "prefix16_local_A": float(prefix["coherence_amplification"]),
        "full32_local_A": float(level["coherence_amplification"]),
        "full32_sign_flip_upper95": float(level["sign_flip_null"]["upper_95"]),
        "full32_sign_flip_one_sided_p": float(level["sign_flip_null"]["one_sided_p"]),
        "adamw_local_persistent": bool(level["above_sign_flip_95"]),
        "historical_sgd_source_persistent": historical_sgd_positive,
        "prefix16_local_rms": float(prefix["diffusive_scale_l2"]) / 4.0,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "oracle_repair_v3/same_optimizer_oracle_v3.json")
    args = parser.parse_args()
    summary = load(BASE / "consequence_summary.json")
    controls = []
    for item in summary["rows"]:
        raw_path = ROOT / item["source"]
        raw = load(raw_path)
        controls.append(row_from_level(
            case_id=item["case_id"], architecture=item["architecture"],
            family="RESULT_BLIND_SCREEN_CONTROL", level=raw["statistics"]["levels"]["local"],
            source=str(raw_path.relative_to(ROOT)), historical_sgd_positive=False,
        ))
    if len(controls) != 12:
        raise RuntimeError(f"expected all twelve result-blind sampled rows, got {len(controls)}")

    qwen_path = BASE / "oracle_repair_v3/qwen_seq128_adamw_consequence32_with_stages.json"
    phi_path = BASE / "oracle_repair_v3/phi_seq64_adamw_consequence32.json"
    liger_path = BASE / "oracle_repair_v3/liger_adamw_consequence32.json"
    qwen = load(qwen_path); phi = load(phi_path); liger = load(liger_path)
    cases = [
        row_from_level(
            case_id="qwen_seq128_lmhead_dx", architecture="qwen",
            family="HISTORICAL_SGD_HEADLINE_REMEASURED_WITH_ADAMW",
            level=qwen["statistics"]["levels"]["local"], source=str(qwen_path.relative_to(ROOT)),
            historical_sgd_positive=True,
        ),
        row_from_level(
            case_id="phi4_seq64_lmhead_dx", architecture="phi",
            family="HISTORICAL_SGD_HEADLINE_REMEASURED_WITH_ADAMW",
            level=phi["statistics"]["levels"]["local"], source=str(phi_path.relative_to(ROOT)),
            historical_sgd_positive=True,
        ),
        row_from_level(
            case_id="liger_fused_ce_t128", architecture="qwen3-1p7b",
            family="HISTORICAL_SGD_HEADLINE_REMEASURED_WITH_ADAMW",
            level=liger["statistics"]["local_complete_gram"], source=str(liger_path.relative_to(ROOT)),
            historical_sgd_positive=True,
        ),
    ]
    rows = controls + cases
    labels = [bool(row["adamw_local_persistent"]) for row in rows]
    scores = [float(row["prefix16_local_A"]) for row in rows]
    rms = [float(row["prefix16_local_rms"]) for row in rows]
    predicted = [score > 1.0 for score in scores]
    tp = sum(p and y for p, y in zip(predicted, labels)); fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels)); tn = sum(not p and not y for p, y in zip(predicted, labels))
    payload = {
        "schema": "kernel-analyzer-same-optimizer-oracle-v3",
        "status": "COMPLETE_15_ROW_ADAMW_COMPARISON",
        "cohort": {
            "rows": len(rows), "adamw_positive_rows": sum(labels),
            "adamw_negative_rows": len(rows) - sum(labels),
            "result_blind_sample_rows": len(controls),
            "result_blind_sample_adamw_positive_rows": sum(
                bool(row["adamw_local_persistent"]) for row in controls
            ),
            "historical_sgd_positive_rows": sum(bool(row["historical_sgd_source_persistent"]) for row in rows),
            "historical_sgd_rows_remaining_positive_under_adamw": sum(
                bool(row["historical_sgd_source_persistent"]) and bool(row["adamw_local_persistent"])
                for row in rows
            ),
            "optimizer": "AdamW for every row",
        },
        "score": "prefix16 local effective-update coherence under AdamW",
        "label": "full32 local effective-update coherence exceeds its own sign-flip 95% bound under AdamW",
        "comparisons": {
            "prefix16_local_A": {"auroc": auc(labels, scores)},
            "prefix16_local_rms": {"auroc": auc(labels, rms)},
            "frozen_fail_closed_threshold_score_gt_1": {
                "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
                "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            },
        },
        "rows": rows,
        "cohort_correction": "oracle_repair_v3/cohort_correction.json",
        "supersedes_for_performance_claims": "oracle_baselines/frozen_evaluation_v2/comparison_v2.json",
        "why_v2_is_not_a_performance_estimate": "v2 compared three stateless-SGD headline rows with eleven AdamW controls.",
        "claim_boundary": "Retrospective carrier-scale evaluation under one optimizer. All twelve result-blind sampled rows are retained, including the one later found to have mixed local and feedback persistence. No SAFE verdict and no universal unseen-implementation accuracy claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "positives": sum(labels), "auroc": payload["comparisons"]["prefix16_local_A"]["auroc"]}, sort_keys=True))


if __name__ == "__main__":
    main()
