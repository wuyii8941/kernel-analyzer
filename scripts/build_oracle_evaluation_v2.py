#!/usr/bin/env python3
"""Build the corrected frozen Oracle evaluation cohort.

The first comparison table had twelve sampled controls but only one weak
mixed local/feedback positive row.  This tool keeps the eleven original
controls, retains that legacy row in an audit-only exclusion list, and adds
the three predeclared source-persistence headline cases.  The score is always the
16-step effective-update coherence score; no 32-step result is used to choose
the cohort or to tune the threshold.

The resulting table is an evaluation table, not a claim of universal Oracle
accuracy.  The three added rows come from the previously frozen headline-case
manifest and are labelled before the statistics are recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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
    wins = math.fsum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positive for n in negative
    )
    return wins / (len(positive) * len(negative))


def bootstrap_auc(labels: list[bool], scores: list[float], *, seed: int,
                  draws: int = 10_000) -> list[float] | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        pp = [rng.choice(positive) for _ in positive]
        nn = [rng.choice(negative) for _ in negative]
        values.append(auc([True] * len(pp) + [False] * len(nn), pp + nn))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def operating_point(labels: list[bool], scores: list[float], threshold: float) -> dict[str, Any]:
    predicted = [score > threshold for score in scores]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    tn = sum(not p and not y for p, y in zip(predicted, labels))
    return {
        "threshold_rule": f"score>{threshold}", "flagged": tp + fp,
        "total": len(labels), "flag_rate": (tp + fp) / len(labels),
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "recall": tp / max(tp + fn, 1),
        "miss_rate": fn / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def horizon(stage: dict[str, Any], value: int) -> dict[str, float]:
    rows = [row for row in stage["coherence_curve"] if int(row["horizon"]) == value]
    if len(rows) != 1:
        raise RuntimeError(f"expected one horizon={value}, found {len(rows)}")
    row = rows[0]
    return {key: float(row[key]) for key in ("coherence_amplification", "path_rms_l2", "resultant_l2")}


def add_headline_rows() -> list[dict[str, Any]]:
    # This list is frozen from source_persistence_reclassification.json.  It is
    # deliberately explicit so a later statistic cannot silently change the
    # cohort membership.
    specs = [
        ("liger_fused_ce_t128", "liger_three_stage_reference.json"),
        ("phi4_seq64_lmhead_dx", "phi_three_stage_reference.json"),
        ("qwen_seq256_lmhead_dx", "qwen_three_stage_reference.json"),
    ]
    rows = []
    for case_id, filename in specs:
        payload = load(BASE / filename)
        if payload["case_id"] != case_id or payload["status"] != "COMPLETE_ORDERED_32_STATE_REFERENCE":
            raise RuntimeError(f"headline artifact mismatch for {case_id}")
        score = horizon(payload["stages"]["effective_update_error"], 16)
        rms = score["path_rms_l2"] / math.sqrt(16.0)
        rows.append({
            "case_id": case_id,
            "architecture": {"liger_fused_ce_t128": "qwen3-1p7b", "phi4_seq64_lmhead_dx": "phi", "qwen_seq256_lmhead_dx": "qwen"}[case_id],
            "family": "SOURCE_OR_TRANSPORT_PERSISTENT_HEADLINE",
            "label_32step_local_persistent": True,
            "label_source": "predeclared_mainline_closure_headline_cases",
            "evaluation_cohort": "headline_positive_added_before_v2_statistics",
            "oracle_prefix16_effective_update_A": score["coherence_amplification"],
            "oracle_prefix16_local_A": score["coherence_amplification"],
            "local_effective_update_rms": rms,
            "candidate_dtype_is_bfloat16": 1.0,
            "reduction_extent": None,
            "reduction_binding_method": "NOT_BOUND_FOR_HEADLINE_COHORT",
            "source": f"results/property/joint_bias_formation_v1/{filename}",
            "protocol_note": payload["claim_boundary"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "oracle_baselines/frozen_evaluation_v2")
    args = parser.parse_args()
    old = load(BASE / "oracle_baselines/comparison.json")
    rows = []
    excluded_legacy_rows = []
    for row in old["rows"]:
        # The old 12-row table contains one weak mixed local/feedback positive
        # (Phi cell 0543).  It is not one of the three frozen
        # source-persistence headline cases.  Keep it for auditability, but do
        # not mix that different label with the new source-persistence cohort.
        if bool(row["label_32step_local_persistent"]):
            legacy = dict(row)
            legacy["excluded_reason"] = "legacy_mixed_local_feedback_positive_not_source_headline"
            excluded_legacy_rows.append(legacy)
            continue
        copied = dict(row)
        copied["label_source"] = "frozen_residual_nonzero_parameter_reachable_12_row_cohort"
        copied["evaluation_cohort"] = "original_control_rows"
        copied["oracle_prefix16_effective_update_A"] = float(row["oracle_prefix16_local_A"])
        rows.append(copied)
    existing = {row["case_id"] for row in rows}
    for row in add_headline_rows():
        if row["case_id"] in existing:
            raise RuntimeError(f"duplicate case in frozen cohort: {row['case_id']}")
        rows.append(row)
    if len(rows) != 14 or sum(bool(row["label_32step_local_persistent"]) for row in rows) != 3:
        raise RuntimeError("unexpected frozen cohort cardinality")
    labels = [bool(row["label_32step_local_persistent"]) for row in rows]
    score = [float(row["oracle_prefix16_effective_update_A"]) for row in rows]
    comparisons = {
        "prefix16_effective_update_persistence_oracle": {
            "auroc": auc(labels, score),
            "stratified_bootstrap_95": bootstrap_auc(labels, score, seed=20260823),
            "danger_orientation": "larger_score_is_riskier",
            "threshold": operating_point(labels, score, 1.0),
        },
        "local_effective_update_rms": {
            "auroc": auc(labels, [float(row["local_effective_update_rms"]) for row in rows]),
            "stratified_bootstrap_95": bootstrap_auc(labels, [float(row["local_effective_update_rms"]) for row in rows], seed=20260824),
            "danger_orientation": "larger_score_is_riskier",
        },
        "dtype_is_bfloat16": {
            "auroc": auc(labels, [float(row["candidate_dtype_is_bfloat16"]) for row in rows]),
            "stratified_bootstrap_95": bootstrap_auc(labels, [float(row["candidate_dtype_is_bfloat16"]) for row in rows], seed=20260825),
            "danger_orientation": "larger_score_is_riskier",
        },
    }
    reduction_rows = [row for row in rows if row.get("reduction_extent") is not None]
    reduction_labels = [bool(row["label_32step_local_persistent"]) for row in reduction_rows]
    reduction_scores = [float(row["reduction_extent"]) for row in reduction_rows]
    comparisons["compiled_reduction_extent"] = {
        "auroc": auc(reduction_labels, reduction_scores),
        "stratified_bootstrap_95": bootstrap_auc(reduction_labels, reduction_scores, seed=20260826),
        "danger_orientation": "larger_score_is_riskier",
        "available_rows": len(reduction_rows),
        "excluded_rows": len(rows) - len(reduction_rows),
    }
    manifest = {
        "schema": "kernel-analyzer-oracle-evaluation-manifest-v2",
        "status": "FROZEN_14_ROW_EVALUATION_WITH_THREE_PREDECLARED_HEADLINE_CASES",
        "source_control_table": "oracle_baselines/comparison.json",
        "headline_positive_cases": ["liger_fused_ce_t128", "phi4_seq64_lmhead_dx", "qwen_seq256_lmhead_dx"],
        "headline_positive_selection": "copied from frozen mainline_closure/source_persistence_reclassification before v2 statistics",
        "rows": len(rows),
        "positive_rows": sum(labels),
        "added_headline_rows": 3,
        "excluded_legacy_rows": len(excluded_legacy_rows),
        "threshold": "score>1.0, frozen from prior diffusive boundary rule",
        "score_definition": "16-step effective-update coherence on the same declared one-parameter/carrier protocol where retained; inherited control rows use their frozen 16-step local effective-update score.",
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema": "kernel-analyzer-oracle-baseline-comparison-v2",
        "status": "COMPLETE_FROZEN_14_ROW_COMPARISON_3_POSITIVE_HEADLINES",
        "cohort": {
            "rows": len(rows), "positive_rows": sum(labels), "negative_rows": len(rows) - sum(labels),
            "added_headline_positive_rows": 3,
            "excluded_legacy_rows": len(excluded_legacy_rows),
            "note": "The original 12-row control table's mixed local/feedback positive is retained as an excluded audit row; the active cohort has the three source-persistence headline cases and eleven original negatives.",
        },
        "manifest": "results/property/joint_bias_formation_v1/oracle_baselines/frozen_evaluation_v2/manifest.json",
        "comparisons": comparisons,
        "rows": rows,
        "excluded_rows": excluded_legacy_rows,
        "claim_boundary": "Retrospective frozen evaluation of the declared carrier-scale cohort. It is not universal accuracy and reduction baselines are reported only where exact bindings exist.",
    }
    (args.output_dir / "comparison_v2.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "comparison_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "architecture", "label_32step_local_persistent", "label_source", "oracle_prefix16_effective_update_A", "local_effective_update_rms", "reduction_extent"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "positive_rows": sum(labels), "auroc": comparisons["prefix16_effective_update_persistence_oracle"]["auroc"]}, sort_keys=True))


if __name__ == "__main__":
    main()
