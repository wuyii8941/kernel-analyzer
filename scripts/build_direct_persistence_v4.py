#!/usr/bin/env python3
"""Build the auditable Direct Persistence Screen v4 development package.

This command only consumes already committed JSON artifacts.  It deliberately
does not invent per-step vectors: when an attribution requires raw vectors that
were not saved, the output records ``ABSTAIN`` and the missing evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/property/joint_bias_formation_v1/oracle_repair_v3"
DEFAULT_OUT = ROOT / "results/property/direct_persistence_v4"

PREDECLARED = {
    "liger_fused_ce_t128",
    "phi4_seq64_lmhead_dx",
    "qwen_seq128_lmhead_dx",
}
UNREPLICATED_CANDIDATE = "multishape-backward-cell-0543"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def holm_adjust(rows: Iterable[dict[str, Any]], p_key: str) -> dict[str, dict[str, Any]]:
    """Return Holm family-wise adjusted p-values and decisions."""

    items = sorted(
        ((str(row["case_id"]), float(row[p_key])) for row in rows),
        key=lambda item: item[1],
    )
    count = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (case_id, p_value) in enumerate(items):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[case_id] = running
    return {
        case_id: {
            "raw_p": p_value,
            "holm_adjusted_p": adjusted[case_id],
            "holm_reject_alpha_0_05": adjusted[case_id] <= 0.05,
        }
        for case_id, p_value in items
    }


def bh_adjust(rows: Iterable[dict[str, Any]], p_key: str) -> dict[str, float]:
    items = sorted(
        ((str(row["case_id"]), float(row[p_key])) for row in rows),
        key=lambda item: item[1],
    )
    count = len(items)
    values = [0.0] * count
    running = 1.0
    for index in range(count - 1, -1, -1):
        running = min(running, items[index][1] * count / (index + 1))
        values[index] = min(1.0, running)
    return {case_id: values[index] for index, (case_id, _) in enumerate(items)}


def exact_binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> list[float] | None:
    """Clopper--Pearson interval, with a fail-closed dependency boundary."""

    if trials <= 0 or successes < 0 or successes > trials:
        return None
    try:
        from scipy.stats import beta  # type: ignore
    except Exception:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [lower, upper]


def bootstrap_auc(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    score_key: str,
    draws: int = 5000,
    seed: int = 20260824,
) -> list[float] | None:
    positives = [row for row in rows if bool(row[label_key])]
    negatives = [row for row in rows if not bool(row[label_key])]
    if not positives or not negatives:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        positive_sample = [rng.choice(positives) for _ in positives]
        negative_sample = [rng.choice(negatives) for _ in negatives]
        sample = positive_sample + negative_sample
        samples.append(auc([bool(row[label_key]) for row in sample], [float(row[score_key]) for row in sample]) or 0.0)
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def threshold_summary(rows: list[dict[str, Any]], score_key: str, threshold: float = 1.0) -> dict[str, Any]:
    predicted = [float(row[score_key]) > threshold for row in rows]
    actual = [bool(row["confirmed_label"]) for row in rows]
    tp = sum(pred and label for pred, label in zip(predicted, actual))
    fp = sum(pred and not label for pred, label in zip(predicted, actual))
    fn = sum(not pred and label for pred, label in zip(predicted, actual))
    tn = sum(not pred and not label for pred, label in zip(predicted, actual))
    return {
        "rule": f"{score_key}>{threshold}",
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_exact_95": exact_binomial_interval(tp, tp + fn),
        "precision_exact_95": exact_binomial_interval(tp, tp + fp),
    }


def pairwise_inversions(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    positives = [row for row in rows if bool(row["confirmed_label"])]
    negatives = [row for row in rows if not bool(row["confirmed_label"])]
    inversions = []
    ties = []
    for positive in positives:
        for negative in negatives:
            p_score = float(positive[score_key])
            n_score = float(negative[score_key])
            if p_score < n_score:
                inversions.append([positive["case_id"], negative["case_id"]])
            elif p_score == n_score:
                ties.append([positive["case_id"], negative["case_id"]])
    return {
        "positive_negative_pairs": len(positives) * len(negatives),
        "strict_inversions": len(inversions),
        "ties": len(ties),
        "inversion_pairs": inversions,
        "tie_pairs": ties,
    }


def metrics_view(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    null_scores = [row.get("full32_null_excess") for row in rows]
    return {
        "name": name,
        "rows": len(rows),
        "positives": sum(bool(row["confirmed_label"]) for row in rows),
        "negatives": sum(not bool(row["confirmed_label"]) for row in rows),
        "directionality_auc": auc(
            [bool(row["confirmed_label"]) for row in rows],
            [float(row["prefix16_local_A"]) for row in rows],
        ),
        "update_rms_auc": auc(
            [bool(row["confirmed_label"]) for row in rows],
            [float(row["prefix16_local_rms"]) for row in rows],
        ),
        "null_normalized_full32_auc": auc(
            [bool(row["confirmed_label"]) for row in rows],
            [float(score) for score in null_scores],
        ) if all(score is not None for score in null_scores) else None,
        "directionality_auc_bootstrap_95": bootstrap_auc(
            rows, label_key="confirmed_label", score_key="prefix16_local_A"
        ),
        "threshold": threshold_summary(rows, "prefix16_local_A"),
        "pairwise_ranking": pairwise_inversions(rows, "prefix16_local_A"),
        "leave_one_positive_out": [
            {
                "removed_case_id": row["case_id"],
                "directionality_auc": auc(
                    [bool(item["confirmed_label"]) for item in rows if item is not row],
                    [float(item["prefix16_local_A"]) for item in rows if item is not row],
                ),
            }
            for row in rows
            if bool(row["confirmed_label"])
        ],
    }


def get_level(document: dict[str, Any], name: str) -> dict[str, Any]:
    if "levels" in document.get("statistics", {}):
        return document["statistics"]["levels"][name]
    stream_name = {"local": "local_stream", "feedback": "feedback_stream", "actual": "actual_stream"}[name]
    return document["statistics"][stream_name]


def get_null_level(document: dict[str, Any], name: str) -> dict[str, Any]:
    """Use the exact complete Gram when the stream summary lacks a null."""

    if name == "local" and "local_complete_gram" in document.get("statistics", {}):
        return document["statistics"]["local_complete_gram"]
    return get_level(document, name)


def get_cosines(document: dict[str, Any], values: dict[str, float]) -> dict[str, float]:
    exported = document.get("statistics", {}).get("resultant_cosines", {})
    if exported:
        return {
            "local_actual": float(exported["local__actual"]),
            "feedback_actual": float(exported["feedback__actual"]),
            "local_feedback": float(exported["local__feedback"]),
        }
    local = values["local_resultant"]
    feedback = values["feedback_resultant"]
    actual = values["actual_resultant"]
    if actual <= 0.0 or local <= 0.0 or feedback <= 0.0:
        raise ValueError("cannot derive resultant cosines from zero vectors")
    local_feedback_dot = (actual * actual - local * local - feedback * feedback) / 2.0
    local_actual_dot = local * local + local_feedback_dot
    feedback_actual_dot = feedback * feedback + local_feedback_dot
    return {
        "local_actual": local_actual_dot / (local * actual),
        "feedback_actual": feedback_actual_dot / (feedback * actual),
        "local_feedback": local_feedback_dot / (local * feedback),
    }


def contribution_row(case_id: str, path: Path) -> dict[str, Any]:
    document = load(path)
    levels = {name: get_level(document, name) for name in ("local", "feedback", "actual")}
    values: dict[str, float] = {}
    for name, level in levels.items():
        values[f"{name}_resultant"] = float(level["resultant_l2"])
        values[f"{name}_path_energy"] = float(level.get("energy", level["diffusive_scale_l2"] ** 2))
        values[f"{name}_path_l2"] = float(level["path_l2"])
        values[f"{name}_A32"] = float(level["coherence_amplification"])
        values[f"{name}_A16"] = float(level["prefix"]["16"]["coherence_amplification"])
    cosines = get_cosines(document, values)
    actual = values["actual_resultant"]
    if actual <= 0.0:
        return {
            "case_id": case_id,
            "status": "ABSTAIN_ZERO_ACTUAL_RESULTANT",
            "source": str(path.relative_to(ROOT)),
        }
    signed_local = values["local_resultant"] * cosines["local_actual"] / actual
    signed_feedback = values["feedback_resultant"] * cosines["feedback_actual"] / actual
    signed_recurrence = 1.0 - signed_local - signed_feedback
    recurrence = document.get("global_recurrence_residual", {})
    recurrence_norm = float(recurrence.get("norm", 0.0))
    recurrence_relative = float(document.get("max_recurrence_relative", 0.0))
    matrix = [
        [values["local_resultant"] ** 2, values["local_resultant"] * values["feedback_resultant"] * cosines["local_feedback"], values["local_resultant"] * actual * cosines["local_actual"]],
        [values["local_resultant"] * values["feedback_resultant"] * cosines["local_feedback"], values["feedback_resultant"] ** 2, values["feedback_resultant"] * actual * cosines["feedback_actual"]],
        [values["local_resultant"] * actual * cosines["local_actual"], values["feedback_resultant"] * actual * cosines["feedback_actual"], actual ** 2],
    ]
    direct_dominant = abs(signed_local) >= abs(signed_feedback)
    feedback_dominant = values["feedback_resultant"] > values["local_resultant"] and abs(signed_feedback) > abs(signed_local)
    return {
        "case_id": case_id,
        "status": "COMPLETE_DERIVED_RESULTANT_ATTRIBUTION",
        "source": str(path.relative_to(ROOT)),
        "optimizer": document.get("optimizer", {"name": "UNKNOWN"}),
        "steps": int(document.get("steps", document.get("step_count", 0))),
        "state_ids": document.get("state_ids", document.get("statistics", {}).get("state_ids", [])),
        "levels": {
            name: {
                "resultant_l2": values[f"{name}_resultant"],
                "path_l2": values[f"{name}_path_l2"],
                "path_energy": values[f"{name}_path_energy"],
                "A16": values[f"{name}_A16"],
                "A32": values[f"{name}_A32"],
            }
            for name in ("local", "feedback", "actual")
        },
        "cosine_with_actual": {
            "local": cosines["local_actual"],
            "feedback": cosines["feedback_actual"],
        },
        "signed_projection_share": {
            "local": signed_local,
            "feedback": signed_feedback,
            "recurrence_residual": signed_recurrence,
        },
        "resultant_inner_product_matrix_order": ["local", "feedback", "actual"],
        "resultant_inner_product_matrix": matrix,
        "recurrence_residual": {
            "norm": recurrence_norm,
            "max_relative": recurrence_relative,
            "projection_share_derived_by_identity": signed_recurrence,
        },
        "classification": (
            "FEEDBACK_DOMINATED" if feedback_dominant
            else "DIRECT_OR_MIXED" if direct_dominant
            else "FEEDBACK_ALIGNED_BUT_NOT_DOMINANT"
        ),
        "raw_vector_evidence": {
            "per_step_vectors_available": False,
            "per_step_cross_gram_recomputable": False,
            "resultant_3x3_matrix_recomputable": True,
            "limitation": "Only exported resultant norms/cosines were available; no per-step vector was imputed.",
        },
    }


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["confirmed_label"] = bool(row["adamw_local_persistent"])
    result["nominal_label"] = bool(row["adamw_local_persistent"])
    result["confirmation_role"] = "PREDECLARED" if row["case_id"] in PREDECLARED else "RESULT_BLIND_DISCOVERY"
    if row["case_id"] == UNREPLICATED_CANDIDATE:
        result["confirmed_label"] = False
        result["nominal_label"] = True
        result["confirmation_role"] = "UNRESOLVED_CANDIDATE"
    upper = float(row["full32_sign_flip_upper95"])
    median = None
    source_path = ROOT / str(row.get("source", ""))
    if source_path.exists():
        try:
            median = float(get_null_level(load(source_path), "local")["sign_flip_null"]["median"])
        except (KeyError, TypeError, ValueError):
            median = None
    if median is not None and upper > median:
        result["full32_null_excess"] = (
            float(row["full32_local_A"]) - median
        ) / (upper - median)
        result["full32_null_excess_status"] = "COMPLETE_FROM_SOURCE_MEDIAN"
        result["full32_null_median"] = median
    else:
        # Never infer a median from an upper bound.  The comparison remains
        # unavailable when the source does not export the null median.
        result["full32_null_excess"] = None
        result["full32_null_excess_status"] = "ABSTAIN_MISSING_SIGN_FLIP_MEDIAN"
    result["holm"] = {}
    result["bh_adjusted_p"] = None
    result["screen_decision"] = (
        "ESCALATE" if float(row["prefix16_local_A"]) > 1.0 else "NO_ESCALATION_UNDER_SHORT_SCREEN"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    v3 = load(V3 / "same_optimizer_oracle_v3.json")
    rows = [build_row(row) for row in v3["rows"]]
    predeclared = [row for row in rows if row["confirmation_role"] == "PREDECLARED"]
    # The discovery family is defined by selection history, not by whether a
    # row later became unresolved.  The 0543 candidate therefore remains in
    # the 12-row multiplicity family, while being excluded only from the
    # confirmed-only performance denominator.
    discovery = [row for row in rows if row["case_id"] not in PREDECLARED]
    resolved = [row for row in rows if row["confirmation_role"] != "UNRESOLVED_CANDIDATE"]

    for family_name, family in (
        ("PREDECLARED_3", predeclared),
        ("RESULT_BLIND_DISCOVERY_12", discovery),
        ("ALL_15_SENSITIVITY", rows),
    ):
        adjusted = holm_adjust(family, "full32_sign_flip_one_sided_p")
        bh = bh_adjust(family, "full32_sign_flip_one_sided_p")
        for row in family:
            row["holm"][family_name] = {
                "family_size": len(family),
                **adjusted[row["case_id"]],
            }
            if not isinstance(row.get("bh_adjusted_p"), dict):
                row["bh_adjusted_p"] = {}
            row["bh_adjusted_p"][family_name] = bh[row["case_id"]]

    # Three exported exact trajectories have enough summary information to
    # derive the signed contribution shares without pretending raw vectors exist.
    contribution_sources = {
        "liger_fused_ce_t128": V3 / "liger_adamw_consequence32.json",
        "phi4_seq64_lmhead_dx": V3 / "phi_seq64_adamw_consequence32.json",
        "qwen_seq128_lmhead_dx": V3 / "qwen_seq128_adamw_consequence32_with_stages.json",
    }
    contributions = [
        contribution_row(case_id, path) for case_id, path in contribution_sources.items()
    ]

    nominal_rows = [dict(row, confirmed_label=row["nominal_label"]) for row in rows]
    confirmed_rows = [dict(row) for row in resolved]
    retrospective = {
        "schema": "kernel-analyzer-direct-persistence-v4-retrospective-v1",
        "status": "COMPLETE_OFFLINE_REANALYSIS_WITH_EXPLICIT_UNRESOLVED_FIELDS",
        "optimizer": "cold-start AdamW: moments initialized to zero then evolved normally",
        "nominal_discovery_view": metrics_view(nominal_rows, "NOMINAL_3_POSITIVE_12_NEGATIVE"),
        "confirmed_only_view": metrics_view(confirmed_rows, "CONFIRMED_2_POSITIVE_11_NEGATIVE_PLUS_QWEN"),
        "candidate": {
            "case_id": UNREPLICATED_CANDIDATE,
            "status": "UNRESOLVED_CANDIDATE",
            "reason": "Nominal p=0.026 but does not survive the predeclared 12-row Holm family; it is not relabeled negative.",
        },
        "multiple_testing": {
            "primary": "Holm family-wise correction",
            "families": {
                "PREDECLARED_3": [row["case_id"] for row in predeclared],
                "RESULT_BLIND_DISCOVERY_12": [row["case_id"] for row in discovery],
                "ALL_15_SENSITIVITY": [row["case_id"] for row in rows],
            },
            "secondary": "Benjamini-Hochberg exploratory FDR only",
        },
        "short_screen": {
            "name": "Cold-start AdamW Direct Persistence Screen",
            "primary_rule": "prefix16_local_A > 1.0",
            "outputs": ["ESCALATE", "NO_ESCALATION_UNDER_SHORT_SCREEN", "ABSTAIN"],
            "no_safe_verdict": True,
            "prefix_null_calibration": "ABSTAIN_MISSING_PREFIX_SIGN_FLIP_GRAM",
        },
        "confidence_boundary": {
            "auc_bootstrap_draws": 5000,
            "binomial_interval": "Clopper-Pearson when scipy is available; otherwise null",
            "raw_vectors": "not available for existing v3 exports",
        },
    }

    (args.output / "retrospective_metrics.json").write_text(
        json.dumps(retrospective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    multiplicity = {
        "schema": "kernel-analyzer-direct-persistence-v4-multiplicity-v1",
        "status": "COMPLETE_OFFLINE_REANALYSIS",
        "primary_method": "Holm family-wise correction",
        "secondary_method": "Benjamini-Hochberg exploratory FDR only",
        "families": {
            family_name: {
                "case_ids": [row["case_id"] for row in family],
                "family_size": len(family),
                "rows": {
                    row["case_id"]: {
                        "raw_p": row["full32_sign_flip_one_sided_p"],
                        "holm": row["holm"][family_name],
                        "bh_adjusted_p": row["bh_adjusted_p"][family_name],
                        "confirmed_label": row["confirmed_label"],
                        "nominal_label": row["nominal_label"],
                    }
                    for row in family
                },
            }
            for family_name, family in (
                ("PREDECLARED_3", predeclared),
                ("RESULT_BLIND_DISCOVERY_12", discovery),
                ("ALL_15_SENSITIVITY", rows),
            )
        },
        "candidate_policy": {
            "case_id": UNREPLICATED_CANDIDATE,
            "status": "UNRESOLVED_CANDIDATE",
            "excluded_from_confirmed_only_metrics": True,
            "never_relabel_as_negative": True,
        },
    }
    (args.output / "multiplicity.json").write_text(
        json.dumps(multiplicity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "contribution_table.json").write_text(
        json.dumps({"schema": "kernel-analyzer-direct-persistence-contribution-v1", "rows": contributions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "contribution_table.csv").open("w", encoding="utf-8") as handle:
        handle.write("case_id,status,local_resultant,feedback_resultant,actual_resultant,A_local,A_feedback,A_actual,local_share,feedback_share,recurrence_share,classification\n")
        for row in contributions:
            if row["status"].startswith("ABSTAIN"):
                handle.write(f"{row['case_id']},{row['status']},,,,,,,,,,\n")
                continue
            levels = row["levels"]
            shares = row["signed_projection_share"]
            handle.write(
                f"{row['case_id']},{row['status']},{levels['local']['resultant_l2']},{levels['feedback']['resultant_l2']},{levels['actual']['resultant_l2']},"
                f"{levels['local']['A32']},{levels['feedback']['A32']},{levels['actual']['A32']},{shares['local']},{shares['feedback']},{shares['recurrence_residual']},{row['classification']}\n"
            )
    (args.output / "cohort.json").write_text(
        json.dumps({
            "schema": "kernel-analyzer-direct-persistence-v4-cohort-v1",
            "status": "RETROSPECTIVE_COHORT_COMPLETE_HELDOUT_NOT_STARTED",
            "source": "results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json",
            "rows": rows,
            "predeclared_case_ids": sorted(PREDECLARED),
            "unresolved_candidates": [UNREPLICATED_CANDIDATE],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    optimizer_manifest_path = args.output / "optimizer_state" / "manifest.json"
    optimizer_run_manifest_path = args.output / "optimizer_state_run_manifest.json"
    optimizer_progress = load(optimizer_manifest_path) if optimizer_manifest_path.exists() else {
        "status": "NOT_STARTED",
        "completed_cases": [],
    }
    optimizer_run_progress = load(optimizer_run_manifest_path) if optimizer_run_manifest_path.exists() else {}
    external_heldout_path = args.output / "heldout_gemma_confirmation.json"
    external_heldout_validation_path = args.output / "heldout_gemma_validation.json"
    external_heldout = load(external_heldout_path) if external_heldout_path.exists() else None
    external_heldout_validation = load(external_heldout_validation_path) if external_heldout_validation_path.exists() else None
    summary = {
        "schema": "kernel-analyzer-direct-persistence-v4-summary-v1",
        "status": "DEVELOPMENT_REANALYSIS_COMPLETE_HELDOUT_AND_OPTIMIZER_PHASES_PENDING",
        "confirmed_headline_cases": ["liger_fused_ce_t128", "phi4_seq64_lmhead_dx"],
        "unresolved_candidate": UNREPLICATED_CANDIDATE,
        "qwen": "not direct-persistent under cold-start AdamW",
        "phi_sr_scope": "16-state stateless-SGD source intervention; not evidence for Phi AdamW A=1.029",
        "raw_vector_limitation": "Per-step vectors and full sequence cross-Grams were not saved in v3; no vectors were imputed.",
        "gpu_preflight": {
            "status": "ABSTAIN_RUNTIME_MISMATCH",
            "artifact": "results/property/direct_persistence_v4/gpu_preflight.json",
            "scientific_result": False,
        },
        "optimizer_state_progress": {
            "status": optimizer_progress.get("status", "NOT_STARTED"),
            "completed_cases": optimizer_progress.get("same_state_ablation", {}).get("completed_cases", []),
            "natural_phase_status": optimizer_run_progress.get("phase_conditioned_natural", "NOT_STARTED"),
            "claim_boundary": optimizer_progress.get("claim_boundary"),
        },
        "heldout_external_progress": {
            "status": external_heldout.get("status", "NOT_STARTED") if external_heldout else "NOT_STARTED",
            "validation": external_heldout_validation.get("status", "NOT_STARTED") if external_heldout_validation else "NOT_STARTED",
            "eligible_rows": external_heldout.get("metrics", {}).get("eligible_rows") if external_heldout else 0,
            "confirmed_positive": external_heldout.get("metrics", {}).get("confirmed_positive") if external_heldout else 0,
            "confirmed_negative": external_heldout.get("metrics", {}).get("confirmed_negative") if external_heldout else 0,
            "claim_boundary": "One independently frozen NEW_IMPL negative is complete; recall/AUROC remain undefined until more rows exist.",
        },
        "next_required": [
            "same-state optimizer ablation for 0543 when exact raw captures exist",
            "phase-conditioned natural optimizer evaluation with real early/middle/late moments",
            "expand the independently frozen NEW_IMPL pool beyond the one completed Gemma negative",
            "complete tolerance metrics on a common held-out pool",
            "prospective executable catch-and-fix if a candidate escalates",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": retrospective["status"], "output": str(args.output), "rows": len(rows), "contributions": len(contributions)}, sort_keys=True))


if __name__ == "__main__":
    main()
