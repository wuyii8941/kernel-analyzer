#!/usr/bin/env python3
"""Build one conservative long-horizon audit of every historical bias candidate.

This registry deliberately separates three questions:
1. Was a direct direction measured for 4096 steps?
2. Did it pass the long-horizon sign-flip/window check?
3. Was a paired parameter/loss separation observed?

For a direct-source case, the long direct gate is sufficient to establish a
persistent bias; a paired loss gap is reported as its consequence when
available. A feedback-sustained case may instead qualify when the direct
effect is not persistent but a completed long-horizon feedback separation has
an observed paired loss gap; it is reported separately from direct-source
persistence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "results/property/declared_persistent_4096"
PAIRED = ROOT / "results/property/paired_loss_4096"
OUT = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
MD = ROOT / "docs/all_bias_long_horizon_audit.md"


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def long_row(path: Path) -> dict[str, Any]:
    payload = read(path)
    if payload is None:
        return {"status": "NOT_RUN", "artifact": str(path.relative_to(ROOT))}
    # The generic bound-endpoint runner stores the three paths under
    # statistics.levels.  It does not keep 4096 rolling windows, so use its
    # frozen sign-flip null and mark the window check as unavailable rather
    # than silently treating a 32-step result as long.
    if payload.get("runner", "").endswith("run_bound_endpoint_consequence_v21.py"):
        levels = payload.get("statistics", {}).get("levels", {})
        actual = levels.get("actual", {})
        if actual:
            null = actual.get("sign_flip_null", {})
            a = float(actual.get("coherence_amplification", 0.0))
            upper = float(null.get("upper_95", 0.0))
            p = float(null.get("one_sided_p", 1.0))
            return {
                "status": payload.get("trajectory_status", payload.get("status", "COMPLETE")),
                "long_direct": "ROBUST" if p <= 0.05 and a > upper else "NOT_ROBUST",
                "steps": int(payload.get("steps", payload.get("step_count", 0))),
                "A4096": a, "null95": upper, "p": p,
                "late_windows": None, "late_windows_above_own_null": None,
                "loss_audit": payload.get("loss_audit", {}),
                "local_A4096": levels.get("local", {}).get("coherence_amplification"),
                "feedback_A4096": levels.get("feedback", {}).get("coherence_amplification"),
                "final_drift_l2": payload.get("cumulative", {}).get("actual", {}).get("resultant_l2"),
                "artifact": str(path.relative_to(ROOT)),
            }
    if "statistics" in payload and "levels" in payload.get("statistics", {}):
        # Held-out lm-head consequence runner: it has the same three-vector
        # statistics as the bound runner, but no rolling-window export.
        actual = payload["statistics"]["levels"].get("actual", {})
        if actual:
            null = actual.get("sign_flip_null", {})
            a = float(actual.get("coherence_amplification", 0.0))
            upper = float(null.get("upper_95", 0.0))
            p = float(null.get("one_sided_p", 1.0))
            return {
                "status": payload.get("status", "COMPLETE"),
                "long_direct": "ROBUST" if p <= 0.05 and a > upper else "NOT_ROBUST",
                "steps": int(payload.get("steps", 0)),
                "A4096": a, "null95": upper, "p": p,
                "late_windows": None, "late_windows_above_own_null": None,
                "local_A4096": payload["statistics"]["levels"].get("local", {}).get("coherence_amplification"),
                "feedback_A4096": payload["statistics"]["levels"].get("feedback", {}).get("coherence_amplification"),
                "final_drift_l2": payload.get("final_master_drift_l2"),
                "artifact": str(path.relative_to(ROOT)),
            }
    # The common runner writes a top-level status only in the completed artifact.
    if "statistics" in payload:
        stats = payload["statistics"]
        exact = stats["exact_full_coordinate"]
        sketch = stats["sketch_diagnostics"]
        windows = exact.get("rolling_windows", [])
        late = windows[len(windows) // 2 :]
        p = float(sketch["sign_flip_null"]["one_sided_p"])
        null95 = float(sketch["sign_flip_null"]["upper_95"])
        a = float(exact["coherence_amplification"])
        above = sum(float(w["coherence_amplification"]) > 1.0 for w in late)
        robust = p <= 0.05 and late and above / len(late) >= 0.75
        return {
            "status": "COMPLETE_4096",
            "long_direct": "ROBUST" if robust else "NOT_ROBUST",
            "steps": int(payload["protocol"]["measurement_steps"]),
            "A4096": a,
            "null95": null95,
            "p": p,
            "late_windows_above_one": above,
            "late_windows": len(late),
            "artifact": str(path.relative_to(ROOT)),
        }
    # Phi uses a slightly different schema.
    if "full" in payload and "windows" in payload:
        full = payload["full"]
        late = payload["windows"][len(payload["windows"]) // 2 :]
        p = float(full["sign_flip_null"]["one_sided_p"])
        null95 = float(full["sign_flip_null"]["upper_95"])
        above = sum(bool(w.get("above_sign_flip_95")) for w in late)
        robust = p <= 0.05 and late and above / len(late) >= 0.75
        return {
            "status": "COMPLETE_4096",
            "long_direct": "ROBUST" if robust else "NOT_ROBUST",
            "steps": int(payload["protocol"]["measurement_steps"]),
            "A4096": float(full["coherence_amplification"]),
            "null95": null95,
            "p": p,
            "late_windows_above_own_null": above,
            "late_windows": len(late),
            "artifact": str(path.relative_to(ROOT)),
        }
    if "summaries" in payload and payload.get("status") == "COMPLETE":
        verdict = str(payload.get("verdict", ""))
        summary = payload["summaries"]
        if verdict == "PERSISTENT_LOCAL_BIAS":
            long_status = "ROBUST"
        elif verdict == "FEEDBACK_SUSTAINED_SEPARATION":
            long_status = "FEEDBACK_SUSTAINED"
        else:
            long_status = "NOT_ROBUST"
        return {
            "status": "COMPLETE_4096",
            "long_direct": long_status,
            "verdict": verdict,
            "steps": int(payload.get("protocol", {}).get("steps", 0)),
            "A4096": float(summary["actual_drift_increment"].get("coherence_amplification", 0.0)),
            "local_A4096": float(summary["local"].get("coherence_amplification", 0.0)),
            "feedback_A4096": float(summary["feedback"].get("coherence_amplification", 0.0)),
            "final_drift_l2": summary.get("final_drift_l2"),
            "paired_loss_gap_final": summary.get("paired_loss_gap_final"),
            "paired_loss_gap_mean_last_512": summary.get("paired_loss_gap_mean_last_512"),
            "paired_loss_gap_std_last_512": summary.get("paired_loss_gap_std_last_512"),
            "artifact": str(path.relative_to(ROOT)),
        }
    if payload.get("schema", "").startswith("kernel-analyzer-gemma4-target-consequence") and str(payload.get("status", "")).startswith("COMPLETE"):
        stats = payload.get("statistics", {})
        local = stats.get("local", {})
        feedback = stats.get("feedback", {})
        actual = stats.get("actual", {})
        long_status = "FEEDBACK_SUSTAINED" if (
            float(feedback.get("coherence_amplification", 0.0)) > 1.25
            and float(local.get("coherence_amplification", 0.0)) < 1.25
        ) else "NOT_ROBUST"
        return {
            "status": "COMPLETE_4096",
            "long_direct": long_status,
            "verdict": "FEEDBACK_SUSTAINED_SEPARATION" if long_status == "FEEDBACK_SUSTAINED" else "NO_ROBUST_LONG_DIRECT",
            "steps": int(payload.get("steps", 0)),
            "A4096": float(actual.get("coherence_amplification", 0.0)),
            "local_A4096": float(local.get("coherence_amplification", 0.0)),
            "feedback_A4096": float(feedback.get("coherence_amplification", 0.0)),
            "paired_loss_gap_final": payload.get("paired_loss_gap_final"),
            "paired_loss_gap_mean_last_512": payload.get("paired_loss_gap_mean_last_512"),
            "paired_loss_gap_std_last_512": payload.get("paired_loss_gap_std_last_512"),
            "loss_audit": {
                "recorded": bool(payload.get("records")),
                "any_period_split": any(abs(float(r.get("paired_loss_gap_candidate_minus_repair", 0.0))) > 1e-8 for r in payload.get("records", [])),
                "max_abs_gap": max((abs(float(r.get("paired_loss_gap_candidate_minus_repair", 0.0))) for r in payload.get("records", [])), default=0.0),
            },
            "artifact": str(path.relative_to(ROOT)),
        }
    if payload.get("status", "").startswith("ABSTAIN") or "decision" in payload:
        return {
            "status": "ABSTAIN",
            "long_direct": "ABSTAIN",
            "reason": payload.get("decision", payload.get("reason", "")),
            "artifact": str(path.relative_to(ROOT)),
        }
    return {"status": "INVALID_OR_INCOMPLETE", "artifact": str(path.relative_to(ROOT))}


def paired_row(path: Path) -> dict[str, Any]:
    payload = read(path)
    if payload is None:
        return {"status": "NOT_RUN", "artifact": str(path.relative_to(ROOT))}
    final = payload.get("final", {})
    recent = payload.get("last_512_train_steps", {}).get("paired_loss_gap", {})
    # The earlier Phi convergence runner used a different, but still auditable,
    # 4096-step schema.  Keep it in the same consequence column without
    # pretending its frozen-plateau gate passed.
    if not recent and "train_rows" in payload:
        tail = payload["train_rows"][-min(512, len(payload["train_rows"])) :]
        gaps = [float(row.get("loss_gap_candidate_minus_repair", 0.0)) for row in tail]
        if gaps:
            import statistics
            recent = {
                "mean": statistics.fmean(gaps),
                "population_std": statistics.pstdev(gaps),
            }
    if not recent and "records" in payload:
        tail = payload["records"][-min(512, len(payload["records"])) :]
        gaps = [
            float(row.get("paired_loss_gap", row.get("paired_loss_gap_candidate_minus_repair", 0.0)))
            for row in tail
        ]
        if gaps:
            import statistics
            recent = {
                "mean": statistics.fmean(gaps),
                "population_std": statistics.pstdev(gaps),
            }
    return {
        "status": payload.get("status", "UNKNOWN"),
        "loss_separation_observed": bool(
            payload.get("loss_separation_observed", False)
            or (
                final.get("parameter_distance_l2", 0.0) > 0.0
                and abs(final.get("loss_gap_candidate_minus_repair", 0.0)) > 0.0
            )
            or any(
                abs(float(row.get("paired_loss_gap", row.get("paired_loss_gap_candidate_minus_repair", 0.0)))) > 1e-8
                for row in payload.get("records", [])
            )
        ),
        "steps": payload.get("protocol", {}).get("steps") or payload.get("steps_completed") or payload.get("steps"),
        "parameter_distance": final.get("parameter_distance_l2") or payload.get("final_master_drift_l2"),
        "final_loss_gap": final.get("loss_gap_candidate_minus_repair") if final else (
            payload.get("records", [{}])[-1].get("paired_loss_gap", payload.get("records", [{}])[-1].get("paired_loss_gap_candidate_minus_repair"))
            if payload.get("records") else None
        ),
        "recent_loss_gap_mean": recent.get("mean"),
        "recent_loss_gap_std": recent.get("population_std"),
        "loss_audit": payload.get("loss_audit", {
            "recorded": bool(recent or final),
            "any_period_split": bool(abs(float(final.get("loss_gap_candidate_minus_repair", 0.0) or 0.0)) > 1e-8 or abs(float(recent.get("mean", 0.0) or 0.0)) > 1e-8),
        }),
        "artifact": str(path.relative_to(ROOT)),
    }


CASES = [
    ("Liger fused CE", "Qwen3-1.7B", "fused CE dW accumulation", "event/pairing imbalance", LONG / "liger_fused_ce.json", PAIRED / "liger_fused_ce.json"),
    ("Phi lm_head dX", "Phi-4-mini", "lm_head backward dX", "event/pairing imbalance + backward transport", LONG / "phi_lmhead_dx.json", ROOT / "results/property/convergence_v1/phi_carrier_adamw_convergence.json"),
    ("Qwen lm_head dX", "Qwen3-1.7B", "lm_head backward dX", "event/pairing imbalance + backward transport", LONG / "qwen_lmhead_dx.json", PAIRED / "qwen_lmhead_dx.json"),
    ("Qwen64 v_proj", "Qwen3-1.7B", "seq64 v_proj MM + output rounding", "conditional source asymmetry", LONG / "qwen64_vproj_output.json", None),
    ("Qwen v_proj", "Qwen3-1.7B", "v_proj MM/output rounding", "local arithmetic/pairing", LONG / "qwen_vproj_output.json", None),
    ("Mamba in_proj", "Mamba-130M", "in_proj matrix multiply", "local arithmetic/pairing", LONG / "mamba_in_proj.json", None),
    ("saved-P", "Qwen3-1.7B", "layer-27 saved-P softmax backward", "response/state-contract imbalance", LONG / "qwen_saved_p.json", None),
    ("Qwen3-VL SiLU", "Qwen3-VL-Reranker-2B", "SiLU backward", "response asymmetry", LONG / "qwen3vl_silu_4096.json", None),
    ("layer-23 attention", "Qwen3-1.7B", "attention S_bwd/K to q_proj", "event/pairing imbalance", LONG / "layer23_qproj_attention_state_region.json", None),
    ("DeepSeek layer-35 dV", "DeepSeek-R1-Qwen3-8B", "attention dV BMM", "formation unresolved; event/pairing candidate", LONG / "deepseek_l35_dv.json", None),
    ("Llama lm_head dX", "Llama-3.2-3B", "lm_head backward dX", "event/pairing family replication", ROOT / "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/lmhead_consequence4096.json", ROOT / "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/lmhead_consequence4096.json"),
    ("Ministral lm_head dX", "Ministral-3-3B", "lm_head backward dX", "event/pairing family replication", ROOT / "results/property/tcmp_allop_v1/heldout/ministral3_3b_text128/lmhead_consequence4096.json", ROOT / "results/property/tcmp_allop_v1/heldout/ministral3_3b_text128/lmhead_consequence4096.json"),
    ("Gemma4 RMSNorm", "Gemma-4 E2B", "RMSNorm / projection feedback region", "response asymmetry / feedback candidate", ROOT / "results/property/declared_persistent_4096/gemma4_norm_4096_v4.json", ROOT / "results/property/declared_persistent_4096/gemma4_norm_4096_v4.json"),
]


def final_label(direct: dict[str, Any], paired: dict[str, Any], formation_path: str = "") -> str:
    loss_audit = direct.get("loss_audit", {}) or paired.get("loss_audit", {}) or {}
    any_loss = bool(loss_audit.get("any_period_split") or paired.get("loss_separation_observed"))
    if direct.get("long_direct") == "FEEDBACK_SUSTAINED":
        if any_loss:
            return "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT"
        if float(direct.get("final_drift_l2") or paired.get("parameter_distance") or 0.0) > 0.0:
            return "FEEDBACK_SUSTAINED_PARAMETER_SPLIT_LOSS_NOT_RECORDED"
        return "FEEDBACK_SUSTAINED_LOSS_NOT_RECORDED"
    if direct.get("long_direct") == "ROBUST" and any_loss:
        return "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT"
    if direct.get("long_direct") == "ROBUST":
        return "ROBUST_LONG_DIRECT_BIAS_LOSS_NOT_RUN"
    if direct.get("status") == "ABSTAIN":
        return "ABSTAIN_NOT_REPLAYABLE"
    if direct.get("long_direct") == "NOT_ROBUST":
        return "NO_ROBUST_LONG_DIRECT_BIAS"
    if direct.get("status") == "NOT_RUN":
        if "formation unresolved" in formation_path:
            return "UNRESOLVED_FORMATION"
        if "no direct source" in formation_path:
            return "NOT_ESCALATED_NO_DIRECT_SOURCE"
        return "NOT_RUN"
    if str(direct.get("status", "")).startswith("UNRESOLVED"):
        return "UNRESOLVED_LONG_REPLAY"
    return "UNRESOLVED"


def main() -> None:
    rows: list[dict[str, Any]] = []
    known: set[str] = {
        "qwen_seq128_lmhead_dx", "liger_fused_ce_t128", "phi4_seq64_lmhead_dx",
        "qwen64_vproj", "qwen128_vproj_output", "mamba_seq64_in_proj", "qwen_saved_p_seq128",
        "qwen3vl_silu_backward", "layer23_qproj_attention_state_region",
        "gemma4_norm", "deepseek8b_seq64_l35_attention_dv",
    }
    for name, model, region, path, long_path, paired_path in CASES:
        unresolved_paths = {
            "Llama lm_head dX": LONG / "unresolved/llama32_lmhead_4096_unresolved.json",
            "Ministral lm_head dX": LONG / "unresolved/ministral3_lmhead_4096_unresolved.json",
            "Gemma4 RMSNorm": LONG / "unresolved/gemma4_norm_4096_unresolved.json",
        }
        # The first SiLU run predated loss-gap recording.  Prefer the exact
        # same protocol rerun once its richer artifact exists.
        if name == "Qwen3-VL SiLU" and (LONG / "qwen3vl_silu_4096_with_loss.json").exists():
            long_path = LONG / "qwen3vl_silu_4096_with_loss.json"
        if not long_path.exists() and name in unresolved_paths and unresolved_paths[name].exists():
            unresolved = read(unresolved_paths[name]) or {}
            direct = {
                "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY"),
                "long_direct": "UNRESOLVED",
                "reason": unresolved.get("reason", "long replay unavailable"),
                "artifact": str(unresolved_paths[name].relative_to(ROOT)),
            }
        else:
            direct = long_row(long_path)
        if name == "Gemma4 RMSNorm" and direct.get("status") == "NOT_RUN":
            direct = {
                "status": "UNRESOLVED_REPLAY_RUNTIME",
                "long_direct": "UNRESOLVED",
                "reason": "frozen runtime wrapper bytes no longer match the available compiler environment",
                "artifact": str(long_path.relative_to(ROOT)),
            }
        paired = paired_row(paired_path) if paired_path else {"status": "NOT_RUN"}
        # SiLU's long recurrence runner records the paired parameter and loss
        # gap in the same artifact; do not report that evidence as "unmeasured"
        # merely because it has no separate paired-loss file.
        if name == "Qwen3-VL SiLU" and direct.get("paired_loss_gap_mean_last_512") is not None:
            paired = {
                "status": "COMPLETE_IN_LONG_RECURRENCE",
                "loss_separation_observed": abs(float(direct.get("paired_loss_gap_mean_last_512") or 0.0)) > 0.0,
                "steps": direct.get("steps"),
                "parameter_distance": direct.get("final_drift_l2"),
                "final_loss_gap": direct.get("paired_loss_gap_final"),
                "recent_loss_gap_mean": direct.get("paired_loss_gap_mean_last_512"),
                "recent_loss_gap_std": direct.get("paired_loss_gap_std_last_512"),
                "artifact": direct.get("artifact"),
            }
        rows.append({
            "case": name,
            "model": model,
            "operator_or_region": region,
            "formation_path": path,
            "long_direct": direct,
            "paired_consequence": paired,
            "scope": "historical_candidate",
            "final_label": final_label(direct, paired, path),
            "direct_sha256": sha(long_path),
            "paired_sha256": sha(paired_path) if paired_path else None,
        })

    # Keep the complete 23-case roster visible.  Rows that never passed the
    # nonzero, parameter-reachable direct-bias gate are recorded as screened
    # negatives or unresolved; they are not silently dropped and are not
    # promoted to a 4096-step experiment without a valid source contrast.
    matrix_path = ROOT / "results/evidence_v2/case_stage_matrix.json"
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
        for case_id, info in matrix.get("case_index", {}).items():
            if case_id in known or case_id in {row["case"] for row in rows}:
                continue
            records = [r for r in matrix.get("records", []) if r.get("case_id") == case_id]
            info_row = records[0] if records else {}
            statuses = [str(r.get("formation_status", r.get("status", ""))) for r in records]
            if any("UNRESOLVED" in s for s in statuses):
                label = "UNRESOLVED_FORMATION"
            elif any("FEEDBACK" in s for s in statuses):
                label = "FEEDBACK_CONTROL_NO_DIRECT_GATE"
            elif any("NOT_APPLICABLE" in s for s in statuses):
                label = "NOT_APPLICABLE_NO_CARRIER_EFFECT"
            else:
                label = "SCREENED_NO_CONFIRMED_DIRECT_BIAS"
            rows.append({
                "case": case_id,
                "model": info_row.get("model", "from case-stage matrix"),
                "operator_or_region": info_row.get("operator_or_region", info_row.get("endpoint", "case-stage matrix row")),
                "formation_path": "complete roster row; no confirmed long-source gate",
                "long_direct": {"status": "NOT_ESCALATED", "long_direct": "NOT_RUN", "reason": statuses},
                "paired_consequence": {"status": "NOT_RUN"},
                "final_label": label,
                "scope": "roster_control_or_unresolved",
                "direct_sha256": None,
                "paired_sha256": None,
            })
    matrix_count = 0
    matrix_ids: set[str] = set()
    coherent_endpoint_count = 0
    endpoint_population_count = 0
    matrix_path = ROOT / "results/evidence_v2/case_stage_matrix.json"
    if matrix_path.exists():
        matrix_payload = json.loads(matrix_path.read_text())
        matrix_count = int(matrix_payload.get("case_count", 0))
        matrix_ids = set(matrix_payload.get("case_index", {}))
    population_path = ROOT / "results/property/bias_formation/bias_population.csv"
    if population_path.exists():
        import csv
        with population_path.open(newline="", encoding="utf-8") as handle:
            population_rows = list(csv.DictReader(handle))
        endpoint_population_count = sum(r.get("population_kind") == "ENDPOINT_UNIT" for r in population_rows)
        coherent_endpoint_count = sum(r.get("legacy_observed_role") == "COHERENT_F_B_BIAS" for r in population_rows)
    historical_ids = {
        "qwen_seq128_lmhead_dx", "liger_fused_ce_t128", "phi4_seq64_lmhead_dx",
        "qwen64_vproj", "qwen128_vproj_output", "mamba_seq64_in_proj",
        "qwen_saved_p_seq128", "qwen3vl_silu_backward", "gemma4_norm",
        "layer23_qproj_attention_state_region", "deepseek8b_seq64_l35_attention_dv",
    }
    payload = {
        "schema": "all-historical-bias-candidates-long-audit-v1",
        "definition": "A final bias case requires either robust 4096-step direct direction (persistent source bias) or a long-run feedback-sustained separation with an observed paired parameter/loss split. A loss split is recorded as a consequence of a direct case when available; different converged losses are not required or claimed.",
        "long_direct_rule": "one-sided sign-flip p<=0.05 and at least 75% of late rolling windows directional",
        "case_count": len(rows),
        "unique_matrix_case_count": matrix_count,
        "endpoint_population_count": endpoint_population_count,
        "coherent_endpoint_witness_count": coherent_endpoint_count,
        "grouping_rule": "Repeated concrete endpoint occurrences are grouped by the frozen case-stage matrix ID; they are not silently dropped, but they do not count as independent cases.",
        "historical_candidate_count": 11,
        "extended_candidate_count": len(CASES),
        "extended_candidate_note": "13 rows = 11 historical candidates plus Llama and Ministral lm_head family replication rows; Gemma4 is included in the extended candidate list, not a new source-positive.",
        "unindexed_historical_candidate_count": len(historical_ids - matrix_ids),
        "scope_counts": {
            "historical_candidate": sum(r.get("scope") == "historical_candidate" for r in rows),
            "roster_control_or_unresolved": sum(r.get("scope") == "roster_control_or_unresolved" for r in rows),
        },
        "final_case_count": sum(
            r["final_label"] in {
                "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT",
                "ROBUST_LONG_DIRECT_BIAS_LOSS_NOT_RUN",
                "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT",
            }
            for r in rows
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 所有历史偏差候选的长程复核",
        "",
        f"仓库中有 **{matrix_count} 个唯一主矩阵 case ID**；本审计逐行复核 **{len(CASES)} 个 extended candidate rows**（其中 11 个来自历史候选，另含 Llama、Ministral 的同族复现行）。合并后表共有 **{len(rows)} 行**。这些数字分别表示覆盖分母、候选分母和逐行审计行数，不能混用。",
        "",
        "直接源案例只要 4096 步直接更新差异仍有稳定方向，就已经是持久性 bias；如果配对训练还观察到参数或 loss 分叉，就作为后果一并报告。反馈维持型案例只有在 4096 步反馈分离并观察到配对参数/loss gap 时才单独计入，不冒充直接源 bias。这里不要求两条训练轨迹收敛到不同的最终 loss，也不作这种声称。",
        "",
        "| 模型 | 算子或位置 | 形成路径 | 4096 步直接结果 | 参数/loss 分叉 | 最终分类 |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT": "最终持久性 bias 案例",
        "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT": "反馈维持型 bias，且有 loss 分叉",
        "FEEDBACK_SUSTAINED_LOSS_NOT_RECORDED": "反馈维持，但 loss 尚未记录",
        "ROBUST_LONG_DIRECT_BIAS_LOSS_NOT_RUN": "长程方向成立，loss 尚未测",
        "NO_ROBUST_LONG_DIRECT_BIAS": "长程未保持",
        "ABSTAIN_NOT_REPLAYABLE": "不可安全重放",
        "NOT_RUN": "尚未运行",
        "UNRESOLVED_LONG_REPLAY": "长程运行环境不再可重放，未决",
        "NOT_ESCALATED_NO_DIRECT_SOURCE": "没有通过直接偏差门，未升级长程",
        "UNRESOLVED": "未决",
        "UNRESOLVED_FORMATION": "形成阶段未确认，不升级长程",
        "FEEDBACK_CONTROL_NO_DIRECT_GATE": "反馈对照，没有直接 bias 门",
        "NOT_APPLICABLE_NO_CARRIER_EFFECT": "没有可达载体，不适用",
        "SCREENED_NO_CONFIRMED_DIRECT_BIAS": "短程筛查未确认直接 bias",
    }
    for row in rows:
        d, p = row["long_direct"], row["paired_consequence"]
        if d.get("long_direct") == "ROBUST":
            if d.get("late_windows") is None:
                direct = f"A4096={d['A4096']:.3f}，超过自身随机基线（窗口统计未导出）"
            else:
                direct = f"A4096={d['A4096']:.3f}，后半程 {d.get('late_windows_above_one', d.get('late_windows_above_own_null', 0))}/{d['late_windows']}"
        elif d.get("long_direct") == "NOT_ROBUST":
            direct = f"A4096={d.get('A4096', 0):.3f}，p={d.get('p', 1):.3f}"
        else:
            direct = d.get("status", "—")
        if p.get("loss_separation_observed"):
            consequence = f"是；参数距离 {p.get('parameter_distance', 0):.3g}，末步 loss gap {p.get('final_loss_gap', 0):+.3g}"
        elif p.get("status") == "NOT_RUN":
            consequence = "未测"
        else:
            consequence = "未观察到"
        lines.append(f"| {row['model']} | {row['operator_or_region']} | {row['formation_path']} | {direct} | {consequence} | {labels[row['final_label']]} |")
    lines += [
        "",
        "## 口径",
        "",
        "- 32 步只能说明短程方向性，不能单独称为持久性 bias。",
        "- 4096 步是同一训练状态下的直接更新审计，不等于完整全参数训练收敛。",
        "- 配对 loss gap 是功能后果信号；这里不要求、也不声称两条轨迹收敛到不同的最终 loss。",
        "- 反馈造成的轨迹分离不能替代直接 bias 证据。",
        "",
    ]
    MD.write_text("\n".join(lines))
    print(json.dumps({"output": str(OUT), "final_case_count": payload["final_case_count"], "case_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
