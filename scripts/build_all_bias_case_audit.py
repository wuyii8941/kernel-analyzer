#!/usr/bin/env python3
"""Build one conservative long-horizon audit of every historical bias candidate.

This registry deliberately separates three questions:
1. Was a direct direction measured for 4096 steps?
2. Did it pass the long-horizon sign-flip/window check?
3. Was a paired parameter/loss separation observed?

For a direct-source case, the long direct gate is sufficient to establish a
persistent bias; a paired loss gap is reported as its consequence when
available. A feedback-sustained case may qualify only when the feedback
effective-update direction itself remains long-range and a paired parameter or
loss split is observed. A loss split by itself never establishes a persistent
bias case; it is retained as a consequence-only control.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "results/property/declared_persistent_4096"
PAIRED = ROOT / "results/property/paired_loss_4096"
OUT = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
MD = ROOT / "docs/all_bias_long_horizon_audit.md"
MANIFEST = LONG / "expanded_controls/manifest.json"
RETRY_SCHEDULED_IDS = {
    "multishape-backward-cell-0103",
    "multishape-backward-cell-0153",
    "multishape-backward-cell-0190",
    "multishape-backward-cell-0501",
    "multishape-backward-cell-0508",
}

GEMMA_GELU_LONG = LONG / "gemma4_random_gelu_loss_backward_long" / "consequence.json"
GEMMA_GELU_LOG = LONG / "expanded_controls/logs/gemma4_random_gelu_loss_backward_long.log"
GEMMA_GELU_UNRESOLVED = LONG / "unresolved/gemma4_random_gelu_loss_backward_long_unresolved.json"
GEMMA_NORM_LOG = LONG / "expanded_controls/logs/gemma4_norm_v3_long.log"
TARGET_MANIFEST = LONG / "operator_scan_target_manifest.json"


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def window_sidecar(path: Path) -> dict[str, Any] | None:
    """Return offline late-window evidence when it has been computed."""
    candidate = path.with_name(path.stem + "_windows.json")
    return read(candidate)


def long_row(path: Path) -> dict[str, Any]:
    payload = read(path)
    if payload is None:
        return {"status": "NOT_RUN", "artifact": str(path.relative_to(ROOT))}
    if str(payload.get("status", "")).startswith("UNRESOLVED"):
        return {
            "status": payload.get("status"),
            "long_direct": "UNRESOLVED",
            "reason": payload.get("reason", "long replay unavailable"),
            "artifact": str(path.relative_to(ROOT)),
        }
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
            measured_steps = int(payload.get("steps", payload.get("step_count", 0)))
            windows = window_sidecar(path)
            local_windows = (windows or {}).get("levels", {}).get("local", {})
            feedback_windows = (windows or {}).get("levels", {}).get("feedback", {})
            if windows is not None:
                if local_windows.get("long_persistent"):
                    long_direct = "ROBUST"
                elif feedback_windows.get("long_persistent"):
                    long_direct = "FEEDBACK_SUSTAINED"
                else:
                    long_direct = "NOT_ROBUST"
            else:
                # Preserve the historical aggregate interpretation until the
                # checkpoint-window sidecar is available. New replays are
                # upgraded by the offline analyzer before final publication.
                long_direct = "ROBUST" if p <= 0.05 and a > upper else "NOT_ROBUST"
            return {
                "status": "COMPLETE_4096" if measured_steps >= 4096 and payload.get("status") == "COMPLETE" else payload.get("trajectory_status", payload.get("status", "COMPLETE")),
                "long_direct": long_direct,
                "steps": measured_steps,
                "A4096": a, "null95": upper, "p": p,
                "late_windows": local_windows.get("late_windows") if windows is not None else None,
                "late_windows_above_own_null": local_windows.get("late_windows_above_own_null") if windows is not None else None,
                "loss_audit": payload.get("loss_audit", {}),
                "local_A4096": levels.get("local", {}).get("coherence_amplification"),
                "feedback_A4096": levels.get("feedback", {}).get("coherence_amplification"),
                "window_evidence": windows,
                "final_drift_l2": payload.get("cumulative", {}).get("actual", {}).get("resultant_l2"),
                "paired_loss_gap_final": payload.get("loss_audit", {}).get("final_gap"),
                "paired_loss_gap_mean_last_512": payload.get("loss_audit", {}).get("last_512_mean"),
                "paired_loss_gap_std_last_512": None,
                "artifact": str(path.relative_to(ROOT)),
            }
    if payload.get("compact_long_horizon") and "statistics" in payload:
        stats = payload["statistics"]
        levels = stats.get("levels", {})
        sidecar = read(path.with_name(path.stem + "_windows.json")) or {}
        side_levels = sidecar.get("levels", {})
        local_side = side_levels.get("local", {})
        feedback_side = side_levels.get("feedback", {})
        actual_side = side_levels.get("actual", {})
        local_robust = bool(local_side.get("long_persistent", False))
        feedback_robust = bool(feedback_side.get("long_persistent", False))
        if local_robust:
            long_status = "ROBUST"
            verdict = "PERSISTENT_LOCAL_BIAS"
        elif feedback_robust:
            long_status = "FEEDBACK_SUSTAINED"
            verdict = "FEEDBACK_SUSTAINED_SEPARATION"
        else:
            long_status = "NOT_ROBUST"
            verdict = "NO_ROBUST_LONG_DIRECT"
        actual = levels.get("actual", {})
        null = actual.get("sign_flip_null", {})
        return {
            "status": "COMPLETE_4096" if int(payload.get("steps", 0)) >= 4096 else payload.get("status", "COMPLETE"),
            "long_direct": long_status,
            "verdict": verdict,
            "steps": int(payload.get("steps", 0)),
            "A4096": float(actual.get("coherence_amplification", 0.0)),
            "local_A4096": float(levels.get("local", {}).get("coherence_amplification", 0.0)),
            "feedback_A4096": float(levels.get("feedback", {}).get("coherence_amplification", 0.0)),
            "null95": null.get("upper_95"),
            "p": null.get("one_sided_p"),
            "late_windows": actual_side.get("late_windows"),
            "late_windows_above_own_null": actual_side.get("late_windows_above_own_null"),
            "loss_audit": payload.get("loss_audit", {}),
            "final_drift_l2": payload.get("final_drift_l2"),
            "paired_loss_gap_final": payload.get("loss_audit", {}).get("final_gap"),
            "paired_loss_gap_mean_last_512": payload.get("loss_audit", {}).get("last_512_mean"),
            "paired_loss_gap_std_last_512": None,
            "artifact": str(path.relative_to(ROOT)),
            "window_evidence": str(path.with_name(path.stem + "_windows.json").relative_to(ROOT)) if sidecar else None,
            "measurement_geometry": "COUNT_SKETCH_256",
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
                "paired_loss_gap_final": payload.get("loss_audit", {}).get("final_gap"),
                "paired_loss_gap_mean_last_512": payload.get("loss_audit", {}).get("last_512_mean"),
                "paired_loss_gap_std_last_512": None,
                "loss_audit": payload.get("loss_audit", {}),
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
    if payload.get("schema", "").startswith("kernel-analyzer-gemma4-v3-consequence") and str(payload.get("status", "")).startswith("COMPLETE"):
        stats = payload.get("statistics", {})
        short_path = path.with_name("short_screen.json")
        short = read(short_path) or {}
        short_cases = {str(row.get("case_id")): row for row in short.get("cases", [])}
        feedback = short_cases.get(f"{payload.get('case_id')}::feedback", {})
        local = short_cases.get(f"{payload.get('case_id')}::local", {})
        actual = short_cases.get(f"{payload.get('case_id')}::actual", {})
        if feedback.get("status") == "RISK_CANDIDATE":
            long_status = "FEEDBACK_SUSTAINED"
        elif local.get("status") == "RISK_CANDIDATE":
            long_status = "ROBUST"
        else:
            long_status = "NOT_ROBUST"
        return {
            "status": "COMPLETE_4096" if int(payload.get("steps", 0)) >= 4096 else payload.get("status", "COMPLETE"),
            "long_direct": long_status,
            "verdict": "FEEDBACK_SUSTAINED_SEPARATION" if long_status == "FEEDBACK_SUSTAINED" else ("PERSISTENT_LOCAL_BIAS" if long_status == "ROBUST" else "NO_ROBUST_LONG_DIRECT"),
            "steps": int(payload.get("steps", 0)),
            "A4096": float(actual.get("observed_amplification", stats.get("actual", {}).get("coherence_amplification", 0.0))),
            "local_A4096": float(local.get("observed_amplification", stats.get("local", {}).get("coherence_amplification", 0.0))),
            "feedback_A4096": float(feedback.get("observed_amplification", stats.get("feedback", {}).get("coherence_amplification", 0.0))),
            "null95": feedback.get("sign_flip_null", {}).get("upper_95") or local.get("sign_flip_null", {}).get("upper_95"),
            "p": feedback.get("sign_flip_null", {}).get("one_sided_p") or local.get("sign_flip_null", {}).get("one_sided_p"),
            "final_drift_l2": payload.get("final_drift_l2"),
            "paired_loss_gap_final": payload.get("paired_loss_gap_final"),
            "paired_loss_gap_mean_last_512": payload.get("paired_loss_gap_mean_last_512"),
            "paired_loss_gap_std_last_512": payload.get("paired_loss_gap_std_last_512"),
            "loss_audit": {"recorded": bool(payload.get("records")), "any_period_split": any(abs(float(row.get("paired_loss_gap_candidate_minus_repair", 0.0))) > 1e-8 for row in payload.get("records", []))},
            "artifact": str(path.relative_to(ROOT)),
            "window_evidence": str(short_path.relative_to(ROOT)) if short_path.exists() else None,
        }
    if payload.get("schema", "").startswith("kernel-analyzer-target-v3-consequence") and str(payload.get("status", "")).startswith("COMPLETE"):
        # The generic target runner uses the same sketch/null protocol as the
        # Gemma runner, but also supports ordinary text-only LMs.  Its sibling
        # short_screen.json is the long-horizon window evidence.
        short_path = path.with_name("short_screen.json")
        short = read(short_path) or {}
        short_cases = {str(row.get("case_id")): row for row in short.get("cases", [])}
        case_id = str(payload.get("case_id", ""))
        local = short_cases.get(f"{case_id}::local", {})
        feedback = short_cases.get(f"{case_id}::feedback", {})
        actual = short_cases.get(f"{case_id}::actual", {})
        local_risk = local.get("status") == "RISK_CANDIDATE"
        feedback_risk = feedback.get("status") == "RISK_CANDIDATE"
        long_status = "ROBUST" if local_risk else ("FEEDBACK_SUSTAINED" if feedback_risk else "NOT_ROBUST")
        chosen = local if local_risk else (feedback if feedback_risk else actual)
        return {
            "status": "COMPLETE_4096" if int(payload.get("steps", 0)) >= 4096 else payload.get("status", "COMPLETE"),
            "long_direct": long_status,
            "verdict": "PERSISTENT_LOCAL_BIAS" if local_risk else ("FEEDBACK_SUSTAINED_SEPARATION" if feedback_risk else "NO_ROBUST_LONG_DIRECT"),
            "steps": int(payload.get("steps", 0)),
            "A4096": float(chosen.get("observed_amplification", 0.0)),
            "local_A4096": float(local.get("observed_amplification", 0.0)),
            "feedback_A4096": float(feedback.get("observed_amplification", 0.0)),
            "null95": chosen.get("sign_flip_null", {}).get("upper_95"),
            "p": chosen.get("sign_flip_null", {}).get("one_sided_p"),
            "paired_loss_gap_final": payload.get("loss_audit", {}).get("final_gap"),
            "paired_loss_gap_mean_last_512": payload.get("loss_audit", {}).get("last_512_mean"),
            "loss_audit": payload.get("loss_audit", {}),
            "artifact": str(path.relative_to(ROOT)),
            "window_evidence": str(short_path.relative_to(ROOT)) if short_path.exists() else None,
            "measurement_geometry": "COUNT_SKETCH_256",
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
    recent_block = payload.get("last_512_train_steps", {})
    recent = recent_block.get("paired_loss_gap", {}) or recent_block.get("loss_gap_candidate_minus_repair", {})
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
        "steps": (
            payload.get("protocol", {}).get("steps")
            or payload.get("protocol", {}).get("measurement_steps")
            or payload.get("steps_completed")
            or payload.get("steps")
            or (len(payload.get("train_rows", [])) if payload.get("train_rows") else None)
        ),
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
    ("Qwen64 v_proj", "Qwen3-1.7B", "seq64 v_proj MM + output rounding", "conditional source asymmetry", LONG / "qwen64_vproj_output.json", LONG / "qwen64_vproj_live_refresh_live_paired.json"),
    ("Qwen v_proj", "Qwen3-1.7B", "v_proj MM/output rounding", "local arithmetic/pairing", LONG / "qwen_vproj_output.json", LONG / "qwen_vproj_live_refresh_live_paired.json"),
    ("Mamba in_proj", "Mamba-130M", "in_proj matrix multiply", "local arithmetic/pairing", LONG / "mamba_in_proj.json", LONG / "mamba_in_proj_live_refresh_live_paired.json"),
    ("saved-P", "Qwen3-1.7B", "layer-27 saved-P softmax backward", "response/state-contract imbalance", LONG / "qwen_saved_p.json", LONG / "qwen_saved_p_live_refresh_live_paired.json"),
    ("Qwen3-VL SiLU", "Qwen3-VL-Reranker-2B", "SiLU backward", "response asymmetry", LONG / "qwen3vl_silu_4096.json", None),
    ("layer-23 attention", "Qwen3-1.7B", "attention S_bwd/K to q_proj", "event/pairing imbalance", LONG / "layer23_qproj_attention_state_region.json", None),
    ("DeepSeek layer-35 dV", "DeepSeek-R1-Qwen3-8B", "attention dV BMM", "formation unresolved; event/pairing candidate", LONG / "deepseek_l35_dv.json", None),
    ("Llama lm_head dX", "Llama-3.2-3B", "lm_head backward dX", "event/pairing family replication", ROOT / "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/lmhead_consequence4096.json", ROOT / "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/lmhead_consequence4096.json"),
    ("Ministral lm_head dX", "Ministral-3-3B", "lm_head backward dX", "event/pairing family replication", ROOT / "results/property/tcmp_allop_v1/heldout/ministral3_3b_text128/lmhead_consequence4096.json", ROOT / "results/property/tcmp_allop_v1/heldout/ministral3_3b_text128/lmhead_consequence4096.json"),
    ("Gemma4 RMSNorm", "Gemma-4 E2B", "RMSNorm / projection feedback region", "response asymmetry / feedback candidate", ROOT / "results/property/declared_persistent_4096/gemma4_norm_v3_long_projection/consequence.json", ROOT / "results/property/declared_persistent_4096/gemma4_norm_v3_long_projection/consequence.json"),
]

HISTORICAL_MATRIX_IDS = {
    "Liger fused CE": "liger_fused_ce_t128",
    "Phi lm_head dX": "phi4_seq64_lmhead_dx",
    "Qwen lm_head dX": "qwen_seq128_lmhead_dx",
    "Qwen64 v_proj": "qwen64_vproj",
    "Qwen v_proj": "qwen128_vproj_output",
    "Mamba in_proj": "mamba_seq64_in_proj",
    "saved-P": "qwen_saved_p_seq128",
    "Qwen3-VL SiLU": "qwen3vl_silu_backward",
    "layer-23 attention": "layer23_qproj_attention_state_region",
    "DeepSeek layer-35 dV": "deepseek8b_seq64_l35_attention_dv",
    "Gemma4 RMSNorm": "gemma4_norm",
}

# The 12 rows below were not chosen after seeing a 4096-step result.  They
# were mechanically drawn from the frozen result-blind screen-negative pool,
# but their 32-step consequence files showed an actual/feedback separation.
# Under the broader audit rule they are valid long-consequence candidates,
# not negatives.  Keep their exact runtime provenance here so an interrupted
# replay remains visible as unresolved instead of disappearing into the
# screen-negative count.
EXPANDED_CANDIDATES = [
    ("multishape-backward-cell-0057", "DeepSeek-R1-Qwen3-8B", "DeepSeek backward cell 0057; post-attention LayerNorm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0103", "DeepSeek-R1-Qwen3-8B", "DeepSeek backward cell 0103; input LayerNorm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0153", "DeepSeek-R1-Qwen3-8B", "DeepSeek backward cell 0153; attention k-norm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0190", "DeepSeek-R1-Qwen3-8B", "DeepSeek backward cell 0190; attention q-norm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0191", "DeepSeek-R1-Qwen3-8B", "DeepSeek backward cell 0191; attention q-norm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0450", "Mamba-130M", "Mamba backward cell 0450; dt-projection bias carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0501", "Phi-4-mini", "Phi backward cell 0501; post-attention LayerNorm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0508", "Phi-4-mini", "Phi backward cell 0508; post-attention LayerNorm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0543", "Phi-4-mini", "Phi backward cell 0543; final norm carrier", "small mixed candidate"),
    ("multishape-backward-cell-0654", "Qwen3-1.7B", "Qwen backward cell 0654; input LayerNorm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0745", "Qwen3-1.7B", "Qwen backward cell 0745; attention q-norm carrier", "feedback-sustained candidate"),
    ("multishape-backward-cell-0747", "Qwen3-1.7B", "Qwen backward cell 0747; attention k-norm carrier", "feedback-sustained candidate"),
]

# A separate held-out Gemma row with a real 32-step consequence artifact.  It
# is kept outside the historical/12-row denominators until its 4096-step
# replay finishes.
ADDITIONAL_OUTCOME_CANDIDATES = [
    (
        "gemma4_random_gelu_loss_backward",
        "Gemma-4 E2B",
        "GELU/loss backward region 1401; projection carrier",
        "response asymmetry / feedback candidate",
    ),
]


def expanded_long_path(case_id: str) -> Path:
    return LONG / "expanded_controls" / f"{case_id}_4096.json"


def retry_is_in_progress(case_id: str) -> bool:
    """A stale first-attempt failure must not hide a later live retry."""
    log = LONG / "expanded_controls/logs" / f"{case_id}_safe.log"
    if not log.exists():
        return False
    lines = log.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if "START " in line and "compact=yes" in line]
    if not starts:
        return False
    tail = lines[starts[-1] + 1 :]
    return not any("FAILED " in line or "COMPLETE " in line for line in tail)


def gemma_gelu_retry_is_in_progress() -> bool:
    """A queued/active Gemma GELU retry is not a negative result."""
    if GEMMA_GELU_LONG.exists() or not GEMMA_GELU_LOG.exists():
        return False
    lines = GEMMA_GELU_LOG.read_text(errors="replace").splitlines()
    return bool(lines) and not any(
        "COMPLETE Gemma GELU long" in line or "UNRESOLVED Gemma GELU long" in line
        for line in lines
    )


def gemma_norm_retry_is_in_progress() -> bool:
    """Do not turn an older Gemma runtime failure into a negative while a fresh retry runs."""
    if GEMMA_NORM_LOG.exists() and (LONG / "gemma4_norm_v3_long_projection" / "consequence.json").exists():
        return False
    if not GEMMA_NORM_LOG.exists():
        return False
    lines = GEMMA_NORM_LOG.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if "START Gemma v3 long" in line]
    if not starts:
        return False
    tail = lines[starts[-1] + 1 :]
    return not any("COMPLETE Gemma v3 long" in line or "UNRESOLVED Gemma v3 long" in line for line in tail)


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
        # A paired loss split without a long-range bias component is retained
        # as a consequence-only control. It must not be promoted to a
        # persistent-bias case: a bias-bearing direct or feedback component
        # must itself survive the long horizon.
        if str(paired.get("status", "")).startswith("COMPLETE") and paired.get("loss_separation_observed"):
            return "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE"
        return "NO_ROBUST_LONG_DIRECT_BIAS"
    if direct.get("status") == "NOT_RUN":
        if "formation unresolved" in formation_path:
            return "UNRESOLVED_FORMATION"
        if "no direct source" in formation_path:
            return "NOT_ESCALATED_NO_DIRECT_SOURCE"
        return "NOT_RUN"
    if direct.get("status") == "UNRESOLVED_LONG_REPLAY_PENDING":
        return "UNRESOLVED_LONG_REPLAY_PENDING"
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
    expanded_ids = {case_id for case_id, *_ in EXPANDED_CANDIDATES}
    known.update(expanded_ids)
    known.update(case_id for case_id, *_ in ADDITIONAL_OUTCOME_CANDIDATES)
    target_manifest = read(TARGET_MANIFEST) or {}
    target_manifest_rows = target_manifest.get("rows", [])
    known.update(row.get("case_id") for row in target_manifest_rows if row.get("case_id"))
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
        if name == "Gemma4 RMSNorm":
            rebound_path = LONG / "gemma4_norm_v3_long_projection/consequence.json"
            projection_unresolved = LONG / "unresolved/gemma4_norm_v3_long_projection_unresolved.json"
            rebound = read(rebound_path) or {}
            if str(rebound.get("status", "")).startswith("COMPLETE"):
                long_path = rebound_path
            elif projection_unresolved.exists():
                long_path = projection_unresolved
            elif (LONG / "unresolved/gemma4_norm_4096_rebound_unresolved.json").exists():
                long_path = LONG / "unresolved/gemma4_norm_4096_rebound_unresolved.json"
        # A refreshed long replay supersedes an older direct-only artifact,
        # but only once the new file is complete.  This keeps interrupted
        # runs from silently changing the audit.
        refreshed = {
            "Qwen64 v_proj": LONG / "qwen64_vproj_live_refresh.json",
            "Qwen v_proj": LONG / "qwen_vproj_live_refresh.json",
            "Mamba in_proj": LONG / "mamba_in_proj_live_refresh.json",
            "saved-P": LONG / "qwen_saved_p_live_refresh.json",
        }.get(name)
        if refreshed is not None and refreshed.exists():
            candidate_payload = read(refreshed) or {}
            if candidate_payload.get("status") == "COMPLETE":
                long_path = refreshed
        gemma_norm_pending = name == "Gemma4 RMSNorm" and gemma_norm_retry_is_in_progress()
        if name == "Gemma4 RMSNorm" and (gemma_norm_pending or not long_path.exists()):
            projection_unresolved = LONG / "unresolved/gemma4_norm_v3_long_projection_unresolved.json"
            if gemma_norm_retry_is_in_progress():
                direct = {
                    "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                    "long_direct": "UNRESOLVED",
                    "reason": "A same-process Gemma RMSNorm 4096-step retry is currently running; the earlier runtime failure is not treated as a negative.",
                    "artifact": str((LONG / "gemma4_norm_v3_long_projection" / "consequence.json").relative_to(ROOT)),
                }
            elif projection_unresolved.exists():
                unresolved = read(projection_unresolved) or {}
                direct = {
                    "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY_RESOURCE"),
                    "long_direct": "UNRESOLVED",
                    "reason": unresolved.get("reason", "long replay unavailable"),
                    "artifact": str(projection_unresolved.relative_to(ROOT)),
                }
            else:
                direct = {
                    "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                    "long_direct": "UNRESOLVED",
                    "reason": "A resource-safe rebound replay is still running; the historical failure is not treated as a negative.",
                    "artifact": str((LONG / "gemma4_norm_4096_rebound.json").relative_to(ROOT)),
                }
        elif not long_path.exists() and name in unresolved_paths and unresolved_paths[name].exists():
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
        paired_unresolved_paths = {
            "Mamba in_proj": LONG / "unresolved/mamba_in_proj_live_refresh_live_paired_unresolved.json",
        }
        if paired_path and paired_path.exists():
            paired = paired_row(paired_path)
        elif name in paired_unresolved_paths and paired_unresolved_paths[name].exists():
            paired = paired_row(paired_unresolved_paths[name])
        else:
            paired = {"status": "NOT_RUN"}
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
            "matrix_case_id": HISTORICAL_MATRIX_IDS.get(name),
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

    # These rows passed the result-blind 32-step consequence screen and are
    # therefore valid long-audit candidates.  Until a 4096-step artifact is
    # present they remain explicitly unresolved, never screen negatives.
    consequence_summary_path = ROOT / "results/property/joint_bias_formation_v1/consequence_summary.json"
    consequence_rows = {}
    if consequence_summary_path.exists():
        summary = json.loads(consequence_summary_path.read_text())
        consequence_rows = {str(row.get("case_id")): row for row in summary.get("rows", [])}
    for case_id, model, region, formation_path in EXPANDED_CANDIDATES:
        path = expanded_long_path(case_id)
        unresolved_path = LONG / "unresolved" / f"{case_id}_4096_unresolved.json"
        if path.exists():
            direct = long_row(path)
            loss_audit = direct.get("loss_audit", {}) or {}
            paired = {
                "status": "COMPLETE_IN_LONG_REPLAY",
                "loss_separation_observed": bool(loss_audit.get("any_period_split")),
                "steps": direct.get("steps"),
                "parameter_distance": direct.get("final_drift_l2"),
                "final_loss_gap": direct.get("paired_loss_gap_final"),
                "recent_loss_gap_mean": direct.get("paired_loss_gap_mean_last_512"),
                "recent_loss_gap_std": direct.get("paired_loss_gap_std_last_512"),
                "loss_audit": loss_audit,
                "artifact": direct.get("artifact"),
            }
        elif retry_is_in_progress(case_id):
            direct = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "long_direct": "UNRESOLVED",
                "reason": "A compact 4096-step retry is currently in progress; the earlier attempt failure is not treated as a negative.",
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
            paired = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "loss_separation_observed": False,
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
        elif (LONG / "expanded_controls" / "retry_failures" / f"{case_id}.json").exists() or (
            unresolved_path.exists() and case_id not in RETRY_SCHEDULED_IDS
        ):
            chosen_unresolved = (
                LONG / "expanded_controls" / "retry_failures" / f"{case_id}.json"
                if (LONG / "expanded_controls" / "retry_failures" / f"{case_id}.json").exists()
                else unresolved_path
            )
            unresolved = read(chosen_unresolved) or {}
            direct = {
                "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY"),
                "long_direct": "UNRESOLVED",
                "reason": unresolved.get("reason", "long replay unavailable"),
                "steps": unresolved.get("steps_completed"),
                "artifact": str(chosen_unresolved.relative_to(ROOT)),
            }
            paired = {
                "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY"),
                "loss_separation_observed": False,
                "steps": unresolved.get("steps_completed"),
                "artifact": str(chosen_unresolved.relative_to(ROOT)),
            }
        else:
            direct = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "long_direct": "UNRESOLVED",
                "reason": "The 32-step result-blind consequence candidate still requires a 4096-step live candidate/repair replay.",
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
            paired = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "loss_separation_observed": False,
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
        rows.append({
            "case": case_id,
            "matrix_case_id": case_id,
            "model": model,
            "operator_or_region": region,
            "formation_path": formation_path,
            "long_direct": direct,
            "paired_consequence": paired,
            "scope": "expanded_bias_candidate",
            "final_label": final_label(direct, paired, formation_path),
            "direct_sha256": sha(path),
            "paired_sha256": sha(path),
            "short_consequence_artifact": consequence_rows.get(case_id, {}).get("source"),
            "short_regime": consequence_rows.get(case_id, {}).get("regime"),
        })

    # The Gemma GELU/loss row is a genuine residual-nonzero, parameter-
    # reachable held-out consequence candidate, but it was not part of the
    # original 12-row mechanical sample.  Keep it visible while the dedicated
    # 4096 retry is running; an old runtime failure is never converted into a
    # negative label.
    for case_id, model, region, formation_path in ADDITIONAL_OUTCOME_CANDIDATES:
        path = GEMMA_GELU_LONG
        if path.exists():
            direct = long_row(path)
            loss_audit = direct.get("loss_audit", {}) or {}
            paired = {
                "status": "COMPLETE_IN_LONG_REPLAY",
                "loss_separation_observed": bool(loss_audit.get("any_period_split")),
                "steps": direct.get("steps"),
                "parameter_distance": direct.get("final_drift_l2"),
                "final_loss_gap": direct.get("paired_loss_gap_final"),
                "recent_loss_gap_mean": direct.get("paired_loss_gap_mean_last_512"),
                "recent_loss_gap_std": direct.get("paired_loss_gap_std_last_512"),
                "loss_audit": loss_audit,
                "artifact": direct.get("artifact"),
            }
        elif gemma_gelu_retry_is_in_progress():
            direct = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "long_direct": "UNRESOLVED",
                "reason": "The dedicated Gemma GELU 4096-step retry is queued or running; no negative label is assigned.",
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
            paired = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "loss_separation_observed": False,
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
        elif GEMMA_GELU_UNRESOLVED.exists():
            unresolved = read(GEMMA_GELU_UNRESOLVED) or {}
            direct = {
                "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY"),
                "long_direct": "UNRESOLVED",
                "reason": unresolved.get("claim_boundary", "Gemma GELU long replay unavailable"),
                "steps": unresolved.get("steps_completed"),
                "artifact": str(GEMMA_GELU_UNRESOLVED.relative_to(ROOT)),
            }
            paired = {
                "status": unresolved.get("status", "UNRESOLVED_LONG_REPLAY"),
                "loss_separation_observed": False,
                "steps": unresolved.get("steps_completed"),
                "artifact": str(GEMMA_GELU_UNRESOLVED.relative_to(ROOT)),
            }
        else:
            direct = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "long_direct": "UNRESOLVED",
                "reason": "A dedicated 4096-step Gemma GELU replay has been scheduled but has not started yet.",
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
            paired = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "loss_separation_observed": False,
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
        rows.append({
            "case": case_id,
            "matrix_case_id": case_id,
            "model": model,
            "operator_or_region": region,
            "formation_path": formation_path,
            "long_direct": direct,
            "paired_consequence": paired,
            "scope": "additional_outcome_candidate",
            "final_label": final_label(direct, paired, formation_path),
            "direct_sha256": sha(path),
            "paired_sha256": sha(path),
            "short_consequence_artifact": str((ROOT / "results/property/direct_persistence_v4/heldout/gemma4_random_gelu_loss_backward.consequence.json").relative_to(ROOT)),
            "short_regime": "feedback-sustained 32-step consequence candidate",
        })

    # New Gemma/Llama scan targets are part of the same audit file, but they
    # stay unresolved until exact F+B reachability and the long replay are
    # both present.  A pattern-screen amplification is never a final label.
    for target in target_manifest_rows:
        case_id = str(target["case_id"])
        target_dir = LONG / "operator_scan_targets" / case_id
        path = target_dir / "consequence.json"
        pilot_dirs = sorted(LONG.glob(f"operator_scan_targets/{case_id}_pilot*"))
        completed_pilots = [d for d in pilot_dirs if (d / "prediction.json").exists() and (d / "short_screen.json").exists()]
        pilot_dir = completed_pilots[-1] if completed_pilots else None
        if path.exists():
            direct = long_row(path)
            paired = paired_row(path)
        elif pilot_dir is not None:
            prediction = read(pilot_dir / "prediction.json") or {}
            short = read(pilot_dir / "short_screen.json") or {}
            short_cases = short.get("cases", [])
            # A zero-energy pilot is not a negative result for a frozen scan
            # target.  It can mean that the freshly compiled wrapper did not
            # bind the historical screen endpoint (especially when Inductor
            # renumbered or fused repeated regions).  Keep it unresolved
            # until the target has a nonzero, parameter-reachable contrast.
            zero_energy = bool(short_cases) and all(
                row.get("status") == "UNRESOLVED_ZERO_ENERGY"
                for row in short_cases
            )
            risk = prediction.get("source_prediction") == "SOURCE_PERSISTENCE_RISK" or any(
                row.get("status") == "RISK_CANDIDATE" for row in short_cases
            )
            if zero_energy:
                direct = {
                    "status": "UNRESOLVED_PARAMETER_BINDING",
                    "long_direct": "UNRESOLVED",
                    "reason": "The target pilot produced zero candidate/repair energy; this is retained as unresolved because the historical screen endpoint may not have rebound to the freshly compiled callsite.",
                    "steps": 16,
                    "artifact": str((pilot_dir / "short_screen.json").relative_to(ROOT)),
                }
                paired = {
                    "status": "UNRESOLVED_PARAMETER_BINDING",
                    "loss_separation_observed": False,
                    "steps": 16,
                    "artifact": str((pilot_dir / "consequence.json").relative_to(ROOT)),
                }
            elif risk:
                direct = {
                    "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                    "long_direct": "UNRESOLVED",
                    "reason": "The frozen target pilot is risk-positive; its 4096-step consequence is still required.",
                    "steps": 16,
                    "artifact": str(path.relative_to(ROOT)),
                }
                paired = {
                    "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                    "loss_separation_observed": False,
                    "steps": 16,
                    "artifact": str(path.relative_to(ROOT)),
                }
            else:
                direct = {
                    "status": "COMPLETE_SHORT_NO_RISK",
                    "long_direct": "NOT_ROBUST",
                    "reason": "The frozen 16-step target pilot did not trigger the source or local/feedback risk screen; no 4096-step consequence was escalated.",
                    "steps": 16,
                    "artifact": str((pilot_dir / "short_screen.json").relative_to(ROOT)),
                }
                paired = {"status": "NOT_ESCALATED", "loss_separation_observed": False, "steps": 0}
        else:
            direct = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "long_direct": "UNRESOLVED",
                "reason": "Frozen Gemma/Llama target requires a parameter-reachable same-process F+B replay before it can be called a bias case.",
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
            paired = {
                "status": "UNRESOLVED_LONG_REPLAY_PENDING",
                "loss_separation_observed": False,
                "steps": None,
                "artifact": str(path.relative_to(ROOT)),
            }
        rows.append({
            "case": case_id,
            "matrix_case_id": case_id,
            "model": target["model"],
            "operator_or_region": f"{target['target_region']} {target['target_symbol']} [{target['target_endpoint']}]",
            "formation_path": "new operator scan; exact generated target replay",
            "long_direct": direct,
            "paired_consequence": paired,
            "scope": "operator_scan_candidate",
            "final_label": final_label(direct, paired, "new operator scan"),
            "direct_sha256": sha(path),
            "paired_sha256": sha(path),
            "screen_metadata": {
                "screen_exact_implementation_id": target.get("screen_exact_implementation_id"),
                "screen_amplification": target.get("screen_amplification"),
                "screen_p_value": target.get("screen_p_value"),
                "source_screen": target.get("source_screen"),
            },
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
                "matrix_case_id": case_id,
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
    operator_scan_path = ROOT / "results/property/declared_persistent_4096/operator_scan_gemma_llama.json"
    operator_scan = read(operator_scan_path) or {}
    operator_scan_models = operator_scan.get("models", [])
    operator_scan_summary = {
        "artifact": str(operator_scan_path.relative_to(ROOT)) if operator_scan_path.exists() else None,
        "models": [
            {
                "model": row.get("model"),
                "screened_rows": row.get("screened_rows", 0),
                "frozen_gate_positives": row.get("bh_positive_rows", 0),
                "nonzero_rows_without_legal_replay": row.get("nonzero_new_impl_rows_requiring_legal_replay", 0),
                "status": "SCREEN_ONLY_NO_NEW_LONG_CANDIDATE",
            }
            for row in operator_scan_models
        ],
        "screened_rows": sum(int(row.get("screened_rows", 0)) for row in operator_scan_models),
        "frozen_gate_positives": sum(int(row.get("bh_positive_rows", 0)) for row in operator_scan_models),
        "nonzero_rows_without_legal_replay": sum(int(row.get("nonzero_new_impl_rows_requiring_legal_replay", 0)) for row in operator_scan_models),
        "claim_boundary": "Pattern-screen signals are not bias cases without an exact parameter-reachable repair and long replay; no new Gemma/Llama row passed the frozen gate.",
    }
    operator_scan_target_summary = {
        "manifest": str(TARGET_MANIFEST.relative_to(ROOT)) if TARGET_MANIFEST.exists() else None,
        "rows": len(target_manifest_rows),
        "completed": sum((LONG / "operator_scan_targets" / str(row.get("case_id")) / "consequence.json").exists() for row in target_manifest_rows),
        "claim_boundary": "Rows are selected before target replay results. Pattern-screen amplification alone is never promoted to a bias case.",
    }
    payload = {
        "schema": "all-historical-bias-candidates-long-audit-v1",
        "definition": "A final persistent-bias case requires a bias-bearing component itself to survive the 4096-step horizon: either robust direct source direction, or robust feedback effective-update direction, and a paired parameter/loss split when the corresponding consequence run is available. A loss split alone is consequence-only and never establishes persistent bias. Different converged losses are not required or claimed.",
        "long_direct_rule": "one-sided sign-flip p<=0.05 and at least 75% of late rolling windows directional",
        "case_count": len(rows),
        "unique_matrix_case_count": matrix_count,
        "endpoint_population_count": endpoint_population_count,
        "coherent_endpoint_witness_count": coherent_endpoint_count,
        "grouping_rule": "Repeated concrete endpoint occurrences are grouped by the frozen case-stage matrix ID; they are not silently dropped, but they do not count as independent cases.",
        "historical_candidate_count": 11,
        "extended_candidate_count": len(CASES) + len(EXPANDED_CANDIDATES) + len(ADDITIONAL_OUTCOME_CANDIDATES) + len(target_manifest_rows),
        "extended_candidate_note": "The extended roster includes historical candidates, Llama/Ministral family replication rows, Gemma4, the 12 result-blind 32-step consequence candidates, the separately tracked Gemma GELU consequence candidate, and six new Gemma/Llama target-replay rows. These rows are not negatives until their long replay is complete.",
        "operator_scan": operator_scan_summary,
        "operator_scan_target_replay": operator_scan_target_summary,
        "unindexed_historical_candidate_count": len(historical_ids - matrix_ids),
        "matrix_case_ids_covered_by_rows": sorted(
            {r.get("matrix_case_id") for r in rows if r.get("matrix_case_id")}
        ),
        "matrix_case_ids_missing_from_rows": sorted(
            matrix_ids - {r.get("matrix_case_id") for r in rows if r.get("matrix_case_id")}
        ),
        "scope_counts": {
            "historical_candidate": sum(r.get("scope") == "historical_candidate" for r in rows),
            "roster_control_or_unresolved": sum(r.get("scope") == "roster_control_or_unresolved" for r in rows),
            "expanded_bias_candidate": sum(r.get("scope") == "expanded_bias_candidate" for r in rows),
            "additional_outcome_candidate": sum(r.get("scope") == "additional_outcome_candidate" for r in rows),
            "operator_scan_candidate": sum(r.get("scope") == "operator_scan_candidate" for r in rows),
        },
        "final_case_count": sum(
            r["final_label"] in {
                "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT",
                "ROBUST_LONG_DIRECT_BIAS_LOSS_NOT_RUN",
                "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT",
            }
            for r in rows
        ),
        "label_counts": dict(Counter(r["final_label"] for r in rows)),
        "direct_persistent_case_count": sum(
            r["long_direct"].get("long_direct") == "ROBUST"
            for r in rows
        ),
        "long_loss_split_without_direct_count": sum(
            r["final_label"] == "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE"
            for r in rows
        ),
        # Keep the two scientific questions separate: a persistent bias
        # component is the primary case count, while a paired long-run loss
        # split is a secondary outcome count even when its source component
        # did not remain directional.
        "outcome_relevant_case_count": sum(
            r["final_label"] in {
                "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT",
                "ROBUST_LONG_DIRECT_BIAS_LOSS_NOT_RUN",
                "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT",
                "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE",
            }
            for r in rows
        ),
        "unresolved_or_abstain_count": sum(
            r["final_label"].startswith(("UNRESOLVED", "ABSTAIN"))
            for r in rows
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 所有历史偏差候选的长程复核",
        "",
        f"仓库中有 **{matrix_count} 个唯一主矩阵 case ID**；本审计逐行复核 **{len(CASES) + len(EXPANDED_CANDIDATES) + len(ADDITIONAL_OUTCOME_CANDIDATES)} 个 extended candidate rows**（其中包含历史候选、Llama/Ministral 同族复现、Gemma4、12 个结果盲抽的长程 consequence 候选，以及单独追踪的 Gemma GELU consequence 候选）。合并后表共有 **{len(rows)} 行**。这些数字分别表示覆盖分母、候选分母和逐行审计行数，不能混用。",
        "",
        f"按当前口径，最终计入 **{payload['final_case_count']} 个持久性 bias 案例**：其中直接长程方向案例 **{payload['direct_persistent_case_count']} 个**，反馈维持型案例 **{sum(r['final_label'] == 'FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT' for r in rows)} 个**。另有 **{payload['long_loss_split_without_direct_count']} 个**虽没有持久 bias 组件、但确实出现长程配对 loss 分叉；因此当前共有 **{payload['outcome_relevant_case_count']} 个训练结果相关记录**，但不能把后果对照改称为持久性 bias。未决/不可安全重放共 **{payload['unresolved_or_abstain_count']} 个**，不作阴性判断。",
        "",
        f"Gemma/Llama 的追加首轮扫描另有 **{operator_scan_summary['screened_rows']} 行**，其中冻结升级门通过 **{operator_scan_summary['frozen_gate_positives']} 行**；本轮按冻结规则选出 **{len(target_manifest_rows)} 个**新的 exact F+B 目标，目前完成长程重放 **{operator_scan_target_summary['completed']} 个**。没有完成合法重放的行不计入 bias 案例数，也不改判为阴性。",
        "",
        "持久性 bias 的必要条件是 bias 本身在 4096 步仍然存在：直接源方向或反馈有效更新方向至少有一个通过长程检验。配对训练中的参数或 loss 分叉是后果证据，不足以单独把一个没有持久 bias 组件的记录升级为案例。这里不要求两条训练轨迹收敛到不同的最终 loss，也不作这种声称。",
        "",
        "| 模型 | 算子或位置 | 形成路径 | 4096 步直接结果 | 参数/loss 分叉 | 最终分类 |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT": "最终持久性 bias 案例",
        "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT": "反馈维持型 bias，且有 loss 分叉",
        "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE": "后果对照：有长程 loss 分叉，但没有持久 bias 组件",
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
        "UNRESOLVED_LONG_REPLAY_PENDING": "已通过短程 consequence 筛查，4096 步仍未完成",
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
        elif str(p.get("status", "")).startswith("UNRESOLVED"):
            consequence = "配对长程阶段未能安全完成，未决"
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
    manifest_rows = []
    for row in rows:
        if row.get("scope") != "expanded_bias_candidate":
            continue
        manifest_rows.append({
            "case_id": row["case"],
            "model": row["model"],
            "operator_or_region": row["operator_or_region"],
            "short_consequence_artifact": row.get("short_consequence_artifact"),
            "short_regime": row.get("short_regime"),
            "long_artifact": str(expanded_long_path(row["case"]).relative_to(ROOT)),
            "unresolved_artifact": str((LONG / "unresolved" / f"{row['case']}_4096_unresolved.json").relative_to(ROOT)),
            "trajectory_protocol": "CYCLED_32_STATE_SYNTHETIC_STREAM" if row["model"] == "Phi-4-mini" else "NATURAL_4096_STATE_TRAJECTORY_BANK",
            "status": row["final_label"],
            "claim_boundary": "Only a completed 4096-step candidate/repair replay with an observed paired loss or parameter split can promote this row; missing or failed runtime remains unresolved.",
        })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "schema": "expanded-long-replay-manifest-v1",
        "selection": "All 12 mechanically sampled 32-step consequence rows with actual/feedback separation.",
        "rows": manifest_rows,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "final_case_count": payload["final_case_count"], "case_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
