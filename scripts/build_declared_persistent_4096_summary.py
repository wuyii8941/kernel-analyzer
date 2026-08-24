#!/usr/bin/env python3
"""Summarize the 4096-step audit of every historically declared case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/property/declared_persistent_4096"


CASES = (
    {
        "case_id": "qwen_seq128_lmhead_dx",
        "model": "Qwen3-1.7B",
        "endpoint": "lm_head backward dX",
        "path": RESULT_ROOT / "qwen_lmhead_dx.json",
        "schema": "common",
    },
    {
        "case_id": "liger_fused_ce_t128",
        "model": "Qwen3-1.7B + Liger",
        "endpoint": "fused cross-entropy dW accumulation",
        "path": RESULT_ROOT / "liger_fused_ce.json",
        "schema": "common",
    },
    {
        "case_id": "phi4_seq64_lmhead_dx",
        "model": "Phi-4-mini",
        "endpoint": "lm_head backward dX",
        "path": RESULT_ROOT / "phi_lmhead_dx.json",
        "schema": "phi",
    },
    {
        "case_id": "mamba_seq64_in_proj",
        "model": "Mamba-130M",
        "endpoint": "in_proj matrix multiply",
        "path": RESULT_ROOT / "mamba_in_proj.json",
        "schema": "common",
    },
    {
        "case_id": "qwen_saved_p_seq128",
        "model": "Qwen3-1.7B",
        "endpoint": "layer-27 saved-P softmax backward",
        "path": RESULT_ROOT / "qwen_saved_p.json",
        "schema": "common",
    },
    {
        "case_id": "layer23_qproj_attention_state_region",
        "model": "Qwen3-1.7B",
        "endpoint": "layer-23 attention S_bwd/K to q_proj",
        "path": RESULT_ROOT / "layer23_qproj_attention_state_region.json",
        "schema": "abstain",
    },
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_row(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    exact = payload["statistics"]["exact_full_coordinate"]
    sketch = payload["statistics"]["sketch_diagnostics"]
    windows = exact["rolling_windows"]
    late = windows[len(windows) // 2:]
    late_above = sum(row["coherence_amplification"] > 1.0 for row in late)
    p_value = float(sketch["sign_flip_null"]["one_sided_p"])
    late_fraction = late_above / len(late)
    descriptive = (
        "ROBUST_4096_DIRECT_DIRECTION"
        if p_value <= 0.05 and late_fraction >= 0.75
        else "NO_ROBUST_4096_DIRECT_DIRECTION"
    )
    return {
        **{key: spec[key] for key in ("case_id", "model", "endpoint")},
        "execution_status": "COMPLETE_4096",
        "measurement_steps": int(payload["protocol"]["measurement_steps"]),
        "exact_A4096": float(exact["coherence_amplification"]),
        "sketch_A4096": float(sketch["coherence_amplification"]),
        "sign_flip_null_upper_95": float(sketch["sign_flip_null"]["upper_95"]),
        "sign_flip_one_sided_p": p_value,
        "late_window_count": len(late),
        "late_windows_A_above_one": late_above,
        "late_window_fraction_A_above_one": late_fraction,
        "effective_rank": float(payload["statistics"]["sketch_effective_rank_participation_ratio"]),
        "descriptive_long_result": descriptive,
        "source_artifact": str(spec["path"].relative_to(ROOT)),
        "source_sha256": digest(spec["path"]),
    }


def phi_row(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    full = payload["full"]
    windows = payload["windows"]
    late = windows[len(windows) // 2:]
    late_above = sum(bool(row["above_sign_flip_95"]) for row in late)
    p_value = float(full["sign_flip_null"]["one_sided_p"])
    late_fraction = late_above / len(late)
    descriptive = (
        "ROBUST_4096_DIRECT_DIRECTION"
        if p_value <= 0.05 and late_fraction >= 0.75
        else "NO_ROBUST_4096_DIRECT_DIRECTION"
    )
    return {
        **{key: spec[key] for key in ("case_id", "model", "endpoint")},
        "execution_status": "COMPLETE_4096",
        "measurement_steps": int(payload["protocol"]["measurement_steps"]),
        "exact_A4096": float(full["coherence_amplification"]),
        "sketch_A4096": None,
        "sign_flip_null_upper_95": float(full["sign_flip_null"]["upper_95"]),
        "sign_flip_one_sided_p": p_value,
        "late_window_count": len(late),
        "late_windows_above_own_null": late_above,
        "late_window_fraction_above_own_null": late_fraction,
        "effective_rank": float(
            payload["measurement_geometry"]["effective_rank_participation_ratio"]
        ),
        "descriptive_long_result": descriptive,
        "source_artifact": str(spec["path"].relative_to(ROOT)),
        "source_sha256": digest(spec["path"]),
    }


def abstain_row(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: spec[key] for key in ("case_id", "model", "endpoint")},
        "execution_status": payload["status"],
        "measurement_steps": int(payload["completed_engineering_screen_steps"]),
        "exact_A4096": None,
        "descriptive_long_result": "ABSTAIN_REPAIR_IDENTITY_DRIFT",
        "reason": payload["decision"],
        "source_artifact": str(spec["path"].relative_to(ROOT)),
        "source_sha256": digest(spec["path"]),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 历史有效案例的 4096 步复核",
        "",
        "这里复核的是同一训练状态下，candidate 相对 repair 的直接更新差异。"
        "它不是完整训练收敛，也不包含两条轨迹分开后的反馈。",
        "",
        "| 模型 | 算子或位置 | 执行情况 | 4096 步方向分数 | 后半程窗口 | 结果 |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in payload["cases"]:
        score = "—" if row.get("exact_A4096") is None else f"{row['exact_A4096']:.3f}"
        if "late_window_fraction_A_above_one" in row:
            late = f"{row['late_windows_A_above_one']}/{row['late_window_count']} A>1"
        elif "late_window_fraction_above_own_null" in row:
            late = f"{row['late_windows_above_own_null']}/{row['late_window_count']} 过自身随机上界"
        else:
            late = "—"
        label = {
            "ROBUST_4096_DIRECT_DIRECTION": "4096 步仍有稳定直接方向",
            "NO_ROBUST_4096_DIRECT_DIRECTION": "未形成稳健的 4096 步直接方向",
            "ABSTAIN_REPAIR_IDENTITY_DRIFT": "历史实现不可重放，暂不判断",
            "PENDING": "仍在运行",
        }[row["descriptive_long_result"]]
        lines.append(
            f"| {row['model']} | {row['endpoint']} | {row['execution_status']} | "
            f"{score} | {late} | {label} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 32 步结果只叫短程方向性；本表单独报告 4096 步直接作用。",
        "- `ROBUST` 是便于阅读的事后描述：整体 sign-flip p≤0.05，且后半程至少 75% 的窗口保持方向。原始分数、随机上界和窗口计数仍是主要证据。",
        "- 4096 步直接作用仍不等于 loss 收敛后的功能差异。",
        "- layer-23 的历史编译文件已不可用；当前重编译后 repair 为零差异，因此 fail-closed 地 abstain。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "summary.json")
    parser.add_argument("--markdown", type=Path, default=RESULT_ROOT / "summary.md")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in CASES:
        path = spec["path"]
        if not path.exists():
            missing.append(spec["case_id"])
            rows.append({
                **{key: spec[key] for key in ("case_id", "model", "endpoint")},
                "execution_status": "PENDING",
                "exact_A4096": None,
                "descriptive_long_result": "PENDING",
            })
            continue
        value = json.loads(path.read_text())
        if spec["schema"] == "common":
            rows.append(common_row(spec, value))
        elif spec["schema"] == "phi":
            rows.append(phi_row(spec, value))
        else:
            rows.append(abstain_row(spec, value))
    if missing and not args.allow_incomplete:
        raise RuntimeError("missing 4096 artifacts: " + ", ".join(missing))

    result = {
        "schema": "kernel-analyzer-declared-persistent-4096-summary-v1",
        "status": "INCOMPLETE" if missing else "COMPLETE_WITH_FAIL_CLOSED_ABSTENTION",
        "population": "all six repository records historically declared as 32-step directional/persistent",
        "case_count": len(CASES),
        "complete_4096_count": sum(row["execution_status"] == "COMPLETE_4096" for row in rows),
        "robust_4096_direct_count": sum(
            row["descriptive_long_result"] == "ROBUST_4096_DIRECT_DIRECTION" for row in rows
        ),
        "abstention_count": sum(
            row["descriptive_long_result"] == "ABSTAIN_REPAIR_IDENTITY_DRIFT" for row in rows
        ),
        "missing": missing,
        "cases": rows,
        "claim_boundary": "4096-step same-state direct update audit; not full training convergence or closed-loop feedback",
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown(result))
    print(json.dumps({
        "status": result["status"],
        "complete": result["complete_4096_count"],
        "robust": result["robust_4096_direct_count"],
        "missing": missing,
    }))


if __name__ == "__main__":
    main()
