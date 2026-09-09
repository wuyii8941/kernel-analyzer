#!/usr/bin/env python3
"""Build the v1 machine summary and Markdown table from retained outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/training_numerical_analysis_v1"
SUMMARY = BASE / "summary.json"
DOCUMENT = ROOT / "docs/training_numerical_analysis_v1.md"


def main() -> None:
    roles = json.loads((ROOT / "results/mainline_case_roles.json").read_text())["cases"]
    reports = {}
    for path in sorted((BASE / "recomputed").glob("*.json")):
        payload = json.loads(path.read_text())
        case_id = payload["case_id"]
        current = reports.get(case_id)
        if current is None or payload["claim_scope"] == "FIXED_SUITE_PARAMETER_WRITE":
            reports[case_id] = payload
    recaptures = {}
    for path in sorted((BASE / "recapture").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("case_id"):
            recaptures[payload["case_id"]] = payload
    selected = [
        "phi4_seq64_lmhead_dx", "gemma4_text128_scan_0037",
        "llama32_text128_scan_0000", "liger_fused_ce_t128",
        "deepseek8b_seq256_backward_1714_in_out_ptr0",
        "deepseek8b_seq128_backward_1256_out_ptr0",
    ]
    rows = []
    role_by_id = {row["case_id"]: row for row in roles}
    for case_id in selected:
        report = reports.get(case_id)
        recapture_payload = recaptures.get(case_id)
        if report:
            status = report["measurement_status"]
            endpoint = report["claim_scope"]
            decision = report["equivalence_decision"]
            analysis = report.get("bias_analysis", {})
            rms = analysis.get("fixed_suite_total_rms")
            aligned = (
                analysis.get("fixed_suite_classification", {})
                .get("branches", {}).get("repair_aligned", {})
                .get("simultaneous_confidence_interval")
            )
        elif recapture_payload:
            status = recapture_payload.get("status", "UNKNOWN")
            endpoint = recapture_payload.get("primary_update_endpoint", "NOT_CAPTURED")
            decision = "NOT_RECOMPUTED"
            rms, aligned = None, None
        else:
            status_record = BASE / "status" / f"{case_id}.json"
            if status_record.exists():
                status_payload = json.loads(status_record.read_text())
                status = status_payload["status"]
                endpoint, decision = "NOT_CAPTURED", "NOT_ASSESSED"
                rms, aligned = None, None
            else:
                status, endpoint, decision = "NOT_RUN", "NOT_CAPTURED", "NOT_ASSESSED"
                rms, aligned = None, None
        rows.append({
            "case_id": case_id,
            "role": role_by_id.get(case_id, {}).get("purpose", "PIPELINE_VALIDATION"),
            "measurement_status": status,
            "primary_endpoint": endpoint,
            "equivalence_decision": decision,
            "fixed_suite_total_rms": rms,
            "repair_aligned_interval": aligned,
        })
    SUMMARY.write_text(json.dumps({
        "schema": "kernel-analyzer-training-numerical-analysis-summary-v1",
        "protocol": "results/property/training_numerical_analysis_v1/protocol.json",
        "rows": rows,
        "training_utility": (
            "results/property/training_numerical_analysis_v1/training_utility_summary.json"
            if (BASE / "training_utility_summary.json").is_file() else None
        ),
        "claim_boundary": (
            "Historical v1 rows use an extra-rounded write simulation, not verified "
            "AdamW readback. Original records retain that limitation; see readback-v2."
        ),
    }, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Training Numerical Analysis v1", "",
        "历史限制：v1 使用额外 BF16 舍入的写入模拟，未通过目标 AdamW 写入一致性验收。",
        "当前修正与验收状态见 [readback-v2](training_numerical_analysis_v2.md)。", "",
        "本页由 `scripts/build_training_numerical_analysis_report.py` 从机器记录生成。",
        "它不替代原始结果，也不将重新分析写成未见确认。", "",
        "| 案例 | 用途 | 测量状态 | 主要位置 | 写入差异 RMS | 相对缩放区间 | 等价性判断 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        rms_text = (
            "—" if row["fixed_suite_total_rms"] is None
            else f"{100 * row['fixed_suite_total_rms']:.3f}%"
        )
        aligned_text = (
            "—" if row["repair_aligned_interval"] is None
            else (
                f"[{100 * row['repair_aligned_interval'][0]:.3f}%, "
                f"{100 * row['repair_aligned_interval'][1]:.3f}%]"
            )
        )
        lines.append(
            f"| {row['case_id']} | {row['role']} | {row['measurement_status']} | "
            f"{row['primary_endpoint']} | "
            f"{rms_text} | {aligned_text} | "
            f"{row['equivalence_decision']} |"
        )
    utility_path = BASE / "training_utility_summary.json"
    if utility_path.is_file():
        utility = json.loads(utility_path.read_text())
        loss = utility["validation_loss"]
        context = utility["historical_context"]
        lines.extend([
            "", "## 配对训练验证", "",
            (
                f"新冻结的 Liger 数据流完成 {utility['steps']} 步；被测实现减参考实现的"
                f"验证 loss 为 {loss['candidate_minus_reference']:+.5f}。"
            ),
            (
                "两条历史数据流的对应差为 +0.02778、+0.02821；三条差的描述性区间为 "
                f"[{context['descriptive_t_interval_95'][0]:+.5f}, "
                f"{context['descriptive_t_interval_95'][1]:+.5f}]，跨过零。"
            ),
            "这支持配对轨迹不同，不支持质量变化具有可重复方向。",
        ])
    lines.extend(["", "完整数值与限制见同目录 `summary.json` 及各个重新计算结果。", ""])
    DOCUMENT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
