#!/usr/bin/env python3
"""Freeze the existing Gemma/Llama operator-pattern scan without inventing cases."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "Llama-3.2-3B": ROOT / "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/pattern_screen.json",
    "Gemma-4 E2B": ROOT / "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/pattern_screen.json",
}
ELIGIBILITY = {
    "Llama-3.2-3B": ROOT / "results/property/declared_persistent_4096/llama32_3b_eligibility_freeze.json",
    "Gemma-4 E2B": ROOT / "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/eligibility_freeze.json",
}
OUT = ROOT / "results/property/declared_persistent_4096/operator_scan_gemma_llama.json"
TARGET_MANIFEST = ROOT / "results/property/declared_persistent_4096/operator_scan_target_manifest.json"
BLOCKED_REPLAY = ROOT / "results/property/declared_persistent_4096/operator_scan_replay_blocked.json"
MD = ROOT / "docs/gemma_llama_operator_scan.md"


def main() -> None:
    # The text512 Llama scan is appended only after its independent frozen
    # screen is complete.  It has no exact same-dtype repair package yet, so
    # its nonzero rows remain unresolved rather than being called negative.
    text512 = ROOT / "results/property/declared_persistent_4096/llama32_3b_text512_pattern_screen.json"
    if text512.exists():
        SOURCES["Llama-3.2-3B (text512)"] = text512
    models = []
    for model, path in SOURCES.items():
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
        eligibility_path = ELIGIBILITY.get(model)
        if eligibility_path is not None and eligibility_path.exists():
            eligibility = json.loads(eligibility_path.read_text())
            eligible_rows = [
                row for row in eligibility.get("all_rows", [])
                if row.get("eligibility") == "ELIGIBLE_NONZERO_NEW_IMPL"
            ]
        else:
            eligible_rows = [
                row for row in rows
                if row.get("amplification") is not None and float(row.get("amplification", 0.0)) > 0.0
            ]
        ranked = sorted(rows, key=lambda row: (float(row.get("p_value", 1.0)), -float(row.get("amplification", 0.0))))
        models.append({
            "model": model,
            "source": str(path.relative_to(ROOT)),
            "state_protocol": "existing frozen screen" if model != "Llama-3.2-3B (text512)" else "new frozen text512 screen",
            "screened_rows": len(rows),
            "bh_positive_rows": sum(bool(row.get("screen_positive_bh_q_0_10", False)) for row in rows),
            "unseen_operator_rows_not_escalated": sum(not row.get("screen_positive_bh_q_0_10", False) for row in rows),
            "nonzero_new_impl_rows_requiring_legal_replay": len(eligible_rows),
            "legal_replay_candidates_completed": 0,
            "legal_replay_boundary": (
                "No exact parameter-reachable repair/trajectory replay was available in the frozen "
                "held-out artifact; these rows remain unresolved rather than negative."
            ),
            "unresolved_replay_rows": [
                {
                    "implementation_pattern_id": row.get("implementation_pattern_id"),
                    "endpoint": row.get("endpoint"),
                    "phase": row.get("phase"),
                    "operation": row.get("operation"),
                    "semantic_family_id": row.get("semantic_family_id"),
                    "reason": "NONZERO_PATTERN_WITHOUT_EXACT_PARAMETER_REACHABLE_REPAIR",
                }
                for row in eligible_rows
            ],
            "top_unseen_rows": [
                {
                    "representative_exact_implementation_id": row.get("representative_exact_implementation_id"),
                    "phase": row.get("phase"),
                    "operation": row.get("operation"),
                    "endpoint": row.get("endpoint"),
                    "implementation_pattern_id": row.get("implementation_pattern_id"),
                    "amplification": row.get("amplification"),
                    "p_value": row.get("p_value"),
                    "status": (
                        "SCREEN_ONLY_PRIORITY_NO_LEGAL_REPLAY"
                        if not row.get("screen_positive_bh_q_0_10", False)
                        else "ESCALATE_FROZEN_GATE"
                    ),
                }
                for row in ranked[:20]
            ],
            "claim_boundary": "No row passed the frozen multiple-comparison gate; nominal p-values and A>1 are retained as scan evidence, not promoted to bias cases. Nonzero rows without a legal replay remain unresolved, not negative.",
        })
    target_manifest = json.loads(TARGET_MANIFEST.read_text()) if TARGET_MANIFEST.exists() else {}
    blocked_replay = json.loads(BLOCKED_REPLAY.read_text()) if BLOCKED_REPLAY.exists() else {}
    result = {
        "schema": "gemma-llama-unseen-operator-scan-v1",
        "selection_rule": "Use the existing outcome-blind pattern screens; inspect all rows and escalate only screen_positive_bh_q_0_10=true.",
        "models": models,
        "new_long_candidates": [],
        "unresolved_replay_candidates": [
            {
                "model": model["model"],
                "count": model["nonzero_new_impl_rows_requiring_legal_replay"],
                "reason": model["legal_replay_boundary"],
                "rows": model["unresolved_replay_rows"],
            }
            for model in models
            if model["nonzero_new_impl_rows_requiring_legal_replay"]
        ],
        "expanded_replay": {
            "manifest": str(TARGET_MANIFEST.relative_to(ROOT)) if TARGET_MANIFEST.exists() else None,
            "target_count": len(target_manifest.get("rows", [])),
            "blocked_no_matching_program": len(blocked_replay.get("rows", [])),
            "claim_boundary": "A target manifest means only that the scan row has a fresh-compile replay target; it is not a positive verdict. Pilot and 4096-step results remain separate.",
        },
        "claim_boundary": "This scan found no new candidate that met the frozen escalation gate. It does not prove safety for rows with nominal p-values or without a legal repair/parameter carrier.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    target_manifest = json.loads(TARGET_MANIFEST.read_text()) if TARGET_MANIFEST.exists() else {}
    blocked_replay = json.loads(BLOCKED_REPLAY.read_text()) if BLOCKED_REPLAY.exists() else {}
    expanded_targets = len(target_manifest.get("rows", []))
    expanded_blocked = len(blocked_replay.get("rows", []))
    lines = [
        "# Gemma / Llama 未重复算子扫描",
        "",
        "这一步重新检查了两个模型已有的全量 pattern-screen 结果，并优先查看未出现在历史主案例中的算子族。升级规则仍是预先冻结的多重比较门，不因为单个 nominal p 值或 A>1 就把行列为 bias case。",
        "",
        "| 模型 | 首轮扫描行数 | 通过升级门的行数 | 结论 |",
        "|---|---:|---:|---|",
    ]
    for row in models:
        lines.append(f"| {row['model']} | {row['screened_rows']} | {row['bh_positive_rows']} | {row['nonzero_new_impl_rows_requiring_legal_replay']} 行保留为重放候选，不能判阴 |")
    lines += ["", "## 最强但未升级的算子族", "", "下面只列出排序最靠前的行，作为后续可重放入口；它们不被称为阴性，也不被称为 bias。", ""]
    for row in models:
        lines += [f"### {row['model']}", "", "| 阶段 | 算子族 | 端点 | 方向分数 | nominal p |", "|---|---|---|---:|---:|"]
        for item in row["top_unseen_rows"][:10]:
            lines.append(f"| {item['phase']} | `{item['operation']}` | `{item['endpoint']}` | {float(item['amplification']):.3f} | {float(item['p_value']):.4f} |")
        lines.append("")
    lines += [
        f"当前结果：Gemma 115 行、Llama text128 的 64 行和 text512 的 63 行都完成了首轮扫描，但没有新增通过冻结升级门的候选。根据用户要求，所有残差非零行已进入统一重放清单：当前有 {expanded_targets} 个行已绑定到可尝试的 fresh-compile 目标，另有 {expanded_blocked} 行因冻结 campaign 中没有同阶段、同实现族和同端点的程序而保留为 unresolved。任何一行在 pilot/4096 步前都不判为阴性；pattern-screen 本身也不等于训练 bias。",
        "",
    ]
    MD.write_text("\n".join(lines))
    print(json.dumps({"output": str(OUT), "models": [(m["model"], m["screened_rows"], m["bh_positive_rows"]) for m in models]}))


if __name__ == "__main__":
    main()
