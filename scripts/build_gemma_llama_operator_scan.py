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
OUT = ROOT / "results/property/declared_persistent_4096/operator_scan_gemma_llama.json"
MD = ROOT / "docs/gemma_llama_operator_scan.md"


def main() -> None:
    models = []
    for model, path in SOURCES.items():
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
        ranked = sorted(rows, key=lambda row: (float(row.get("p_value", 1.0)), -float(row.get("amplification", 0.0))))
        models.append({
            "model": model,
            "source": str(path.relative_to(ROOT)),
            "screened_rows": len(rows),
            "bh_positive_rows": sum(bool(row.get("screen_positive_bh_q_0_10", False)) for row in rows),
            "unseen_operator_rows_not_escalated": sum(not row.get("screen_positive_bh_q_0_10", False) for row in rows),
            "top_unseen_rows": [
                {
                    "phase": row.get("phase"),
                    "operation": row.get("operation"),
                    "amplification": row.get("amplification"),
                    "p_value": row.get("p_value"),
                    "implementation_pattern_id": row.get("implementation_pattern_id"),
                }
                for row in ranked[:10]
            ],
            "claim_boundary": "No row passed the frozen multiple-comparison gate; nominal p-values and A>1 are retained as scan evidence, not promoted to bias cases.",
        })
    result = {
        "schema": "gemma-llama-unseen-operator-scan-v1",
        "selection_rule": "Use the existing outcome-blind pattern screens; inspect all rows and escalate only screen_positive_bh_q_0_10=true.",
        "models": models,
        "new_long_candidates": [],
        "claim_boundary": "This scan found no new candidate that met the frozen escalation gate. It does not prove safety for rows with nominal p-values or without a legal repair/parameter carrier.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Gemma / Llama 未重复算子扫描",
        "",
        "这一步重新检查了两个模型已有的全量 pattern-screen 结果，并优先查看未出现在历史主案例中的算子族。升级规则仍是预先冻结的多重比较门，不因为单个 nominal p 值或 A>1 就把行列为 bias case。",
        "",
        "| 模型 | 首轮扫描行数 | 通过升级门的行数 | 结论 |",
        "|---|---:|---:|---|",
    ]
    for row in models:
        lines.append(f"| {row['model']} | {row['screened_rows']} | {row['bh_positive_rows']} | 未产生新的长程候选 |")
    lines += ["", "## 最强但未升级的算子族", "", "下面只列出排序最靠前的行，作为后续可重放入口；它们不被称为阴性，也不被称为 bias。", ""]
    for row in models:
        lines += [f"### {row['model']}", "", "| 阶段 | 算子族 | 方向分数 | nominal p |", "|---|---|---:|---:|"]
        for item in row["top_unseen_rows"][:5]:
            lines.append(f"| {item['phase']} | `{item['operation']}` | {float(item['amplification']):.3f} | {float(item['p_value']):.4f} |")
        lines.append("")
    lines += [
        "当前结果：Gemma 115 行、Llama 64 行都完成了首轮扫描，但没有新增通过冻结升级门的候选。因此没有凭 nominal 信号强行增加长程任务；若后续要扩大分母，应为这些算子建立合法 repair、载体和 live replay，而不是把 pattern-screen 直接当作训练 bias 证据。",
        "",
    ]
    MD.write_text("\n".join(lines))
    print(json.dumps({"output": str(OUT), "models": [(m["model"], m["screened_rows"], m["bh_positive_rows"]) for m in models]}))


if __name__ == "__main__":
    main()
