#!/usr/bin/env python3
"""docs/rules_to_code_20261008.md from results/closure/chain_table.json (arrows -> code -> tests -> degrade records)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/closure/chain_table.json").read_text())
run = (ROOT / "results/closure/chain_table_tests_run.txt").read_text().strip().splitlines()[-1]
val = (ROOT / "results/closure/chain_table_validation.txt").read_text().strip().splitlines()[-1]
L = ["# 保证链：规则 → 函数 → 测试 → 降级记录（工作项 A，2026-10-08）", "",
     f"由 `results/closure/chain_table.json` 生成（`scripts/closure/build_chain_table.py`，本文件由 `scripts/closure/render_rules_doc.py` 渲染）；"
     f"包内校验器 `specs/phase2/validate_chain_table.py`：**{val}**（`results/closure/chain_table_validation.txt`）；表中引用的测试（含参数化展开）："
     f"`{run}`（`results/closure/chain_table_tests_run.txt`）。检测器 2.3（`docs/detector_changelog.md`）。逐节点求值规则的细目沿用 "
     "`docs/rules_to_code_20261006.md` 的 13 行，本表按保证链的五个箭头组织。", ""]
for a in d["arrows"]:
    L += [f"## 箭头 ({a['id']}) {a['name']}", "", "前提：" + "；".join(a["preconditions"]), "", "规则：" + "；".join(a["rules"]), "",
          "| 代码（函数 / 标识） | 文件 |", "|---|---|"]
    L += [f"| `{c['function']}` | `{c['file']}` |" for c in a["code"]]
    L += ["", "| 测试 | 文件 |", "|---|---|"] + [f"| `{t['name']}` | `{t['file']}` |" for t in a["tests"]]
    L += ["", "| 降级条件 | 处理 | 实际触发的记录 |", "|---|---|---|"]
    L += [f"| {p['condition']} | {p['action']} | `{p['triggered_record']}` |" for p in a["degrade_paths"]]
    L.append("")
(ROOT / "docs/rules_to_code_20261008.md").write_text("\n".join(L) + "\n")
print("rendered")
