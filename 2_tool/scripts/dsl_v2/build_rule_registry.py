#!/usr/bin/env python3
"""Write the DSL v2 rule registry (results/dsl_v2/rule_registry.json) and its report (docs/dsl_v2/rule_registry_report.md)
from kernel_analyzer.reference_eval.rule_registry.build().  The general-v3.1 registry under results/general/ is frozen
and is not rewritten."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval.rule_registry import REGISTRY_OUT, build  # noqa: E402

reg = build()
out = ROOT / REGISTRY_OUT
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
by_cat = Counter((e["category"], e["status"]) for e in reg["entries"])
lines = ["# 规则注册表报告（DSL v2 分支，Triton 3.6.0 回归 profile）", "",
         "由 `scripts/dsl_v2/build_rule_registry.py` 生成；条目键为（操作、属性、类型、子区域）。general-v3.1 的注册表"
         "（results/general/rule_registry.json）冻结不改。这只是旧 profile 的规则账，不是 rc3 官方目标面的覆盖率。", "",
         f"状态计数：{reg['counts']}", "", "| 类别 | 支持 | 声明前提 | 未建立 | 拒绝 |", "|---|---|---|---|---|"]
for cat in sorted({c for c, _ in by_cat}):
    lines.append(f"| {cat} | {by_cat[(cat, 'SUPPORTED')]} | {by_cat[(cat, 'DECLARED_PREMISE')]} | "
                 f"{by_cat[(cat, 'NOT_ESTABLISHED')]} | {by_cat[(cat, 'REJECTED')]} |")
lines += ["", "## 声明前提（交审阅方）", ""]
lines += [f"- `{e['key']['op']}` / {e['key']['subregion']}：{e['containment_argument']}" for e in reg["entries"]
          if e["status"] == "DECLARED_PREMISE"]
(ROOT / "docs/rule_registry_report.md").write_text("\n".join(lines) + "\n")
print(reg["counts"])
