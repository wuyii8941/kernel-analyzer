#!/usr/bin/env python3
"""Write results/general/rule_registry.json and docs/general/rule_registry_report.md from
kernel_analyzer.reference_eval.rule_registry.build()."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval.rule_registry import build  # noqa: E402

reg = build()
(ROOT / "results/general/rule_registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
by_cat = Counter((e["category"], e["status"]) for e in reg["entries"])
lines = ["# 规则注册表报告（通用能力轮，Triton 3.6.0 锁定构建）", "",
         "由 `scripts/general/build_rule_registry.py` 生成；条目键为（操作、属性、类型、子区域）。枚举测试 "
         "`tests/test_rule_registry.py`：锁定构建的每个操作都有条目、引用的测试都存在、入库 JSON 与代码一致、通用路径中没有按 "
         "kernel 名字做语义分支。", "", f"状态计数：{reg['counts']}", "",
         "| 类别 | 支持 | 声明前提 | 未建立 | 拒绝 |", "|---|---|---|---|---|"]
for cat in sorted({c for c, _ in by_cat}):
    lines.append(f"| {cat} | {by_cat[(cat, 'SUPPORTED')]} | {by_cat[(cat, 'DECLARED_PREMISE')]} | "
                 f"{by_cat[(cat, 'NOT_ESTABLISHED')]} | {by_cat[(cat, 'REJECTED')]} |")
lines += ["", "## 声明前提（交审阅方）", ""]
lines += [f"- `{e['key']['op']}` / {e['key']['subregion']}：{e['containment_argument']}" for e in reg["entries"]
          if e["status"] == "DECLARED_PREMISE"]
lines += ["", "## 拒绝（语义缺失）", ""]
lines += [f"- `{e['key']['op']}` {e['key']['attrs'] if e['key']['attrs'] != '*' else ''} {e['key']['subregion'] if e['key']['subregion'] != '*' else ''}："
          f"{e['reason']}" for e in reg["entries"] if e["status"] == "REJECTED" and not e["key"]["op"].startswith("gpu.")]
lines.append(f"- gpu 方言 {sum(1 for e in reg['entries'] if e['key']['op'].startswith('gpu.'))} 个操作：TTIR 阶段不出现，拒绝")
(ROOT / "docs/general").mkdir(parents=True, exist_ok=True)
(ROOT / "docs/general/rule_registry_report.md").write_text("\n".join(lines) + "\n")
print(reg["counts"])
