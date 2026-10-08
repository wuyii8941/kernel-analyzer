#!/usr/bin/env python3
"""docs/general/calibration_table.md from results/general/calibration_five_distributions.json."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/general/calibration_five_distributions.json").read_text())
t = defaultdict(dict)
for r in d["rows"]:
    t[(r["distribution"], r["effect"])][r["rule_or_class"]] = (r["reject_rate"], r["cannot_judge_rate"])
names = {"dense_small": "稠密小值", "sparse_large": "稀疏大值", "group_cancellation": "组内正负抵消",
         "state_dependent": "状态相关", "rare_tail": "稀有大尾部"}
null = [v[k][0] for (dist, eff), v in t.items() if eff == 0.0 for k in ("fixed_mean", "aligned") if k in v]
L = ["# 第 4 项：默认检测在五类合成分布上的校准（统一入口的同一条统计路径）", "",
     f"设计：开发 {d['design']['development']} + 确认 {d['design']['confirmation']} 个单位，{d['design']['coordinates']} 个坐标，"
     f"残差区间宽 {d['design']['interval_width']}，每格 {d['design']['replicates']} 次重复（种子按分布名的 CRC32 固定，可逐位复现）；"
     "效应量 0 / 0.005 / 0.02 / 0.25 个坐标噪声标准差（协议第 4 节与偏离 1，运行前写定）。每格「拒绝率 / 无法判断率」；"
     "固定均值类（R1、R5，类内 Holm）与对齐类（R2、R3，类内 Holm）分开；稀有大尾部另报有界路线（Hoeffding，M = 100·Σ|w|；"
     "协议第 4 节写的是经验 Bernstein，实现用 Hoeffding，见协议偏离）。零效应行的拒绝率即误报率。", "",
     "| 分布 | 效应 | R1 | R5 | 固定均值类 | R2 | R3 | 对齐类 | 有界路线 R1 |", "|---|---|---|---|---|---|---|---|---|"]
for (dist, eff), v in sorted(t.items(), key=lambda x: (list(names).index(x[0][0]), x[0][1])):
    f = lambda k: ("%.3f / %.3f" % v[k]) if k in v else "—"  # noqa: E731
    L.append(f"| {names[dist]} | {eff} | {f('R1')} | {f('R5')} | {f('fixed_mean')} | {f('R2')} | {f('R3')} | {f('aligned')} | {f('bounded_R1')} |")
L += ["", f"零效应下两类拒绝率范围：{min(null):.3f}–{max(null):.3f}（1000 次重复的标准误约 0.007）。"
      "「未确认」不写成「没有 bias」；有界路线只用于这里的校准，统一入口不应用它。"]
(ROOT / "docs/general/calibration_table.md").write_text("\n".join(L) + "\n")
print("null range", min(null), max(null))
