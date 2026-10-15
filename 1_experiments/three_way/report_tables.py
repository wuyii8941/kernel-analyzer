"""Markdown tables for the three_way README from summary.json (summarize.py) and rest.json (summarize_rest.py).

    python report_tables.py summary.json rest.json > tables.md
"""
from __future__ import annotations

import json
import sys

PREMISE_NAMES = {"functions": "近似指令按 IEEE", "fusion": "不做乘加融合", "order": "从左到右求和",
                 "tensor_core": "张量核就近舍入", "subnormals": "次正规数不冲零"}
METHOD_NAMES = {"A0": "A0 概率模型（恒为 0）", "A-exec": "A 教科书前提的精确执行", "C-model": "C 事前预测器",
                "C-exec": "C 真实前提的精确执行"}


def pct(k, n):
    return f"{k}/{n}（{100 * k / n:.0f}%）" if n else "—"


def main():
    s = json.load(open(sys.argv[1]))
    r = json.load(open(sys.argv[2]))
    out = []
    p = s["primary"]
    out += ["### 主结果（逐位相同的单元）", "",
            "| 方法 | 真值有显著偏差：符号对 | 符号与大小都对 | 真值无显著偏差：误报 |", "|---|---|---|---|"]
    for m in ("A0", "A-exec", "C-model", "C-exec"):
        v = p[m]
        out.append(f"| {METHOD_NAMES[m]} | {pct(v['sign_agree'], v['sig_n'])} | {pct(v['size_agree'], v['sig_n'])} | "
                   f"{pct(v['null_false_signal'], v['null_n'])} |")
    out += ["", "### 只换一个前提", "", "| 换掉的前提 | 真值有显著偏差：符号对 | 真值无显著偏差：误报 |", "|---|---|---|"]
    for q, name in PREMISE_NAMES.items():
        v = p[f"A-{q}"]
        out.append(f"| {name} | {pct(v['sign_agree'], v['sig_n'])} | {pct(v['null_false_signal'], v['null_n'])} |")
    h3 = s["H3"]
    out += ["", "### A-exec 的错误归到哪个前提", "",
            f"A-exec 的符号或大小错误共 {h3['errors']} 处，其中 {h3['explained_by_one_premise']} 处能由只换一个前提复现"
            "（一处错误可能同时由几个前提各自复现）：", "", "| 前提 | 复现的错误数 |", "|---|---|"]
    for q, k in sorted(h3["by_premise"].items(), key=lambda x: -x[1]):
        out.append(f"| {PREMISE_NAMES[q]} | {k} |")
    out += ["", "### 按配置与输出类型", "",
            "| 配置 | 输出类型 | 输出数 | 有显著偏差 | A-exec 符号对 | C-model 符号对 | A-exec 误报 | C-model 误报 |",
            "|---|---|---|---|---|---|---|---|"]
    for k, v in s["by_config_elem"].items():
        cfg, elem = eval(k)
        out.append(f"| {cfg} | {elem} | {v['outputs']} | {v['A-exec']['sig_n']} | "
                   f"{pct(v['A-exec']['sign_agree'], v['A-exec']['sig_n'])} | "
                   f"{pct(v['C-model']['sign_agree'], v['C-model']['sig_n'])} | "
                   f"{pct(v['A-exec']['null_false_signal'], v['A-exec']['null_n'])} | "
                   f"{pct(v['C-model']['null_false_signal'], v['C-model']['null_n'])} |")
    out += ["", "### 按 kernel 类型", "", "| 类型 | 输出数 | 有显著偏差 | A-exec 符号对 | C-model 符号对 |", "|---|---|---|---|---|"]
    names = {"pointwise": "逐元素", "reduction": "归约", "persistent": "持久归约", "template": "矩阵乘模板", "other": "其他"}
    for k, v in s["by_ktype"].items():
        out.append(f"| {names.get(k, k)} | {v['outputs']} | {v['A-exec']['sig_n']} | "
                   f"{pct(v['A-exec']['sign_agree'], v['A-exec']['sig_n'])} | "
                   f"{pct(v['C-model']['sign_agree'], v['C-model']['sig_n'])} |")
    sec = s["secondary"]
    out += ["", "### 补充分析：逐位核对未通过的单元（偏离 D1）", "",
            "| 方法 | 真值有显著偏差：符号对 | 真值无显著偏差：误报 |", "|---|---|---|"]
    for m in ("A0", "A-exec", "C-model"):
        v = sec[m]
        out.append(f"| {METHOD_NAMES[m]} | {pct(v['sign_agree'], v['sig_n'])} | {pct(v['null_false_signal'], v['null_n'])} |")
    # plants
    out += ["", "### 植入", "",
            "| 植入 | 类型 | 输出 | C：G′ − f | C：K − G → K′ − G′ | B-f：K′ − f | B-G：K′ − G′ |", "|---|---|---|---|---|---|---|"]
    for row in r["H4"]["plants"]:
        for n in row["semantic_alignment"]:
            sa = row["semantic_alignment"][n]
            v = "不一致" if n in row["flagged_outputs"] else "一致"
            b, a = row["numeric_before_after"][n]
            out.append(f"| {row['id']} | {'语义' if row['kind'] == 'semantic' else '阴性对照'} | {n} | {v}（{sa:+.1e}） | "
                       f"{b:+.1e} → {a:+.1e} | {row['B_f'][n]:+.1e} | {row['B_G'][n]:+.1e} |")
    out += ["", "| 数值植入 | 输出 | 植入行 | C 归因首位 | A 归因首位 |", "|---|---|---|---|---|"]
    for row in r["H5_plants"]["outputs"]:
        out.append(f"| {row['id']} | {row['output']} | {row['planted']} | {row['C_line']} | {row['A_line']} |")
    fx = r["H5_H6_fixes"]
    out += ["", "### 改法验证", "",
            "| 机制 | 改前 a | C 预测改后 | 实测改后 | A 认为的改前 | A 预测改后 | 偏差减半 | C 预测一致 |",
            "|---|---|---|---|---|---|---|---|"]
    for row in fx["fixes"]:
        out.append(f"| {row['mechanism']} | {row['before']:+.2e} | {row['C_predicted_after']:+.2e} | {row['after_measured']:+.2e} | "
                   f"{row['A_before']:+.2e} | {row['A_predicted_after']:+.2e} | {'是' if row['localized'] else '否'} | "
                   f"{'是' if row['C_after_agrees'] else '否'} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
