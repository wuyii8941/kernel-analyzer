#!/usr/bin/env python3
"""blind_test_v2 phase 1 report (same protocol as blind_test_v1 v1.1).

Inputs: the aggregated runs on seeds 0-95 and 96-191 (run_blind_test_v1.py --aggregate on each).  Outputs:
phase1_report_v2.md / .json (both seed sets, fixed-direction rules R1, R5 and alignment rules R2, R3 in
separate tables, the default detector as a second metric, bitwise-identical pairs, graded localization,
interface inventory) and phase1_verdict_matrix_v2.csv.

    python scripts/blind_test_v2_report.py --package .cache/blind/blind_test_v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reference_eval" / "blind_test_v2"
RULES = ("R1", "R2", "R3", "R5")
SIGN = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}
SHORT = {"DETECTED": "检出", "NOT_CONFIRMED": "未确认", "EXPLORATORY_ONLY": "探索", "CANNOT_JUDGE": "无法判断"}


def cell(o, rule):
    x = o["rules"].get(rule) if isinstance(o.get("rules"), dict) else None
    if not isinstance(x, dict) or "mu_interval" not in x:
        return None if not isinstance(x, dict) else {"verdict": x.get("verdict"), "interval": None}
    return {"verdict": x.get("final_verdict", x["verdict"]), "interval": x["mu_interval"]}


def fmt(c):
    if c is None:
        return "—"
    if c["interval"] is None:
        return c["verdict"] or "—"
    return f"{SIGN.get(c['verdict'], '?')} [{c['interval'][0]:.2e}, {c['interval'][1]:.2e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--tarball", type=Path, default=ROOT / "blind_test_v2.tar.gz")
    args = parser.parse_args()
    p1 = json.loads((OUT / "phase1" / "phase1_report.json").read_text())
    rep = json.loads((OUT / "replication_96_191" / "phase1_report.json").read_text())
    manifest = json.loads((args.package / "manifest.json").read_text())
    unscored = {"G7", "G8"}
    order = sorted(p1, key=lambda p: (p1[p]["family"], p))
    matrix, lines_vec, lines_ali, det_lines, csv_rows = {}, [], [], [], []
    for pid in order:
        o1, o2 = p1[pid]["outputs"][0], rep[pid]["outputs"][0]
        fam = p1[pid]["family"] + ("（不计分）" if p1[pid]["family"] in unscored else "")
        row = {}
        for rule in RULES:
            a, b = cell(o1, rule), cell(o2, rule)
            if a is None:
                continue
            reproduced = bool(a["verdict"] and a["verdict"].startswith("DETECTED") and b is not None
                              and a["verdict"] == b["verdict"])
            row[rule] = {"seeds_0_95": a, "seeds_96_191": b, "detected_and_reproduced": reproduced}
        matrix[pid] = {"family": p1[pid]["family"], "cells": row}

        def show(rule):
            if rule not in row:
                return "—", "—"
            c = row[rule]
            return fmt(c["seeds_0_95"]), fmt(c["seeds_96_191"]) + (" ✔" if c["detected_and_reproduced"] else "")
        r1, r5 = show("R1"), show("R5")
        lines_vec.append(f"| {pid} | {fam} | {r1[0]} | {r1[1]} | {r5[0]} | {r5[1]} |")
        r2, r3 = show("R2"), show("R3")
        lines_ali.append(f"| {pid} | {fam} | {r2[0]} | {r2[1]} | {r3[0]} | {r3[1]} |")
        v = [o.get("default_detector", {}).get(f, {}).get("final_verdict", "") for o in (o1, o2)
             for f in ("vector_mean", "alignment")]
        det_lines.append(f"| {pid} | " + " | ".join(SHORT.get(x, x) for x in v) + " |")
        csv_row = {"program": pid, "family": p1[pid]["family"], "scored": p1[pid]["family"] not in unscored}
        for rule in RULES:
            c = row.get(rule)
            for key, label in (("seeds_0_95", "0_95"), ("seeds_96_191", "96_191")):
                x = c[key] if c else None
                csv_row[f"{rule}_verdict_seeds_{label}"] = (x or {}).get("verdict") or "not_applicable"
                iv_ = (x or {}).get("interval")
                csv_row[f"{rule}_mu_interval_seeds_{label}"] = f"[{iv_[0]:.6e}, {iv_[1]:.6e}]" if iv_ else ""
            csv_row[f"{rule}_detected_and_reproduced"] = str(bool(c and c["detected_and_reproduced"])).lower()
        csv_rows.append(csv_row)
    with open(OUT / "phase1_verdict_matrix_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    pairs = sorted({tuple(sorted([p, q])) for p, r in p1.items() for q in r.get("bitwise_identical_to", [])})
    pair_lines = []
    for a, b in pairs:
        same_rep = rep[a].get("output_hashes") == rep[b].get("output_hashes")
        agree = all((matrix[a]["cells"][r]["seeds_0_95"] or {}).get("verdict") ==
                    (matrix[b]["cells"][r]["seeds_0_95"] or {}).get("verdict") for r in matrix[a]["cells"])
        agree_rep = all((matrix[a]["cells"][r]["seeds_96_191"] or {}).get("verdict") ==
                        (matrix[b]["cells"][r]["seeds_96_191"] or {}).get("verdict") for r in matrix[a]["cells"])
        pair_lines.append(f"| {a} = {b} | 是 | {'是' if same_rep else '否'} | {'是' if agree else '否'} | "
                          f"{'是' if agree_rep else '否'} |")
    loc_lines = []
    for pid in order:
        loc = rep[pid].get("localization") or p1[pid].get("localization")
        if not loc:
            continue
        node = (loc.get("localized") or loc.get("error") or "").split("loc(")[0].strip()
        node = node[:80] + ("…" if len(node) > 80 else "")
        failed_list = loc.get("emulation_failed_on", [])
        n_seeds = len(loc.get("per_seed", {})) or 4
        grade = {"stable": "稳定定位", "clue": "定位线索", "inconsistent": "各输入不一致"}.get(loc.get("grade"), "—")
        if loc.get("grade") == "stable" and failed_list:
            grade = (f"条件性定位：{n_seeds} 个输入中 {loc.get('seeds_emulated')} 个可模拟，"
                     f"这 {loc.get('seeds_emulated')} 个中 {loc.get('seeds_agreeing')} 个首位节点一致")
        loc_lines.append(f"| {pid} | {loc.get('rule')} | `{node}` | {loc.get('region_size_ops', '')} | {grade} | "
                         f"{loc.get('seeds_agreeing', '')}/{loc.get('seeds_emulated', '')}（共 {n_seeds} 个输入） | "
                         f"{', '.join(failed_list) or '—'} |")
    inv_lines = []
    for pid in order:
        inv = p1[pid].get("interface") or {}
        rounded_s = inv.get("rounded_runtime_scalars", [])
        rounded_c = inv.get("rounded_compile_time_constants", [])
        n_const = len([c for c in inv.get("float_constants", []) if c.get("value") is not None])
        scalars = ", ".join(f"{s['name']}={s['passed']}（{s['parameter_type']}{'，精确' if s.get('exact') else ''}）"
                            for s in inv.get("runtime_scalars", []) if not s.get("compile_time")) or "—"
        consts = "；".join(f"{c['value']} ≈ {c['rounded_from']['candidate']}（{c['rounded_from']['relative_rounding']:.2e}）"
                          for c in rounded_c) or "无"
        inv_lines.append(f"| {pid} | {scalars} | {'、'.join(rounded_s) or '无'} | {n_const} | {consts} |")
    tested = [o for r in p1.values() for o in r["outputs"] if isinstance(o.get("rules"), dict)]
    n_tests = sum(1 for o in tested for k, x in o["rules"].items() if isinstance(x, dict) and "p" in x)
    det1 = sum(1 for m in matrix.values() for c in m["cells"].values()
               if (c["seeds_0_95"] or {}).get("verdict", "").startswith("DETECTED"))
    det2 = sum(1 for m in matrix.values() for c in m["cells"].values()
               if (c["seeds_96_191"] or {}).get("verdict", "").startswith("DETECTED"))
    repro = sum(1 for m in matrix.values() for c in m["cells"].values() if c["detected_and_reproduced"])
    cov = {pid: p1[pid]["outputs"][0]["coverage_classes"] for pid in order}
    all_complete = all(c["complete_composed"] == 1.0 for c in cov.values())
    sha = hashlib.sha256(args.tarball.read_bytes()).hexdigest() if args.tarball.exists() else "?"
    md = f"""# blind_test_v2 阶段 1 报告（同 v1 协议 v1.1）

盲测包 `blind_test_v2.tar.gz`（sha256 `{sha}`），38 个程序、8 个家族。解包时排除了两个封存答案文件
（`answer_key_SEALED.json`、`answer_key_update_layer_SEALED.json`），它们从未写到磁盘上。没有修改任何程序源码（38 个程序
全部原样编译）。环境：NVIDIA RTX A6000（sm_86），Triton 3.6.0，torch 2.10.0。

**被测版本**：默认检测器与判定层冻结于标签 `detector-v2.1`（提交 `408ab3a`），在打开本包之前。包内 README 写的是第 2 版
（`3490425`）；2.1 只多了上一轮回归审阅要求、在开测前完成的两项修改——所有入口用有向减法构造残差区间，样本不足或数值失效时
返回无法判断。打开本包之后没有改动检测器、方向规则或阈值；为本包所作的只是绑定层的通用改动（工作目录、按模板写 CSV、
缓存每个 seed 的 K 与 K_R 供阶段 2、3 复用）。

seed 0–95（0–31 开发、32–95 确认）与 96–191（96–127 开发、128–191 确认）是两次独立运行，各自对全部 {n_tests} 个
「程序 × 规则」检验做 Holm。每格：Holm 后的判定（+ / − 为检出及 μ 的符号，· 为未确认）与 μ 的端点保守区间；✔ 表示两组 seed
都检出且符号相同。μ 区间是未做多重校正的单格区间（95% t 区间，端点保守），最终判定以 Holm 为准。μ 的符号不等于原始残差的
符号：R1 的方向是 −1/√n，R2、R3 也带负号（R1 为 + 表示残差均值为负；R2 为 + 表示残差与 K_R 反号，即向零收缩；R3 为 + 表示
输出被缩小）。复现是可重复性证据，不是准确率。G7、G8 按协议不计分。全部程序「任务语义未检验」（阶段 2 才有规格 f）。

## 覆盖与参照

38 个程序覆盖完整，没有被拒绝的操作，没有中止的 program 实例；{'每个程序、每个 seed 的全部输出元素都是完整组合参照（参照未建立 0）' if all_complete else '有程序存在条件参照或参照未建立，见 JSON 中的 coverage_classes'}。
G8 用到的 tt.scan（tl.cumsum）在数值差异模式下按声明语义精确支持：参照是实数前缀和的严格包围（γₙ 先验界），不是近似，
也不是参照未建立。检出数：seed 0–95 共 {det1} 格，seed 96–191 共 {det2} 格，两组都检出且同号 {repro} 格。

## 固定方向均值规则（R1 全体同向、R5 学习方向）

R5 的方向在每组的开发 seed 上分别学习，R5 一栏的 ✔ 是程序级复现。

| 程序 | 家族 | R1，seed 0–95 | R1，seed 96–191 | R5，0–95 | R5，96–191（程序级复现） |
|---|---|---|---|---|---|
{chr(10).join(lines_vec)}

## 参照相关的对齐规则（R2 向零收缩、R3 按比例缩放）

| 程序 | 家族 | R2，0–95 | R2，96–191 | R3，0–95 | R3，96–191 |
|---|---|---|---|---|---|
{chr(10).join(lines_ali)}

## 第二指标：默认检测器第 2.1 版（不需要方向规则）

每组 seed 内对全部程序的检验按族做 Holm；无法判断的检验不参与。

| 程序 | 向量均值，0–95 | 对齐，0–95 | 向量均值，96–191 | 对齐，96–191 |
|---|---|---|---|---|
{chr(10).join(det_lines)}

## 逐位相同的配对

| 配对 | 输出逐位相同（0–95） | 输出逐位相同（96–191） | 四条规则判定一致（0–95） | 判定一致（96–191） |
|---|---|---|---|---|
{chr(10).join(pair_lines) or '| （无） | | | | |'}

## 定位

对有检出的程序，在保存的 seed 0、32、96、128 上逐位模拟；模拟逐位复现的输入上，把每个舍入 / 近似节点替换为精确值，按贡献在
检出规则方向上的投影排序。同一节点在两个及以上输入上排第一为稳定定位。

| 程序 | 规则 | 节点 | 覆盖 90% 投影的节点数 | 分级 | 排第一的输入数 / 模拟复现的输入数 | 模拟未复现的输入 |
|---|---|---|---:|---|---|---|
{chr(10).join(loc_lines) or '| （无） | | | | | | |'}

## 接口标量与编译期常数

| 程序 | 运行时标量（传入值，参数类型） | 启动时被舍入的标量 | 浮点常数个数 | 编译期舍入的常数（候选值只是从存储值读出的推测） |
|---|---|---|---:|---|
{chr(10).join(inv_lines)}

## 文件

`phase1/` 与 `replication_96_191/`：每个程序的完整记录（`programs/*.json`，含判定层记录：每条规则每个确认 seed 的投影上下界、
实际方向、原始 p）、汇总 JSON 与按模板填写的 CSV；`phase1_verdict_matrix_v2.csv`：两组 seed 的判定矩阵。
"""
    (OUT / "phase1_report_v2.md").write_text(md)
    (OUT / "phase1_report_v2.json").write_text(json.dumps(matrix, indent=2, default=float) + "\n")
    print("written", OUT / "phase1_report_v2.md", "detected", det1, det2, "reproduced", repro)


if __name__ == "__main__":
    main()
