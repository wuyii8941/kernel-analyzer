#!/usr/bin/env python3
"""blind_test_v1 phase 1 report under protocol revision v1.1.

Inputs: the aggregated phase 1 run (seeds 0-95) and the replication run on the disjoint seeds 96-191 (same
frozen rules).  Output: the 37 x 5 verdict matrix with mu intervals for both seed sets, the vector-mean
rules (R1, R5) and the alignment rules (R2, R3, R4) in separate tables, the default detector as a second
metric, the negative-control criteria for bitwise-identical pairs, and graded localization.  No totals
across rules are formed.

    python scripts/blind_test_v1_report_v11.py --phase1 results/reference_eval/blind_test_v1/phase1_report.json \
        --replication results/reference_eval/blind_test_v1/replication_96_191/phase1_report.json \
        --out results/reference_eval/blind_test_v1/phase1_report_v1.1.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SIGN = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}


def cell(o, rule):
    x = o["rules"].get(rule) if isinstance(o.get("rules"), dict) else None
    if not isinstance(x, dict) or "mu_interval" not in x:
        return None
    return {"verdict": x.get("final_verdict", x["verdict"]), "interval": x["mu_interval"]}


def fmt(c):
    if c is None:
        return "—"
    return f"{SIGN.get(c['verdict'], '?')} [{c['interval'][0]:.2e}, {c['interval'][1]:.2e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p1 = json.loads(args.phase1.read_text())
    rep = json.loads(args.replication.read_text())
    order = sorted(p1, key=lambda p: (p1[p]["family"], p))
    matrix = {}
    lines_vec, lines_ali = [], []
    for pid in order:
        for o1, o2 in zip(p1[pid]["outputs"], rep[pid]["outputs"]):
            name = pid if o1["output"] == "output" else f"{pid}（{o1['output']}）"
            row = {}
            for rule in ("R1", "R2", "R3", "R4", "R5"):
                a, b = cell(o1, rule), cell(o2, rule)
                if a is None:
                    continue
                reproduced = (a["verdict"].startswith("DETECTED") and b is not None and a["verdict"] == b["verdict"])
                row[rule] = {"seeds_0_95": a, "seeds_96_191": b, "detected_and_reproduced": reproduced}
            matrix[name] = {"family": p1[pid]["family"], "cells": row}

            def show(rule):
                if rule not in row:
                    return "—", "—"
                c = row[rule]
                mark = " ✔" if c["detected_and_reproduced"] else ""
                return fmt(c["seeds_0_95"]), fmt(c["seeds_96_191"]) + mark
            r1, r4, r5 = show("R1"), show("R4"), show("R5")
            lines_vec.append(f"| {name} | {p1[pid]['family']} | {r1[0]} | {r1[1]} | {r4[0]} | {r4[1]} | {r5[0]} | {r5[1]} |")
            r2, r3 = show("R2"), show("R3")
            lines_ali.append(f"| {name} | {p1[pid]['family']} | {r2[0]} | {r2[1]} | {r3[0]} | {r3[1]} |")
    # second metric: the default detector
    det_lines = []
    for pid in order:
        for o1, o2 in zip(p1[pid]["outputs"], rep[pid]["outputs"]):
            name = pid if o1["output"] == "output" else f"{pid}（{o1['output']}）"
            v = [(o.get("default_detector", {}).get(f, {}).get("final_verdict", "")) for o in (o1, o2)
                 for f in ("vector_mean", "alignment")]
            short = {"DETECTED": "检出", "NOT_CONFIRMED": "未确认", "EXPLORATORY_ONLY": "探索"}
            det_lines.append(f"| {name} | " + " | ".join(short.get(x, x) for x in v) + " |")
    # negative-control criteria for bitwise-identical pairs: (a) identical outputs, (b) identical verdicts
    pairs = sorted({tuple(sorted([p, q])) for p, r in p1.items() for q in r.get("bitwise_identical_to", [])})
    pair_lines = []
    for a, b in pairs:
        same_rep = rep[a].get("output_hashes") == rep[b].get("output_hashes")
        agree = all(matrix[a]["cells"][r]["seeds_0_95"]["verdict"] == matrix[b]["cells"][r]["seeds_0_95"]["verdict"]
                    for r in matrix[a]["cells"]) if a in matrix and b in matrix else None
        agree_rep = all((matrix[a]["cells"][r]["seeds_96_191"] or {}).get("verdict") ==
                        (matrix[b]["cells"][r]["seeds_96_191"] or {}).get("verdict") for r in matrix[a]["cells"]) \
            if a in matrix and b in matrix else None
        pair_lines.append(f"| {a} = {b} | 是 | {'是' if same_rep else '否'} | {'是' if agree else '否'} | "
                          f"{'是' if agree_rep else '否'} |")
    # localization grades (replication aggregate saw seeds 0, 32, 96 and 128)
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
        failed = ", ".join(failed_list) or "—"
        loc_lines.append(f"| {pid} | {loc.get('rule')} | `{node}` | {loc.get('region_size_ops', '')} | {grade} "
                         f"| {loc.get('seeds_agreeing', '')}/{loc.get('seeds_emulated', '')}（共 {n_seeds} 个输入） | {failed} |")
    md = f"""# blind_test_v1 阶段 1 报告（按协议修订 v1.1）

检测器冻结于 `e43616c`（打开盲测包之前），评测前的通用覆盖补充为 `cf7631c`（协议第 7 节允许）；本修订之后没有改动
检测器、方向规则或阈值。答案文件未打开。环境：NVIDIA RTX A6000，Triton 3.6.0，torch 2.10.0。没有修改任何程序源码。

seed 0–95 是阶段 1（0–31 开发、32–95 确认）；seed 96–191 是 v1.1 第 2 节要求的不相交复现集（96–127 开发、128–191
确认），用同一套冻结规则对**全部 37 个程序**重跑（被测方不知道哪些是干净版本，所以不挑程序）。每个格子：Holm 后的判定
（+ / − 为检出及 μ 的符号，· 为未确认）与 μ 的端点保守区间；✔ 表示两个 seed 集上都检出且符号相同。两次运行各自对
全部 170 个「程序 × 规则」检验做 Holm。全部程序「任务语义未检验」（阶段 2 才有规格 f）。

**表格说明**：μ 区间是未做多重校正的单格区间（95% t 区间、端点保守），最终判定以 Holm 结果为准；区间不含零而判定为未确认
的格子，原因就在这里，数值不改。复现的定性（出题方第 5 节）：同号复现是可重复性证据，不是准确率或召回率——两轮共用同一
参照求值器与统计方法，共享的系统错误不会被复现发现。

## 固定方向均值规则（R1 全体同向、R4 两半反向（仅 F4）、R5 学习方向）

R1、R4 是事前固定的向量；R5 的方向在开发 seed 上学，第二轮在 96–127 上重新学习，所以 R5 一栏的 ✔ 是**程序级复现**
（同一检测程序在新输入上的表现），不表示原方向上的同一个 μ 已跨输入确认。

| 程序 | 家族 | R1，seed 0–95 | R1，seed 96–191 | R4，0–95 | R4，96–191 | R5，0–95 | R5，96–191（程序级复现） |
|---|---|---|---|---|---|---|---|
{chr(10).join(lines_vec)}

## 参照相关的对齐规则（R2 向零收缩、R3 按比例缩放）

| 程序 | 家族 | R2，0–95 | R2，96–191 | R3，0–95 | R3，96–191 |
|---|---|---|---|---|---|
{chr(10).join(lines_ali)}

prog_16 的 R2、R3 记为「初次检出、复现未确认」；它的 R4 作用两轮都检出，两者不矛盾。

## 第二指标：工具默认检测器（不需要方向规则）

| 程序 | 向量均值，0–95 | 对齐，0–95 | 向量均值，96–191 | 对齐，96–191 |
|---|---|---|---|---|
{chr(10).join(det_lines)}

## 逐位相同的配对（v1.1 第 2 节阴性对照的两项）

| 配对 | 输出逐位相同（0–95） | 输出逐位相同（96–191） | 五条规则判定一致（0–95） | 判定一致（96–191） |
|---|---|---|---|---|
{chr(10).join(pair_lines)}

## 定位（v1.1 第 6 节分级）

对两次运行中有检出的程序，在保存的 seed 0、32、96、128 上逐位模拟；模拟逐位复现的输入上，把每个舍入 / 近似节点替换为
精确值，按贡献在检出规则方向上的投影排序。同一节点在两个及以上输入上排第一为稳定定位。

| 程序 | 规则 | 节点 | 覆盖 90% 投影的节点数 | 分级 | 排第一的输入数 / 模拟复现的输入数 | 模拟未复现的输入 |
|---|---|---|---:|---|---|---|
{chr(10).join(loc_lines)}

## 独立核实材料（v1.1 第 5 节）

阶段 1 有检出的每个程序在 seed 0 与 32 上的输入张量、运行时标量、输出 K（.npy 与原始字节）、工具的 K_R 区间、残差区间与
五条规则下的投影（R5 方向另存）见 `.cache/blind_v1_verification/`（打包为 `blind_v1_verification.tar.gz`）。
"""
    args.out.write_text(md)
    (args.out.with_suffix(".json")).write_text(json.dumps(matrix, indent=2, default=float) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
