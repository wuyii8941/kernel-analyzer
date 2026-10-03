#!/usr/bin/env python3
"""Phase 2 of blind_test_v1: e_sem = K_R - f with the published family specifications.

f is evaluated from the published formulas (blind_test_v1_spec.py), independently of every program's TTIR,
once per family and seed (cached).  For each program the same seeds as phase 1 are run (0-31
development, 32-95 confirmation), K_R is evaluated exactly as in phase 1, and e_sem enters as the interval
[L_R - U_f, U_R - L_f].  The same five rules, endpoint-conservative t tests and Holm over program x rule
apply.  Phase 1's e = K - K_R is not changed.

    python scripts/run_blind_test_v1_phase2.py --package .cache/blind/blind_test_v1 --stage f --families F1,F2
    python scripts/run_blind_test_v1_phase2.py --package ... --stage programs --programs prog_01,...
    python scripts/run_blind_test_v1_phase2.py --package ... --stage aggregate
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import _holm  # noqa: E402

OUT = ROOT / "results" / "reference_eval" / "blind_test_v1" / "phase2"
FCACHE = ROOT / ".cache" / "blind_v1_phase2_f"
SEEDS = range(0, 96)
MODES = ("given", "fp32")


def stage_f(package: Path, families):
    from blind_test_v1_spec import SPEC

    sys.path.insert(0, str(package / "programs"))
    inputs = importlib.import_module("inputs")
    FCACHE.mkdir(parents=True, exist_ok=True)
    for fam in families:
        t0 = time.time()
        for seed in SEEDS:
            target = FCACHE / f"{fam}_{seed:03d}.npz"
            if target.exists():
                continue
            inp = inputs.make_inputs(fam, seed)
            arrays = {}
            for mode in MODES:
                lo, hi = SPEC[fam](inp, scalars=mode)
                arrays[f"{mode}_lo"], arrays[f"{mode}_hi"] = lo, hi
            np.savez(target, **arrays)
        print(fam, "f cached", round(time.time() - t0, 1), flush=True)


def stage_programs(package: Path, wanted):
    import run_blind_test_v1 as p1

    manifest = json.loads((package / "manifest.json").read_text())
    work = ROOT / ".cache" / "blind_v1_work_phase2"
    for entry in manifest["programs"]:
        pid, family = entry["id"], entry["family"]
        if wanted and pid not in wanted:
            continue
        target = OUT / "programs" / f"{pid}.json"
        if target.exists():
            continue
        r = p1.run_program(package, pid, family, work)
        report = {"program": pid, "family": family, "seconds": round(r["seconds"], 1), "modes": {}}
        for mode in MODES:
            data = []
            for seed, s in zip(SEEDS, r["per_seed"]):
                f = np.load(FCACHE / f"{family}_{seed:03d}.npz")
                f_lo, f_hi = f[f"{mode}_lo"][s["index"]], f[f"{mode}_hi"][s["index"]]
                # K_R interval recovered from the stored residual interval: lo_R = K - e_hi, hi_R = K - e_lo
                r_lo = iv.add_bounds(s["k"], -s["hi"])[0]
                r_hi = iv.add_bounds(s["k"], -s["lo"])[1]
                e_lo = iv.add_bounds(r_lo, -f_hi)[0]
                e_hi = iv.add_bounds(r_hi, -f_lo)[1]
                # the K_R midpoint stays the direction reference and is what the alignment tests compare with
                data.append({**s, "lo": e_lo, "hi": e_hi, "k": s["kr"]})
            outputs = []
            if family == "F3":
                idx = data[0]["index"]
                for col, name in ((0, "mean"), (1, "rstd")):
                    outputs.append(p1.analyse_output(name, family, data, (idx % 2) == col, None))
            else:
                outputs.append(p1.analyse_output("output", family, data, np.ones(data[0]["lo"].size, dtype=bool),
                                                 r["output_shape"]))
            report["modes"][mode] = outputs
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(pid, family, report["seconds"], {m: [(o["output"], {k: v.get("verdict") for k, v in o["rules"].items()}
                                                    if "verdict" not in o["rules"] else o["rules"])
                                                   for o in outs] for m, outs in report["modes"].items()}, flush=True)


def stage_aggregate(package: Path):
    manifest = json.loads((package / "manifest.json").read_text())
    reports = {e["id"]: json.loads((OUT / "programs" / f"{e['id']}.json").read_text()) for e in manifest["programs"]}
    for mode in MODES:
        tests = [(pid, o, rule) for pid, r in reports.items() for o in r["modes"][mode]
                 if "verdict" not in o["rules"] for rule in ("R1", "R2", "R3", "R4", "R5")
                 if rule in o["rules"] and "p" in o["rules"][rule]]
        for (pid, o, rule), rej in zip(tests, _holm([o["rules"][rule]["p"] for _, o, rule in tests], 0.05)):
            o["rules"][rule]["holm_reject"] = bool(rej)
            o["rules"][rule]["final_verdict"] = o["rules"][rule]["verdict"] if rej else "NOT_CONFIRMED"
    # second metric: the default detector on e_sem, Holm over all its tests of all programs, per family
    for mode in MODES:
        for fam in ("vector_mean", "alignment"):
            dt = [(o, t) for r in reports.values() for o in r["modes"][mode] if "default_detector" in o
                  for t in o["default_detector"][fam]["tests"]]
            for (o, t), rej in zip(dt, _holm([t["p"] for _, t in dt], 0.05)):
                supported = t.get("diagnostics", {}).get("tail_assumption_supported", True)
                t["final_verdict"] = ("DETECTED" if supported else "EXPLORATORY_ONLY") if rej else "NOT_CONFIRMED"
            for r in reports.values():
                for o in r["modes"][mode]:
                    if "default_detector" in o:
                        vs = [t["final_verdict"] for t in o["default_detector"][fam]["tests"]]
                        o["default_detector"][fam]["final_verdict"] = (
                            "DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY" if "EXPLORATORY_ONLY" in vs
                            else "NOT_CONFIRMED")
    (OUT / "phase2_report.json").write_text(json.dumps(reports, indent=2, default=float) + "\n")
    sign = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}

    def table(mode, rules):
        rows = []
        for pid in sorted(reports, key=lambda p: (reports[p]["family"], p)):
            for o in reports[pid]["modes"][mode]:
                name = pid if o["output"] == "output" else f"{pid}（{o['output']}）"
                cells = []
                for rule in rules:
                    x = o["rules"].get(rule)
                    if not isinstance(x, dict) or "mu_interval" not in x:
                        cells.append("—")
                    else:
                        cells.append(f"{sign.get(x['final_verdict'], '?')} [{x['mu_interval'][0]:.2e}, {x['mu_interval'][1]:.2e}]")
                res = o["residual"]
                signs = f"{res['positive_frac']:.0%} / {res['negative_frac']:.0%} / {res['contains_zero_frac']:.0%}"
                rows.append(f"| {name} | {reports[pid]['family']} | {res['mean']:.2e} | {res['max_width']:.1e} | "
                            f"{signs} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    def rows_where(pred):
        return [pid if o["output"] == "output" else f"{pid}（{o['output']}）"
                for pid in sorted(reports, key=lambda p: (reports[p]["family"], p))
                for o, o32 in zip(reports[pid]["modes"]["given"], reports[pid]["modes"]["fp32"]) if pred(o, o32)]

    def hit(o):
        return isinstance(o["rules"], dict) and any(str(v.get("final_verdict", "")).startswith("DETECTED")
                                                    for v in o["rules"].values() if isinstance(v, dict))

    def detector_hit(o):
        return any(o.get("default_detector", {}).get(f, {}).get("final_verdict") == "DETECTED"
                   for f in ("vector_mean", "alignment"))

    both = rows_where(lambda o, o32: hit(o) and hit(o32))
    given_only = rows_where(lambda o, o32: hit(o) and not hit(o32))
    differ_no_mean = rows_where(lambda o, o32: o["residual"]["contains_zero_frac"] <= 0.01 and not hit(o))
    width_level = rows_where(lambda o, o32: o["residual"]["contains_zero_frac"] == 1.0 and detector_hit(o))
    check = json.loads((OUT / "check_f3_merge.json").read_text())

    def compact(mode):
        rows = []
        for pid in sorted(reports, key=lambda p: (reports[p]["family"], p)):
            for o in reports[pid]["modes"][mode]:
                name = pid if o["output"] == "output" else f"{pid}（{o['output']}）"
                v = "".join(f"{r}{sign.get(o['rules'][r]['final_verdict'], '?')} " for r in ("R1", "R4", "R5", "R2", "R3")
                            if isinstance(o["rules"].get(r), dict) and "final_verdict" in o["rules"][r])
                rows.append(f"| {name} | {v.strip()} |")
        return "\n".join(rows)

    def detector(mode):
        rows = []
        short = {"DETECTED": "检出", "NOT_CONFIRMED": "未确认", "EXPLORATORY_ONLY": "探索"}
        for pid in sorted(reports, key=lambda p: (reports[p]["family"], p)):
            for o in reports[pid]["modes"][mode]:
                if "default_detector" not in o:
                    continue
                name = pid if o["output"] == "output" else f"{pid}（{o['output']}）"
                dd = o["default_detector"]
                hits = [t["test"] for f in ("vector_mean", "alignment") for t in dd[f]["tests"]
                        if t.get("final_verdict") == "DETECTED"]
                cz = o["residual"]["contains_zero_frac"]
                rows.append(f"| {name} | {cz:.0%} | {short.get(dd['vector_mean']['final_verdict'])} | "
                            f"{short.get(dd['alignment']['final_verdict'])} | {', '.join(hits) or '—'} |")
        return "\n".join(rows)

    md = f"""# blind_test_v1 阶段 2 报告：e_sem = K_R − f

规格 f 按出题方阶段 2 发布文件第 1 节逐字实现（`scripts/blind_test_v1_spec.py`），独立于每个程序的 TTIR：加减乘除与求和用
区间模块的定向界（求和是端点数组的 fsum 再向外一个 ulp），(·)^(−1/2) 为定向开方加定向倒数，F7 的 exp、log 为 MPFR 定向
舍入，F8 的 sign(0) = +1。f 只依赖输入，每个家族每个 seed 求一次，同家族所有程序共用。K_R 与阶段 1 相同（同一冻结求值器），
e_sem 以区间 [L_R − U_f, U_R − L_f] 进入；规则、seed（0–31 开发、32–95 确认）、端点保守检验与 Holm（全部程序 × 规则）都与
阶段 1 相同；F3 两列分开。检测器、方向规则、阈值未改。阶段 1 的 K − K_R 不变。

**运行时标量**：发布的 f 按 make_inputs 给出的值（Python 浮点数）取运行时标量（下文主表）。kernel 收到的是 float32 参数，
其中 F1 的 eps = 1e-6、F3 的 eps = 1e-5、F4 的 t = 1/√2 不能被 float32 精确表示，这部分差异属于接口精度，不是实现变体；
附表给出把这些标量先舍入到 float32 再求 f 的结果，便于区分。

表格说明：μ 区间是未做多重校正的单格区间（95% t 区间、端点保守），最终判定以 Holm 为准（+ / − 为检出及符号，· 为未确认）。
R1、R4、R5 为固定方向均值规则（R5 在开发 seed 上学习），R2、R3 为参照相关的对齐规则。「区间 >0 / <0 / 含零」是全部
seed、全部坐标中 e_sem 区间整体在零以上、整体在零以下、含零的比例：含零 100% 表示 K_R 与 f 在区间宽度内一致。

## 主表：f 取发布的运行时标量

| 程序 | 家族 | e_sem 均值 | 最大宽度 | 区间 >0 / <0 / 含零 | R1 | R2 | R3 | R4 | R5 |
|---|---|---:|---:|---|---|---|---|---|---|
{table("given", ("R1", "R2", "R3", "R4", "R5"))}

## 第二指标：工具默认检测器对 e_sem（f 取发布的运行时标量）

方向无关的总检验（omnibus）、全体均值、学习方向、行/列分组（向量均值）与沿参照方向、相对作用（对齐）；在全部程序的
检验上各自做 Holm。它不依赖 R1–R5，与上表分开报告。**读这张表要配合「含零」一栏**，见下文解读第 5 条。

| 程序 | e_sem 区间含零 | 向量均值 | 对齐 | 检出的检验 |
|---|---:|---|---|---|
{detector("given")}

## 解读（在判定固定之后写，不改变任何判定）

1. **两种标量取法下都检出**（与接口精度无关）：{"、".join(both)}。

2. **只在发布标量下检出**：{"、".join(given_only)}。F4 的 t = 1/√2 进入 kernel 时舍入为 float32（相对差约 −1.7e-8）；
   K_R 按 kernel 收到的 t 计算，f 按发布值计算，于是旋转部分 e_sem ≈ (t₃₂ − t)/t · y，在 R2、R3 方向上的投影为正。把 t 先舍入
   到 float32 之后这些检出全部消失（附表）。这是接口精度差异，不应记为实现变体。prog_35 在两种取法下都是 R2−、R3−，与此无关。

3. **prog_05、prog_08 的 rstd（约 1e-11）**：约 98.6% 坐标的 e_sem 区间不含零（宽度 1.3e-13），在 float32 标量下仍检出。为排除求值器错误，
   判定固定之后人工读了 prog_05 的源码：它把一行分 8 块合并均值与二阶矩，合并系数 CH/(k+CH) 与 k·CH/(k+CH) 是编译期的
   Python 浮点数，在 TTIR 里成为 float32 常数（1/3、1/5、1/6、1/7 与 341.33…、409.6、426.67…、438.86… 不能精确表示）。按这个
   合并公式、系数取 float32、其余按精确语义计算（`scripts/blind_test_v1_phase2_check_f3.py`），预测的 e_sem 均值为 mean 列
   {check["mean"]["predicted_mean"]:.3e}、rstd 列 {check["rstd"]["predicted_mean"]:.3e}，rstd 中预测为负的比例
   {check["rstd"]["predicted_negative_frac"]:.3f}，与主表一致。所以 K_R 是对的：这是程序声明语义与 f 之间真实但很小的差异
   （相对约 1e-12），是否算偏差由出题方判断。prog_08 的 e_sem 统计与 prog_05 完全相同（没有读 prog_08 的源码）。

4. **K_R ≠ f，但没有可确认的均值**：{"、".join(differ_no_mean)}。这些行至少 99% 的坐标 e_sem 区间不含零，但 R1–R5 与默认
   检测器（含方向无关的总检验）都没有确认均值：差异存在，在这些方向上、这个样本量下没有可确认的平均方向。差异最大的是
   prog_20（e_sem 均值 1.5e-2，正负各半）。

5. **默认检测器的宽度级检出**：{"、".join(width_level)}。这些行 100% 坐标的 e_sem 区间含零（K_R 与 f 在区间宽度内一致），
   默认检测器仍判检出。原因是检测器冻结时只取区间中点，没有端点保守的形式；对 e_sem，中点之差只是两套区间计算的舍入残余
   （不超过区间宽度：F1 不超过 1.4e-12，prog_25 rstd 为 2.0e-12），不能作为 K_R ≠ f 的证据。主表的端点保守规则在这些行上全部未
   确认。检测器没有改动（仍为冻结版本），这里只是报告写法上的说明。阶段 1 不受此影响：两轮（seed 0–95、96–191）中默认检测器检出的
   各行，残差区间含零的比例都低于 1e-5（prog_16 为 50%，未旋转的一半维度残差精确为零，另一半不含零）。

## 第 6 节子集

阶段 1 在 R2 或 R3 下检出的 14 个程序，seed 0 与 1 各一个 .npz（输入张量、运行时标量的 float64 值、输出 K），索引在
`index.json`。优先部分（F1、F3、F5：prog_05、07、11、12、21、26、27、31）与其余部分（prog_01、14、15、16、19、32）分两个
压缩包交付。

## 附表：运行时标量先舍入到 float32（判定摘要）

| 程序 | 判定（R1 R4 R5 / R2 R3） |
|---|---|
{compact("fp32")}
"""
    (OUT / "phase2_report.md").write_text(md)
    print("written", OUT / "phase2_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--stage", choices=("f", "programs", "aggregate"), required=True)
    parser.add_argument("--families", default="F1,F2,F2S,F3,F4,F5,F6,F7,F8")
    parser.add_argument("--programs", default=None)
    args = parser.parse_args()
    if args.stage == "f":
        stage_f(args.package, args.families.split(","))
    elif args.stage == "programs":
        stage_programs(args.package, set(args.programs.split(",")) if args.programs else None)
    else:
        stage_aggregate(args.package)


if __name__ == "__main__":
    main()
