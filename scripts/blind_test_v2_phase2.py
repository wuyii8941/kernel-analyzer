#!/usr/bin/env python3
"""blind_test_v2 phase 2: e_sem = K_R - f with the published family specifications.

f is evaluated from the published formulas (blind_test_v2_spec.py), independently of every program's TTIR,
once per family and seed.  K_R is the phase 1 reference of the same captures (cached per seed by the phase 1
run), and e_sem enters as [L_R - U_f, U_R - L_f] (directed).  Rules R1, R2, R3, R5, the seed split (0-95 and
the replication 96-191), endpoint-conservative inference and Holm over program x rule are those of phase 1
(the unified decision layer, detector-v2.1).  Phase 1's e = K - K_R is not changed.

    python scripts/blind_test_v2_phase2.py --package .cache/blind/blind_test_v2 --stage f
    python scripts/blind_test_v2_phase2.py --package .cache/blind/blind_test_v2 --stage programs
    python scripts/blind_test_v2_phase2.py --package .cache/blind/blind_test_v2 --stage aggregate
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import holm_adjusted, interpret  # noqa: E402

OUT = ROOT / "results" / "reference_eval" / "blind_test_v2" / "phase2"
ARRAYS = ROOT / ".cache" / "blind_v2_arrays"
FCACHE = ROOT / ".cache" / "blind_v2_f"
RUNS = {"seeds_0_95": 0, "seeds_96_191": 96}
UNSCORED = {"G7", "G8"}
SIGN = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}
SHORT = {"DETECTED": "检出", "NOT_CONFIRMED": "未确认", "EXPLORATORY_ONLY": "探索", "CANNOT_JUDGE": "无法判断"}


def _f_task(args):
    package, fam, seed = args
    sys.path.insert(0, str(Path(package) / "programs"))
    inputs = importlib.import_module("inputs")
    from blind_test_v2_spec import SPEC

    target = FCACHE / f"{fam}_{seed:03d}.npz"
    if not target.exists():
        lo, hi = SPEC[fam](inputs.make_inputs(fam, seed, device="cpu"))
        np.savez(target, lo=lo, hi=hi)
    return fam, seed


def stage_f(package, workers):
    FCACHE.mkdir(parents=True, exist_ok=True)
    families = json.loads((package / "manifest.json").read_text())["families"]
    jobs = [(str(package), fam, s) for fam in families for s in range(192)]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_f_task, jobs, chunksize=4)):
            if (i + 1) % 256 == 0:
                print(f"f: {i + 1}/{len(jobs)}", flush=True)


def _shape(package, family):
    sys.path.insert(0, str(Path(package) / "programs"))
    s = importlib.import_module("inputs").SHAPES[family]
    return (s["R"],) if family in ("G5", "G6") else (s["R"], s["D"])


def _program_task(args):
    package, pid, family = args
    import run_blind_test_v1 as p1

    target = OUT / "programs" / f"{pid}.json"
    if target.exists():
        return pid
    shape = _shape(package, family)
    report = {"program": pid, "family": family, "output_shape": list(shape)}
    for run, offset in RUNS.items():
        p1.set_seed_offset(offset)
        data = []
        for seed in list(p1.DEV) + list(p1.CONF):
            c = np.load(ARRAYS / pid / f"seed{seed:03d}.npz")
            f = np.load(FCACHE / f"{family}_{seed:03d}.npz")
            idx = c["index"]
            e_lo, e_hi = iv.isub(c["kr_lo"], c["kr_hi"], f["lo"][idx], f["hi"][idx])
            kr = 0.5 * (c["kr_lo"] + c["kr_hi"])
            data.append({"lo": e_lo, "hi": e_hi, "kr": kr, "k": kr, "st": c["st"], "cond": c["cond"], "index": idx})
        out = p1.analyse_output("output", family, data, np.ones(data[0]["lo"].size, dtype=bool), shape, alignment="kr")
        out["record"]["measurement"] = {"quantity": "semantic", "definition": "K_R - f", "point": "kernel output"}
        report[run] = out
    target.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return pid


def stage_programs(package, workers):
    (OUT / "programs").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "manifest.json").read_text())
    jobs = [(str(package), e["id"], e["family"]) for e in manifest["programs"]]
    with Pool(workers) as pool:
        for pid in pool.imap_unordered(_program_task, jobs):
            print(pid, "done", flush=True)


def stage_aggregate(package):
    manifest = json.loads((package / "manifest.json").read_text())
    recs = {e["id"]: json.loads((OUT / "programs" / f"{e['id']}.json").read_text()) for e in manifest["programs"]}
    order = sorted(recs, key=lambda p: (recs[p]["family"], p))
    for run in RUNS:
        tests = [x for p in order for x in recs[p][run]["rules"].values() if isinstance(x, dict) and "p" in x]
        for x, adj in zip(tests, holm_adjusted([x["p"] for x in tests])):
            x["holm_adjusted_p"] = adj
            x["final_verdict"] = x["verdict"] if adj <= 0.05 else "NOT_CONFIRMED"
        for fam in ("vector_mean", "alignment"):
            dt = [t for p in order for t in recs[p][run]["default_detector"][fam]["tests"] if t.get("p") is not None]
            for t, adj in zip(dt, holm_adjusted([t["p"] for t in dt])):
                supported = t.get("diagnostics", {}).get("tail_assumption_supported", True)
                t["final_verdict"] = ("DETECTED" if supported else "EXPLORATORY_ONLY") if adj <= 0.05 else "NOT_CONFIRMED"
            for p in order:
                d = recs[p][run]["default_detector"][fam]
                vs = [t.get("final_verdict", t["verdict"]) for t in d["tests"]]
                d["final_verdict"] = ("DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY" if "EXPLORATORY_ONLY" in vs
                                      else "CANNOT_JUDGE" if vs and all(v == "CANNOT_JUDGE" for v in vs)
                                      else "NOT_CONFIRMED")
    (OUT / "phase2_report.json").write_text(json.dumps(recs, indent=2, default=float) + "\n")

    def cell(x):
        if not isinstance(x, dict) or "mu_interval" not in x:
            return "—"
        return f"{SIGN.get(x['final_verdict'], '?')} [{x['mu_interval'][0]:.2e}, {x['mu_interval'][1]:.2e}]"

    rows, det_rows, csv_rows, meanings = [], [], [], []
    for p in order:
        r = recs[p]
        fam = r["family"] + ("（不计分）" if r["family"] in UNSCORED else "")
        a, b = r["seeds_0_95"], r["seeds_96_191"]
        res = a["residual"]
        cells = []
        for rule in ("R1", "R2", "R3", "R5"):
            x, y = a["rules"].get(rule), b["rules"].get(rule)
            mark = " ✔" if (isinstance(x, dict) and isinstance(y, dict) and str(x.get("final_verdict")).startswith("DETECTED")
                            and x.get("final_verdict") == y.get("final_verdict")) else ""
            cells.append(f"{cell(x)} / {cell(y)}{mark}")
            if isinstance(x, dict) and str(x.get("final_verdict")).startswith("DETECTED"):
                meanings.append(f"- {p} {rule}{SIGN[x['final_verdict']]}：{interpret(rule, x['final_verdict'])}")
        rows.append(f"| {p} | {fam} | {res['positive_frac']:.0%} / {res['negative_frac']:.0%} / {res['contains_zero_frac']:.0%} | "
                    f"{res['mean']:.2e} | {res['max_width']:.1e} | " + " | ".join(cells) + " |")
        det_rows.append(f"| {p} | {res['contains_zero_frac']:.0%} | " + " | ".join(
            SHORT.get(o["default_detector"][f]["final_verdict"], "") for o in (a, b) for f in ("vector_mean", "alignment")) + " |")
        row = {"program": p, "family": r["family"], "scored": r["family"] not in UNSCORED,
               "e_sem_positive_frac": f"{res['positive_frac']:.4f}", "e_sem_negative_frac": f"{res['negative_frac']:.4f}",
               "e_sem_contains_zero_frac": f"{res['contains_zero_frac']:.4f}", "e_sem_mean": f"{res['mean']:.6e}",
               "e_sem_max_width": f"{res['max_width']:.3e}"}
        for rule in ("R1", "R2", "R3", "R5"):
            for run, label in (("seeds_0_95", "0_95"), ("seeds_96_191", "96_191")):
                x = r[run]["rules"].get(rule)
                ok = isinstance(x, dict) and "mu_interval" in x
                row[f"{rule}_verdict_seeds_{label}"] = x["final_verdict"] if ok else (x or {}).get("verdict", "not_applicable") if isinstance(x, dict) else "not_applicable"
                row[f"{rule}_mu_interval_seeds_{label}"] = f"[{x['mu_interval'][0]:.6e}, {x['mu_interval'][1]:.6e}]" if ok else ""
            x, y = a["rules"].get(rule), b["rules"].get(rule)
            row[f"{rule}_detected_and_reproduced"] = str(bool(isinstance(x, dict) and isinstance(y, dict)
                                                              and str(x.get("final_verdict")).startswith("DETECTED")
                                                              and x.get("final_verdict") == y.get("final_verdict"))).lower()
        for run, label in (("seeds_0_95", "0_95"), ("seeds_96_191", "96_191")):
            for f in ("vector_mean", "alignment"):
                row[f"default_detector_{f}_seeds_{label}"] = r[run]["default_detector"][f]["final_verdict"]
        csv_rows.append(row)
    with open(OUT / "phase2_verdict_matrix_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    det1 = sum(1 for r in csv_rows for k, v in r.items() if k.endswith("_seeds_0_95") and k.startswith("R") and v.startswith("DETECTED"))
    det2 = sum(1 for r in csv_rows for k, v in r.items() if k.endswith("_seeds_96_191") and k.startswith("R") and v.startswith("DETECTED"))
    repro = sum(1 for r in csv_rows for k, v in r.items() if k.endswith("_detected_and_reproduced") and v == "true")
    n_tests = sum(1 for p in order for x in recs[p]["seeds_0_95"]["rules"].values() if isinstance(x, dict) and "p" in x)
    md = f"""# blind_test_v2 阶段 2 报告：e_sem = K_R − f

规格 f 按出题方阶段 2 发布文件第 1 节实现（`scripts/blind_test_v2_spec.py`），独立于每个程序的 TTIR：加减乘除、max、绝对值、
按位置选取与整数位运算用参照求值器的区间运算（float32 / fp16 输入的乘积在 float64 中精确时宽度为 0，否则定向），求和是每个
端点数组的精确和（fsum，正确舍入）再向外一个 ulp，√ 与 (·)^(−1/2) 为定向开方（与定向倒数），G8 的前缀和用精确整数逐项累加
（宽度 0）。f 只依赖输入，每个家族每个 seed 求一次，同家族所有程序共用；用普通 float64 公式交叉核对，相对差在舍入水平。

K_R 是阶段 1 同一批捕获的参照（阶段 1 运行时逐 seed 缓存），e_sem 以区间 [L_R − U_f, U_R − L_f] 进入（有向减法）。规则 R1、R2、
R3、R5，seed 划分（0–95 一轮，96–191 复现），端点保守推断与 Holm（每轮对全部 {n_tests} 个「程序 × 规则」检验）都与阶段 1 相同，
判定层仍是阶段 1 冻结的 `detector-v2.1`（`408ab3a`）。阶段 1 的 K − K_R 不变。阶段 1 的 796 个检验没有一个触发 2.1 相对第 2 版
（`3490425`）新增的保护，所以两个版本在阶段 1 上的判定完全相同；本阶段同样如此（见文末）。所有运行时标量 float32 精确，
接口项为零，只有一种标量口径。

检出：seed 0–95 共 {det1} 格，96–191 共 {det2} 格，两轮都检出且同号 {repro} 格。

表格：「区间 >0 / <0 / 含零」是 seed 0–95 全部坐标中 e_sem 区间整体在零以上、以下、含零的比例（含零 100% 表示 K_R 与 f 在区间
宽度内一致）；每格「0–95 / 96–191」为 Holm 后的判定与 μ 的端点保守区间（单格、未校正，判定以 Holm 为准），✔ 为两轮同号检出。
μ 的符号不等于 e_sem 的符号（R1 的方向是 −1/√n，R2、R3 也带负号），各检出的含义见下文。

| 程序 | 家族 | 区间 >0 / <0 / 含零 | e_sem 均值 | 最大宽度 | R1 | R2 | R3 | R5 |
|---|---|---|---:|---:|---|---|---|---|
{chr(10).join(rows)}

## 检出及其含义（seed 0–95）

{chr(10).join(meanings) or '（无）'}

## 第二指标：默认检测器 2.1 对 e_sem

每轮对全部程序的检验按族做 Holm，无法判断的检验不参与。

| 程序 | e_sem 区间含零 | 向量均值，0–95 | 对齐，0–95 | 向量均值，96–191 | 对齐，96–191 |
|---|---:|---|---|---|---|
{chr(10).join(det_rows)}

## 保护条件

本阶段触发 2.1 新增保护（样本不足、数值失效）的检验：{guard_count(recs)} 个。
"""
    (OUT / "phase2_report.md").write_text(md)
    print("written", OUT / "phase2_report.md", "detected", det1, det2, "reproduced", repro)


def guard_count(recs):
    n = 0
    for r in recs.values():
        for run in RUNS:
            o = r[run]
            n += sum(1 for x in o["record"]["rules"] if x["verdict"].startswith("UNRESOLVED"))
            n += sum(1 for f in ("vector_mean", "alignment") for t in o["default_detector"][f]["tests"] if t.get("p") is None)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--stage", choices=("f", "programs", "aggregate"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    {"f": lambda: stage_f(args.package, args.workers), "programs": lambda: stage_programs(args.package, args.workers),
     "aggregate": lambda: stage_aggregate(args.package)}[args.stage]()


if __name__ == "__main__":
    main()
