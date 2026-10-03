#!/usr/bin/env python3
"""blind_test_v1 as a regression set for the fixed tool (the frozen results stay as submitted).

Fixes under test: default detector version 2 (interval inputs at every exit), strict outward projection
bounds, the corrected endpoint-conservative p-value, and the unified decision layer (analysis.assess_units),
which the blind scripts now only bind to.  Per program, one run over seeds 0-95 (0-31 development, 32-95
confirmation) and five measured quantities, each assessed with the same rules and the default detector:

    implementation        e = K - K_R
    semantics_interface   K_R - f_interface   (runtime scalars as the kernel receives them, float32)
    semantics_published   K_R - f_published   (runtime scalars as published)
    interface_term        f_interface - f_published   (rounding at the launch: an upstream node)
    total_published       K - f_published

so K - f_published = implementation + semantics_published, and semantics_published = semantics_interface +
interface_term.  Each program record also carries the interface inventory (runtime scalars passed /
received, rounded compile-time constants).  Records hold everything needed to recompute the decisions
offline (per-unit projection bounds, directions and normalization scalars, development / confirmation
seeds, raw and Holm-adjusted p-values, exclusions); direction vectors go to .cache/blind_v1_regression_v2.

    python scripts/blind_test_v1_regression.py --package .cache/blind/blind_test_v1 --programs prog_01,prog_02
    python scripts/blind_test_v1_regression.py --package .cache/blind/blind_test_v1 --aggregate
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import apply_holm, holm_adjusted, interpret  # noqa: E402

RES = ROOT / "results" / "reference_eval" / "blind_test_v1"
OUT = RES / "regression_v2"
ARRAYS = ROOT / ".cache" / "blind_v1_regression_v2"
FCACHE = ROOT / ".cache" / "blind_v1_phase2_f"
FIXTURE = ROOT / "tests" / "data" / "blind_v1_zero_semantic_residual_rows.npz"
QUANTITIES = {
    "implementation": ("K - K_R", "k"),
    "semantics_interface": ("K_R - f_interface (runtime scalars as received, float32)", "kr"),
    "semantics_published": ("K_R - f_published (runtime scalars as published)", "kr"),
    "interface_term": ("f_interface - f_published (rounding at the launch interface)", "kr"),
    "total_published": ("K - f_published", "kr"),
}
FIXTURE_COORDS = 128


def _sub(a_lo, a_hi, b_lo, b_hi):
    """Directed [a_lo - b_hi, a_hi - b_lo]."""

    return iv.add_bounds(a_lo, -b_hi)[0], iv.add_bounds(a_hi, -b_lo)[1]


def zero_semantic_rows():
    """Outputs whose frozen phase 2 e_sem intervals all contained zero while the frozen (version 1) default
    detector reported a detection: derived from the frozen report, not listed by hand."""

    report = json.loads((RES / "phase2" / "phase2_report.json").read_text())
    rows = set()
    for pid, rep in report.items():
        for o in rep["modes"]["given"]:
            det = o.get("default_detector", {})
            hit = any(det.get(f, {}).get("final_verdict") == "DETECTED" for f in ("vector_mean", "alignment"))
            if o["residual"]["contains_zero_frac"] == 1.0 and hit:
                rows.add((pid, o["output"]))
    return rows


def run(package: Path, pid: str, family: str):
    import run_blind_test_v1 as p1
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.interface import inventory
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    work = ROOT / ".cache" / "blind_v1_work_regression"
    r = p1.run_program(package, pid, family, work)
    launch = load_launch(work / pid / f"seed{p1.DEV[0]:03d}")
    record = {"program": pid, "family": family, "launch": r["launch"], "coverage_complete": r["coverage"]["complete"],
              "output_hashes": r["hashes"], "interface": inventory(launch, parse_ttir(launch.asm["ttir"])),
              "seconds": round(r["seconds"], 1), "outputs": {}}
    seeds = list(p1.DEV) + list(p1.CONF)
    quantities = {q: [] for q in QUANTITIES}
    for seed, s in zip(seeds, r["per_seed"]):
        f = np.load(FCACHE / f"{family}_{seed:03d}.npz")
        idx = s["index"]
        g_lo, g_hi = f["given_lo"][idx], f["given_hi"][idx]
        i_lo, i_hi = f["fp32_lo"][idx], f["fp32_hi"][idx]
        kr_lo = iv.add_bounds(s["k"], -s["hi"])[0]  # K_R recovered from the residual interval, as in phase 2
        kr_hi = iv.add_bounds(s["k"], -s["lo"])[1]
        base = {k: s[k] for k in ("kr", "k", "st", "cond", "index")}
        quantities["implementation"].append({**base, "lo": s["lo"], "hi": s["hi"]})
        for q, (lo, hi) in (("semantics_interface", _sub(kr_lo, kr_hi, i_lo, i_hi)),
                            ("semantics_published", _sub(kr_lo, kr_hi, g_lo, g_hi)),
                            ("interface_term", _sub(i_lo, i_hi, g_lo, g_hi)),
                            ("total_published", _sub(s["k"], s["k"], g_lo, g_hi))):
            quantities[q].append({**base, "lo": lo, "hi": hi})
    idx = r["per_seed"][0]["index"]
    outputs = {"output": np.ones(idx.size, dtype=bool)} if family != "F3" else \
        {"mean": (idx % 2) == 0, "rstd": (idx % 2) == 1}
    fixture_rows = zero_semantic_rows()
    arrays = {}
    for name, sel in outputs.items():
        shape = r["output_shape"] if family != "F3" else None
        record["outputs"][name] = {}
        for q, (desc, align) in QUANTITIES.items():
            rec, vecs = p1.assess_output(f"{pid}/{name}/{q}", family, quantities[q], sel, shape, alignment=align,
                                         measurement={"quantity": q, "definition": desc, "point": "kernel output"})
            record["outputs"][name][q] = rec
            arrays.update(vecs)
        if (pid, name) in fixture_rows:  # one piece per output; the aggregate step merges them
            data = quantities["semantics_published"]
            cols = np.flatnonzero(sel)[:FIXTURE_COORDS]
            key = f"{pid}__{name}"
            ARRAYS.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ARRAYS / f"fixture__{key}.npz", **{
                f"{key}__lo": np.stack([d["lo"][cols] for d in data]),
                f"{key}__hi": np.stack([d["hi"][cols] for d in data]),
                f"{key}__kr": np.stack([d["kr"][cols] for d in data])})
    ARRAYS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS / f"{pid}.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
    return record


def per_program(args):
    manifest = json.loads((args.package / "manifest.json").read_text())
    wanted = set(args.programs.split(",")) if args.programs else None
    sys.path.insert(0, str(args.package / "programs"))
    for entry in manifest["programs"]:
        pid, family = entry["id"], entry["family"]
        if wanted and pid not in wanted:
            continue
        target = OUT / "programs" / f"{pid}.json"
        if target.exists():
            continue
        record = run(args.package, pid, family)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, default=float) + "\n")
        print(pid, family, record["seconds"], flush=True)


SIGN = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}


def aggregate(package: Path):
    manifest = json.loads((package / "manifest.json").read_text())
    recs = {e["id"]: json.loads((OUT / "programs" / f"{e['id']}.json").read_text()) for e in manifest["programs"]}
    order = sorted(recs, key=lambda p: (recs[p]["family"], p))
    # Holm per quantity over all programs x outputs x rules (the protocol's scope); the default detector:
    # Holm per quantity and family over all its tests, with the tail gate
    for q in QUANTITIES:
        tests = [r for p in order for o in recs[p]["outputs"].values() if "rules" in o[q]
                 for r in o[q]["rules"] if "p_value_two_sided_conservative" in r]
        apply_holm(tests, 0.05)
        for fam in ("vector_mean", "alignment"):
            dt = [t for p in order for o in recs[p]["outputs"].values() if "default_detector" in o[q]
                  for t in o[q]["default_detector"][fam]["tests"]]
            for t, adj in zip(dt, holm_adjusted([t["p"] for t in dt])):
                supported = t.get("diagnostics", {}).get("tail_assumption_supported", True)
                t["holm_adjusted_p_all_programs"] = adj
                t["final_verdict"] = ("DETECTED" if supported else "EXPLORATORY_ONLY") if adj <= 0.05 else "NOT_CONFIRMED"
            for p in order:
                for o in recs[p]["outputs"].values():
                    if "default_detector" in o[q]:
                        vs = [t["final_verdict"] for t in o[q]["default_detector"][fam]["tests"]]
                        o[q]["default_detector"][fam]["final_verdict"] = (
                            "DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY" if "EXPLORATORY_ONLY" in vs
                            else "NOT_CONFIRMED")
    (OUT / "regression_records.json").write_text(json.dumps(recs, indent=2, default=float) + "\n")
    pieces = sorted(ARRAYS.glob("fixture__*.npz"))
    if pieces:
        fx = {}
        for piece in pieces:
            fx.update(dict(np.load(piece)))
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FIXTURE, **fx)

    # comparison with the frozen results
    p1 = json.loads((RES / "phase1_report.json").read_text())
    p2 = json.loads((RES / "phase2" / "phase2_report.json").read_text())
    frozen = {"implementation": lambda pid, name: next(o for o in p1[pid]["outputs"] if o["output"] == name),
              "semantics_published": lambda pid, name: next(o for o in p2[pid]["modes"]["given"] if o["output"] == name),
              "semantics_interface": lambda pid, name: next(o for o in p2[pid]["modes"]["fp32"] if o["output"] == name)}
    changes = {q: {"rules": [], "detector": []} for q in frozen}
    for q, get in frozen.items():
        for p in order:
            for name, o in recs[p]["outputs"].items():
                old = get(p, name)
                new_rules = {r["rule"]: r.get("final_verdict", r["verdict"]) for r in o[q].get("rules", [])}
                old_rules = {r: v.get("final_verdict", v.get("verdict")) for r, v in old["rules"].items()
                             if isinstance(v, dict)} if isinstance(old.get("rules"), dict) else {}
                for rule in sorted(set(new_rules) | set(old_rules)):
                    if new_rules.get(rule) != old_rules.get(rule):
                        changes[q]["rules"].append([p, name, rule, old_rules.get(rule), new_rules.get(rule)])
                for fam in ("vector_mean", "alignment"):
                    a = old.get("default_detector", {}).get(fam, {}).get("final_verdict")
                    b = o[q].get("default_detector", {}).get(fam, {}).get("final_verdict")
                    if a != b:
                        changes[q]["detector"].append([p, name, fam, a, b])
    summary = {"changes_against_frozen": changes, "zero_semantic_rows_from_frozen_report": sorted(map(list, zero_semantic_rows()))}
    (OUT / "regression_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(recs, order, changes)
    print(json.dumps({q: {k: len(v) for k, v in c.items()} for q, c in changes.items()}))


def write_report(recs, order, changes):
    def rows(q, rules):
        out = []
        for p in order:
            for name, o in recs[p]["outputs"].items():
                a = o[q]
                label = p if name == "output" else f"{p}（{name}）"
                res = a.get("residual", {})
                cells = []
                for rule in rules:
                    r = next((x for x in a.get("rules", []) if x["rule"] == rule), None)
                    if r is None or "final_verdict" not in r:
                        cells.append("—")
                    else:
                        cells.append(f"{SIGN.get(r['final_verdict'], '?')} [{r['lower_bound_of_E_l']:.2e}, "
                                     f"{r['upper_bound_of_E_h']:.2e}]")
                det = a.get("default_detector", {})
                d = "/".join({"DETECTED": "检出", "NOT_CONFIRMED": "·", "EXPLORATORY_ONLY": "探索"}.get(
                    det.get(f, {}).get("final_verdict"), "—") for f in ("vector_mean", "alignment"))
                out.append(f"| {label} | {recs[p]['family']} | {res.get('positive_frac', 0):.0%} / "
                           f"{res.get('negative_frac', 0):.0%} / {res.get('contains_zero_frac', 0):.0%} | "
                           + " | ".join(cells) + f" | {d} |")
        return "\n".join(out)

    def detected_lines(q):
        lines = []
        for p in order:
            for name, o in recs[p]["outputs"].items():
                for r in o[q].get("rules", []):
                    if str(r.get("final_verdict", "")).startswith("DETECTED"):
                        label = p if name == "output" else f"{p}（{name}）"
                        lines.append(f"- {label} {r['rule']}{SIGN[r['final_verdict']]}：{interpret(r['rule'], r['final_verdict'])}")
        return "\n".join(lines) or "（无）"

    inv_lines = []
    for p in order:
        inv = recs[p]["interface"]
        sc = [s for s in inv["runtime_scalars"] if s.get("exact") is False]
        cc = inv["rounded_compile_time_constants"]
        if sc or cc:
            parts = [f"{s['name']}: {s['passed']} → {s['received']}（{s['parameter_type']}，相对 {s['relative_rounding']:.2e}）" for s in sc]
            parts += [f"常数 {c['value']} ≈ {c['rounded_from']['candidate']}（相对 {c['rounded_from']['relative_rounding']:.2e}）" for c in cc]
            inv_lines.append(f"| {p} | " + "；".join(parts) + " |")
    chg = []
    for q, c in changes.items():
        chg.append(f"- **{q}**：规则判定变化 {len(c['rules'])} 处；默认检测器判定变化 {len(c['detector'])} 处")
        for item in c["rules"]:
            chg.append(f"  - 规则 {item}")
        for item in c["detector"]:
            chg.append(f"  - 检测器 {item[0]}（{item[1]}）{item[2]}：{item[3]} → {item[4]}")
    head = ("| 程序 | 家族 | 区间 >0 / <0 / 含零 | R1 | R4 | R5 | 默认检测器（向量均值/对齐） |\n|---|---|---|---|---|---|---|")
    head_a = ("| 程序 | 家族 | 区间 >0 / <0 / 含零 | R2 | R3 | 默认检测器（向量均值/对齐） |\n|---|---|---|---|---|---|")
    sections = []
    titles = {"implementation": "实现：K − K_R（数值不一致与偏差）",
              "semantics_interface": "语义（接口口径）：K_R − f_interface",
              "semantics_published": "语义（发布口径）：K_R − f_published",
              "interface_term": "接口项：f_interface − f_published（启动时的标量舍入，上游节点）",
              "total_published": "总差异：K − f_published"}
    for q, title in titles.items():
        sections.append(f"## {title}\n\n### 固定方向均值（R1、R4、R5）\n\n{head}\n{rows(q, ('R1', 'R4', 'R5'))}\n\n"
                        f"### 参照相关的对齐作用（R2、R3）\n\n{head_a}\n{rows(q, ('R2', 'R3'))}\n\n"
                        f"### 检出及其含义\n\n{detected_lines(q)}\n")
    md = f"""# blind_test_v1 回归（修复版工具；冻结结果保持原样）

本文件是修复之后把 blind_test_v1 当回归集重跑的结果，**不替代**已提交的冻结结果（`phase1_report*.md`、`phase2/`）。
这批程序在修复过程中已被看过，新的泛化成绩应来自另一批未参与修改的组合。

修复内容：默认检测器第 2 版（每个出口都按区间做端点保守检验，见 `detect.py`）；正式投影用有向乘法与精确求和；
端点保守 p 值改为 2·min(p(E[l] > 0), p(E[h] < 0))（旧式在跨零区间上偏小，会让 Holm 把它算作拒绝）；判定统一走
`analysis.assess_units`，盲测脚本只做输入与规格绑定。同一次运行（seed 0–31 开发、32–95 确认）评估五个量，
K − f_published = 实现 + 语义（发布口径），语义（发布口径）= 语义（接口口径）+ 接口项。

表格：+ / − 为 Holm 后的检出及 μ 的符号，· 为未确认，括号内为 μ 的端点保守区间（单格、未校正，判定以 Holm 为准）。
**μ 的符号不等于原始残差的符号**：R1 的方向是 −1/√n，R2、R3 也带负号，各检出的含义在「检出及其含义」中逐条写出。

## 与冻结结果的对比

{chr(10).join(chg)}

## 接口与编译期常数清单

| 程序 | 启动时舍入的运行时标量；编译期舍入的常数（候选值只是从存储值读出的推测） |
|---|---|
{chr(10).join(inv_lines)}

{chr(10).join(sections)}
## 离线复算材料

`regression_records.json`：每个程序、每个输出、每个量的记录——坐标集大小、开发/确认 seed、每条规则每个确认 seed 的
投影上下界、实际方向（常数、按单位的归一化标量，或保存的向量名）、原始 p 与 Holm 校正后 p、排除原因、默认检测器各检验
的 p 与校正后 p。向量（坐标集、R4、R5、检测器的学习方向与分组方向）在 `.cache/blind_v1_regression_v2/<程序>.npz`。
"""
    (OUT / "regression_report.md").write_text(md)
    print("written", OUT / "regression_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--programs", default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.package)
    else:
        per_program(args)


if __name__ == "__main__":
    main()
