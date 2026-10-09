#!/usr/bin/env python3
"""Summarize a dsl-v2 regression run of the v1.1 programs against its registered expectations
(docs/dsl_v2/increment_01.md section 4) and against general-v3.1 (run r20261008T2030).  Reads job files only.

    python scripts/dsl_v2/regression_report.py --run-id ID
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "1_experiments/dsl_v2/regression_v11"
CLASSES = ("fixed_mean", "aligned")

# registered before the run (increment_01.md section 4)
EXPECT = {**{p: "complete" for p in ("prog_17", "prog_20", "prog_32", "prog_08")},
          "prog_29": "not established (0/0)",
          **{p: "race" for p in ("prog_01", "prog_13", "prog_22", "prog_23")},
          **{p: "atomic average" for p in ("prog_11", "prog_18", "prog_19", "prog_27")}}


def short(summary):
    if summary.startswith("average effect nonzero"):
        return "nonzero: " + summary.split(":", 1)[1].strip()
    return "cannot judge / n.e." if summary.startswith("cannot judge") else "not confirmed"


def observed(job):
    outs = job.get("outputs", {})
    if job.get("status") != "ok":
        return "job error"
    ev = [o for o in outs.values() if o["status"] == "evaluated"]
    if len(ev) < len(outs):
        return "not established (" + "; ".join(o.get("reason", "")[:60] for o in outs.values() if o["status"] != "evaluated") + ")"
    ex = [o["execution"]["statistics"] for o in ev]
    if any(o["execution"]["race_findings"] for o in ev):
        return "race"
    if all(s == "within-input mean" for s in ex):
        return "atomic average"
    if all(o["reference"]["complete_rate"] == 1.0 for o in ev):
        return "complete"
    return "partial reference (" + ", ".join(f"{o['reference']['complete_rate']:.3g}" for o in ev) + ")"


def build(run_id):
    d = OUT / run_id
    jobs = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("prog_*.json"))}
    rows, safety_violations = [], []
    for pid, job in jobs.items():
        obs = observed(job)
        exp = EXPECT.get(pid, "identical to v3.1")
        cmp_ = job.get("v31_comparison") or {}
        if exp == "identical to v3.1":
            same = all(r.get("reference_equal") and r.get("e_num_record_equal") for r in cmp_.values()) if cmp_ else False
            met = same and obs == "complete" or (same and obs.startswith("partial"))
        else:
            met = obs == exp
        e_num = {}
        for name, o in (job.get("outputs") or {}).items():
            if o["status"] == "evaluated":
                e_num[name] = {c: short(o["e_num_classes"][c]["summary"]) for c in CLASSES}
                if o["execution"]["statistics"] == "withheld" and any(
                        o["e_num_classes"][c]["summary"].startswith("average effect nonzero") for c in CLASSES):
                    safety_violations.append(f"{pid}/{name}")
        rows.append({"program": pid, "family": job.get("family"), "expected": exp, "observed": obs, "met": bool(met),
                     "launches_per_input": job.get("launches_per_input"), "seconds": job.get("seconds"),
                     "e_num": e_num,
                     "execution": {n: o["execution"]["status"] for n, o in (job.get("outputs") or {}).items()
                                   if o["status"] == "evaluated"},
                     "v31_comparison": cmp_})
    return {"run_id": run_id, "programs": len(rows), "expectations_met": sum(r["met"] for r in rows),
            "safety_violations": safety_violations, "rows": rows}


def report(s):
    L = [f"# DSL v2 回归（v1.1 程序，工具 4.0，运行 {s['run_id']}）", "",
         "v1.1 的程序从此只作开发集与回归集，这些不是盲测成绩。预计写在运行之前（docs/dsl_v2/increment_01.md §4），"
         "这里只核对命中与否，不改写预计。", "",
         f"- 预计命中 {s['expectations_met']}/{s['programs']}。",
         f"- 安全标准（判为竞争或不一致、统计不做的输出上出现「非零」）：违反 {len(s['safety_violations'])} 处"
         + (f"：{s['safety_violations']}" if s["safety_violations"] else "") + "。", "",
         "| 程序 | 结构 | 预计 | 观察 | 命中 | 每输入启动数 | K−G（固定均值 / 对齐） | 执行状态 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in s["rows"]:
        en = "; ".join(f"{n}: {v['fixed_mean']} / {v['aligned']}" for n, v in r["e_num"].items())
        ex = "; ".join(sorted(set(r["execution"].values())))[:80]
        L.append(f"| {r['program']} | {r['family']} | {r['expected']} | {r['observed'][:60]} | {'是' if r['met'] else '否'} | "
                 f"{r['launches_per_input']} | {en} | {ex} |")
    L += ["", "「identical to v3.1」的程序逐项比较了 kernel 级参照质量与 K−G 规则记录（逐单位投影端点），两者都与 v3.1 相同才记命中。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    s = build(a.run_id)
    d = OUT / a.run_id
    (d / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (d / "SUMMARY.md").write_text(report(s))
    print(s["expectations_met"], "/", s["programs"], "safety violations:", s["safety_violations"])


if __name__ == "__main__":
    main()
