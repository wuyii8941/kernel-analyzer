#!/usr/bin/env python3
"""Scope revision of run r20261008T2030 (after the issuer's and reviewer's handling of 2026-10-08): the raw execution
table (all 33 items) and the valid-measurement table (which items meet the premise of the finite-mean comparison).
The label-scoring table waits for the original answer files.  Reads the job files only; the original REPORT.md,
summary.json, p-values and family-wise decisions are not changed.

    python scripts/acceptance/structure_v11_scope_tables.py --run-id r20261008T2030
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT.parent / ".cache" / "acceptance" / "structure_v1_1_rc1"  # extract the frozen run archive here
PKG = ROOT.parent / ".cache" / "acceptance" / "structure_acceptance_v1_1_rc1"
RACE_RULING = {"prog_01", "prog_13", "prog_22", "prog_23"}  # issuer ruling of 2026-10-08 (intra-program race)
COMPS = ["FR_e_num", "FR_e_sem", "F_total", "baseline"]


def load(d: Path, sub: str):
    out = {}
    for p in sorted((d / sub).glob("*.json")):
        j = json.loads(p.read_text())
        out[(j["round"], j["mode"], j["program"])] = j
    return out


def nonfinite(o):
    sv = (o or {}).get("special_values") or {}
    return sv.get("k_vs_f_class_mismatch")


def build(run_id):
    d = RESULTS / run_id
    jobs, sup = load(d, "jobs"), load(d, "supplement_D1")
    man = {e["id"]: e for e in json.loads((PKG / "manifest.json").read_text())["programs"]}
    raw, valid = [], []
    for pid, e in sorted(man.items()):
        status = {f"{r}_{m}": (jobs.get((r, m, pid)) or {}).get("status", "not run")
                  for r, m in (("main", "B"), ("main", "A"), ("replication", "B"))}
        status.update({f"D1_{r}": (sup.get((r, "B", pid)) or {}).get("status")
                       for r in ("main", "replication") if (r, "B", pid) in sup})
        jb, ja, js = jobs.get(("main", "B", pid)), jobs.get(("main", "A", pid)), sup.get(("main", "B", pid))
        for name in e["output_names"]:
            ob = (jb or {}).get("outputs", {}).get(name)
            oa = (ja or {}).get("outputs", {}).get(name)
            os_ = (js or {}).get("outputs", {}).get(name)
            ref = ("not run" if ob is None else "established" if ob.get("status") == "evaluated" and
                   ob["reference"]["complete_rate"] == 1.0 else f"not established ({ob.get('failure_class')})")
            nf = nonfinite(ob) if ob and ob.get("status") == "evaluated" else nonfinite(os_)
            raw.append({"program": pid, "family": e["family"], "output": name, "lane": e["execution_lane"],
                        "jobs": status, "reference_kernel_level": ref, "call_level": "not established",
                        "nonfinite_K_elements_main": nf})
            atomic = int(e.get("execution_repeats", 1)) > 1
            repro = None
            if ob and oa and "k_sha256_per_unit" in ob and "k_sha256_per_unit" in oa:
                a, b = oa["k_sha256_per_unit"], ob["k_sha256_per_unit"]
                if atomic:  # per execution digests; compare the whole set per input is not meaningful bitwise
                    repro = "order-random by design (atomic): not required bitwise"
                else:
                    repro = f"{sum(x == y for x, y in zip(a, b))}/{len(a)} units bitwise equal between the two main-round launches (A, B)"
            elif e["execution_lane"] == "measurement":
                repro = "not checked (one launch per input with a kept K)"
            avail = [c for c in COMPS if c in ((ob or {}).get("comparisons") or {}) or c in ((os_ or {}).get("comparisons") or {})]
            if e["execution_lane"] != "measurement":
                premise = ("boundary item: not run this round (environment permission check); not a correct rejection, "
                           "not a numerical negative; does not block other scoring")
            elif pid in RACE_RULING:
                premise = ("race boundary (issuer ruling 2026-10-08: buf written then read in the same program without a "
                           "barrier); removed from finite-mean label scoring; raw records and execution validity kept")
            elif atomic:
                premise = "valid under the atomic protocol (8 executions per input, within-input mean); execution randomness by design"
            else:
                premise = "valid"
            valid.append({"program": pid, "family": e["family"], "output": name, "finite_mean_premise": premise,
                          "launch_reproducibility": repro, "comparison_objects_available": avail,
                          "reference_kernel_level": ref, "nonfinite_K_elements_main": nf})
    progs = sorted(man)
    ok_prog = [p for p in progs if all(v["finite_mean_premise"].startswith("valid") for v in valid if v["program"] == p)]
    est = [p for p in ok_prog if all(v["reference_kernel_level"] == "established" for v in valid if v["program"] == p)]
    summary = {"raw_items": len(progs), "programs_with_valid_finite_mean_premise": len(ok_prog),
               "of_those_reference_established_all_outputs": len(est),
               "excluded": {"race_ruling": sorted(RACE_RULING), "boundary_not_run": [p for p in progs if
                                                                                   man[p]["execution_lane"] != "measurement"]}}
    return d, raw, valid, summary


def md(raw, valid, summary):
    L = ["# 范围修订：原始执行表与有效测量表（运行 r20261008T2030）", "",
         "依据 2026-10-08 出题方与审阅方的处理。原报告 `REPORT.md`、`summary.json`、p 值与各检验族的判断都不改。本轮登记的是",
         "608 个互相独立的检验族，每个族内做 Holm，不是一次全局 Holm。标签评分表要等原出题方交付答案文件，现在不做。", "",
         "- 原报告的程序级「27/33 建立」属于原始执行表，含 T5 的四个常规项，不能当作修订后有效任务的成功率。",
         f"- 有限均值比较前提成立的程序 {summary['programs_with_valid_finite_mean_premise']}/{summary['raw_items']}；其中全部输出参照已建立的 "
         f"{summary['of_those_reference_established_all_outputs']}/{summary['programs_with_valid_finite_mean_premise']}（未建立的是 5 个 T1 程序，"
         "它们的 F 与 baseline 仍可用）。",
         f"- 排除：竞争裁决 {summary['excluded']['race_ruling']}，移出有限均值标签评分但保留记录；边界项 {summary['excluded']['boundary_not_run']}，本轮未执行。",
         "- 这个集合给不出均值检测的误报率：一致性对检查的是程序之间的差，不提供各程序相对 G 的零均值标签。", "",
         "## 原始执行表（全部 33 项）", "",
         "| 程序 | 结构 | 输出 | 通道 | 作业状态 | 参照（kernel 级） | 主轮 K 非有限元素 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in raw:
        st = ", ".join(f"{k} {v}" for k, v in r["jobs"].items() if v)
        L.append(f"| {r['program']} | {r['family']} | {r['output']} | {r['lane']} | {st} | {r['reference_kernel_level']} | "
                 f"{r['nonfinite_K_elements_main'] if r['nonfinite_K_elements_main'] is not None else '–'} |")
    L += ["", "调用级完整一律未建立（协议 §1）。", "", "## 有效测量表", "",
          "| 程序 | 输出 | 有限均值比较前提 | 启动可复现性 | 可用比较对象 |", "| --- | --- | --- | --- | --- |"]
    for v in valid:
        L.append(f"| {v['program']} | {v['output']} | {v['finite_mean_premise']} | {v['launch_reproducibility'] or '–'} | "
                 f"{', '.join(v['comparison_objects_available']) or '–'} |")
    L += ["", "## 标签评分表", "", "未做：等原出题方交付两份原始答案文件、各自完整的 64 位 SHA-256、对应的包版本与计分协议，以及此前的承诺记录。",
          "本轮定位预算为 0，答案里若有定位标签，记为「未执行定位」。", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    d, raw, valid, summary = build(a.run_id)
    out = d / "scope_revision_20261008"
    out.mkdir(exist_ok=True)
    (out / "tables.json").write_text(json.dumps({"summary": summary, "raw_execution": raw, "valid_measurement": valid},
                                                indent=1) + "\n")
    (out / "README.md").write_text(md(raw, valid, summary))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
