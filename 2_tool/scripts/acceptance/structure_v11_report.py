#!/usr/bin/env python3
"""Aggregate a structure acceptance v1.1-rc1 run (jobs/*.json) into summary.json and REPORT.md.  No measurement:
it reads the job files, the entry-equivalence files and RUN_FREEZE.json only.  Written after the freeze.

    python scripts/acceptance/structure_v11_report.py --run-id ID
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT.parent / ".cache" / "acceptance" / "structure_v1_1_rc1"  # extract the frozen run archive here
PKG = ROOT.parent / ".cache" / "acceptance" / "structure_acceptance_v1_1_rc1"
CLASSES = ["fixed_mean", "aligned"]
COMPS = ["FR_e_num", "FR_e_sem", "F_total", "baseline"]


def short(summary: str) -> str:
    if summary.startswith("average effect nonzero"):
        return "nonzero: " + summary.split(":", 1)[1].strip()
    if summary.startswith("cannot judge"):
        return "cannot judge / n.e."
    if summary.startswith("not confirmed"):
        return "not confirmed"
    return summary[:30]


def bucket(summary: str) -> str:
    return ("nonzero" if summary.startswith("average effect nonzero") else
            "cannot judge / not established" if summary.startswith("cannot judge") else "not confirmed")


def load(run_id):
    d = RESULTS / run_id
    jobs = {}
    for p in sorted((d / "jobs").glob("*.json")):
        j = json.loads(p.read_text())
        jobs[(j["round"], j["mode"], j["program"])] = j
    entry = {p.stem: json.loads(p.read_text()) for p in sorted((d / "entry_equivalence").glob("*.json"))}
    sup = {}
    for p in sorted((d / "supplement_D1").glob("*.json")):
        j = json.loads(p.read_text())
        sup[(j["round"], j["program"])] = j
    freeze = json.loads((d / "RUN_FREEZE.json").read_text())
    man = {e["id"]: e for e in json.loads((PKG / "manifest.json").read_text())["programs"]}
    return d, jobs, entry, freeze, man, sup


def established(o) -> bool:
    return o.get("status") == "evaluated" and o["reference"]["complete_rate"] == 1.0


def rule_rows(st, cls):
    out = []
    for rn, j in st["class_statistics_protocol"][cls]["rules"].items():
        out.append({"rule": rn, "judgment": j["judgment"], "mean_projection": j.get("mean_projection"), "n": j.get("n"),
                    "p_raw": next((r.get("p_value_two_sided_conservative") for r in st["record"].get("rules", [])
                                   if r.get("rule") == rn), None),
                    "p_holm_registered": j.get("holm_adjusted_p_registered_family"), "basis": j.get("basis")})
    return out


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def build(run_id):
    d, jobs, entry, freeze, man, sup = load(run_id)
    summary = {"run_id": run_id, "tool_commit": freeze["tool_commit"], "package_sha256sums_sha256":
               freeze["package_sha256sums_sha256"], "protocol_sha256": freeze["protocol_sha256"],
               "answer_commitment": freeze["answer_commitment"], "scoring": freeze["scoring"]}
    expected = json.loads((d / "jobs.json").read_text())
    # ---------------------------------------------------------------- execution validity
    status = Counter(j.get("status") for j in jobs.values())
    missing_jobs = [f"{e['round']}_{e['mode']}_{e['program']}" for e in expected
                    if (e["round"], e["mode"], e["program"]) not in jobs]
    summary["execution"] = {"jobs_expected": len(expected), "jobs_with_result": len(jobs), "status": dict(status),
                            "missing": missing_jobs,
                            "not_ok": {f"{k[0]}_{k[1]}_{k[2]}": {"status": j.get("status"), "reason": j.get("reason"),
                                                                 "failure_class": j.get("failure_class")}
                                       for k, j in jobs.items() if j.get("status") != "ok"}}
    # ---------------------------------------------------------------- completeness (main B is the primary run)
    per_prog, per_out = {}, []
    for pid, e in sorted(man.items()):
        if e["execution_lane"] != "measurement":
            per_prog[pid] = {"family": e["family"], "lane": e["execution_lane"], "status": "not run (isolated opt-in "
                             "not approved)", "all_outputs_established": False}
            for name in e["output_names"]:
                per_out.append({"program": pid, "output": name, "family": e["family"], "status": "not run",
                                "established": False})
            continue
        j = jobs.get(("main", "B", pid))
        if j is None or j.get("status") != "ok":
            per_prog[pid] = {"family": e["family"], "lane": "measurement", "status": (j or {}).get("status", "missing"),
                             "reason": (j or {}).get("reason"), "all_outputs_established": False}
            for name in e["output_names"]:
                per_out.append({"program": pid, "output": name, "family": e["family"], "status": "job missing" if j is None else "job not ok",
                                "established": False, "failure_class": (j or {}).get("failure_class")})
            continue
        ests = []
        for name in e["output_names"]:
            o = j["outputs"][name]
            row = {"program": pid, "output": name, "family": e["family"], "status": o["status"],
                   "established": established(o)}
            if o["status"] == "evaluated":
                q = o["reference"]
                row.update(complete_rate=q["complete_rate"], resolved_fraction=q["resolved_fraction"],
                           width_over_ulp=q["width_over_ulp"], tool_scope=o["scope"]["tool_classification"],
                           failure_classes=o["failure_classes"],
                           e_sem_certified=(o.get("semantic_exact_comparison") or {}).get("e_sem_certified"))
            else:
                row.update(reason=o.get("reason"), failure_class=o.get("failure_class"))
            ests.append(row["established"])
            per_out.append(row)
        per_prog[pid] = {"family": e["family"], "lane": "measurement", "status": "ok",
                         "ttir_coverage_complete": j["notes"].get("ttir_coverage_complete"),
                         "all_outputs_established": all(ests)}
    n_all = len(man)
    n_reg = sum(1 for e in man.values() if e["execution_lane"] == "measurement")
    k = sum(1 for p in per_prog.values() if p["all_outputs_established"])
    k_cov = sum(1 for pid, p in per_prog.items() if p.get("ttir_coverage_complete") and all(p["ttir_coverage_complete"]))
    summary["completeness"] = {
        "definition": "an output is established when the main-round mode-B job evaluated it and its kernel-level "
                      "reference is complete and finite on every element of every unit (complete_rate = 1); a program "
                      "when all its required outputs are",
        "programs_all_outputs_established": {"of_33": [k, n_all], "of_32_regular": [k, n_reg]},
        "outputs_established": [sum(r["established"] for r in per_out), len(per_out)],
        "ttir_coverage_complete_programs": [k_cov, n_reg],
        "call_level": "not established for any program (PROTOCOL section 1)",
        "boundary": {pid: p["status"] for pid, p in per_prog.items() if p["lane"] != "measurement"},
        "per_program": per_prog, "per_output": per_out}
    # ---------------------------------------------------------------- statistics
    stats = []
    for (rnd, mode, pid), j in sorted(jobs.items()):
        if j.get("status") != "ok":
            continue
        for name, o in j["outputs"].items():
            for comp, st in (o.get("comparisons") or {}).items():
                for cls in CLASSES:
                    stats.append({"round": rnd, "mode": mode, "program": pid, "family": man[pid]["family"],
                                  "output": name, "comparison": comp, "class": cls,
                                  "summary": st["class_statistics_protocol"][cls]["summary"],
                                  "summary_frozen_holm": st["class_statistics_frozen"][cls]["summary"],
                                  "rules": rule_rows(st, cls), "source": "main job"})
    for (rnd, pid), j in sorted(sup.items()):  # deviation D1: K - f and K - f64 where the reference is not established
        if j.get("status") != "ok":
            continue
        for name, o in j["outputs"].items():
            for comp, st in o["comparisons"].items():
                for cls in CLASSES:
                    stats.append({"round": rnd, "mode": "B", "program": pid, "family": man[pid]["family"],
                                  "output": name, "comparison": comp, "class": cls,
                                  "summary": st["class_statistics_protocol"][cls]["summary"],
                                  "summary_frozen_holm": st["class_statistics_frozen"][cls]["summary"],
                                  "rules": rule_rows(st, cls), "source": "supplement D1"})
    summary["supplement_D1"] = {f"{r}_{p}": {"status": j.get("status"), "reason": j.get("reason"),
                                             "seconds": j.get("seconds")} for (r, p), j in sorted(sup.items())}
    summary["statistics"] = stats
    # family table with every registered family (missing cells retained)
    fam = json.loads((d / "families.json").read_text())
    got = {f"{s['round']}/{s['program']}/{s['output']}/{s['comparison']}/{s['class']}": s for s in stats if s["mode"] == "B"}
    fam_rows = []
    for f in fam:
        s = got.get(f["family"])
        fam_rows.append({"family": f["family"], "status": bucket(s["summary"]) if s else "not established (missing cell)"})
    counts = defaultdict(Counter)
    for r in fam_rows:
        rnd, pid, name, comp, cls = r["family"].split("/")
        counts[f"{rnd}/{comp}/{cls}"][r["status"]] += 1
    summary["families"] = {"registered": len(fam), "counts": {k: dict(v) for k, v in sorted(counts.items())},
                           "rows": fam_rows}
    # statistics value vs automatic-reference value: K - f64 (plain reference) against K - f (exact specification)
    pair = defaultdict(dict)
    for x in stats:
        if x["mode"] == "B" and x["comparison"] in ("F_total", "baseline"):
            pair[(x["round"], x["program"], x["output"], x["class"])][x["comparison"]] = short(x["summary"])
    same = [k for k, v in pair.items() if len(v) == 2 and v["F_total"] == v["baseline"]]
    summary["baseline_vs_F_total"] = {"compared": sum(1 for v in pair.values() if len(v) == 2), "same_summary": len(same),
                                      "different": [{"key": "/".join(k), **v} for k, v in pair.items()
                                                    if len(v) == 2 and v["F_total"] != v["baseline"]]}
    # ---------------------------------------------------------------- replication agreement (descriptive)
    rep_rows = []
    for s in stats:
        if s["round"] != "main" or s["mode"] != "B":
            continue
        t = next((x for x in stats if x["round"] == "replication" and x["mode"] == "B" and x["program"] == s["program"]
                  and x["output"] == s["output"] and x["comparison"] == s["comparison"] and x["class"] == s["class"]), None)
        if t is not None:
            rep_rows.append({"program": s["program"], "output": s["output"], "comparison": s["comparison"],
                             "class": s["class"], "main": short(s["summary"]), "replication": short(t["summary"])})
    agree = Counter((r["main"] == r["replication"]) for r in rep_rows)
    summary["replication"] = {"note": "rounds corrected separately and never pooled; agreement is descriptive",
                              "same_summary": agree.get(True, 0), "different_summary": agree.get(False, 0),
                              "rows": rep_rows}
    # ---------------------------------------------------------------- mode A vs B consistency
    ab = []
    for pid, e in sorted(man.items()):
        a, b = jobs.get(("main", "A", pid)), jobs.get(("main", "B", pid))
        if not a or not b or a.get("status") != "ok" or b.get("status") != "ok":
            ab.append({"program": pid, "status": "not comparable (a job missing or not ok)"})
            continue
        for name in e["output_names"]:
            oa, ob = a["outputs"][name], b["outputs"][name]
            row = {"program": pid, "output": name, "atomic": int(e.get("execution_repeats", 1)) > 1}
            if oa["status"] != ob["status"]:
                row["status_equal"] = False
            elif oa["status"] == "evaluated":
                row.update(status_equal=True, K_bitwise_equal=oa["k_sha256_per_unit"] == ob["k_sha256_per_unit"],
                           reference_equal=json.dumps(oa["reference"], sort_keys=True) == json.dumps(ob["reference"],
                                                                                                    sort_keys=True),
                           e_num_summary_equal=all(oa["comparisons"]["FR_e_num"]["class_statistics_protocol"][c]["summary"]
                                                   == ob["comparisons"]["FR_e_num"]["class_statistics_protocol"][c]["summary"]
                                                   for c in CLASSES))
            else:
                row.update(status_equal=True, reason_equal=oa.get("reason") == ob.get("reason"))
            ab.append(row)
    summary["mode_A_vs_B"] = ab
    # ---------------------------------------------------------------- adapter vs frozen entry
    eq = {}
    for pid, res in entry.items():
        j = jobs.get(("main", "B", pid))
        if not j or j.get("status") != "ok" or res.get("status") != "ok":
            eq[pid] = {"status": "not comparable", "entry_status": res.get("status")}
            continue
        lv = res["tool_report"]["levels"][0]
        rows = {}
        for name in man[pid]["output_names"]:
            fo = (lv.get("outputs") or {}).get(name) or {}
            ao = j["outputs"][name]
            r = {"entry_status": fo.get("status"), "adapter_status": ao["status"]}
            if fo.get("status") == "evaluated" and ao["status"] == "evaluated":
                r["reference_equal"] = all(fo["reference"].get(k) == ao["reference"].get(k)
                                           for k in ("elements", "complete_finite", "complete_rate", "width_over_ulp",
                                                     "resolved_fraction"))
                r["e_num_class_statistics_equal"] = json.dumps(fo["statistics"], sort_keys=True) == json.dumps(
                    ao["comparisons"]["FR_e_num"]["class_statistics_frozen"], sort_keys=True)
                base = (res.get("baseline") or {}).get("levels", {}).get("{}", {}).get(name)
                r["baseline_class_statistics_equal"] = base is not None and json.dumps(base, sort_keys=True) == \
                    json.dumps(ao["comparisons"]["baseline"]["class_statistics_frozen"], sort_keys=True)
                r["semantic_equal"] = json.dumps(fo.get("semantic"), sort_keys=True) == json.dumps(
                    {k: v for k, v in ao.get("semantic_exact_comparison", {}).items()}, sort_keys=True)
            rows[name] = r
        eq[pid] = {"status": "compared", "outputs": rows, "entry_seconds": res.get("seconds")}
    summary["entry_equivalence"] = eq
    # ---------------------------------------------------------------- E / F / P / renaming
    ef, props = [], []
    for (rnd, mode, pid), j in sorted(jobs.items()):
        if j.get("status") != "ok" or mode != "B":
            continue
        props.append({"round": rnd, "program": pid, **j["property"]})
        for name, o in j["outputs"].items():
            if o["status"] == "evaluated":
                ef.append({"round": rnd, "program": pid, "output": name, "E": o["elementwise"].get("E_K_vs_ref64"),
                           "F": o["elementwise"].get("F_K_minus_f")})
    for (rnd, pid), j in sorted(sup.items()):
        for name, o in (j.get("outputs") or {}).items():
            ef.append({"round": rnd, "program": pid, "output": name + " (D1)", "E": o["elementwise"].get("E_K_vs_ref64"),
                       "F": o["elementwise"].get("F_K_minus_f")})
    summary["E_F"] = ef
    summary["P"] = props
    groups = defaultdict(list)
    for pid, e in sorted(man.items()):
        j = jobs.get(("main", "B", pid))
        if j and j.get("status") == "ok" and int(e.get("execution_repeats", 1)) == 1:
            for name, o in j["outputs"].items():
                if "k_sha256_per_unit" in o:
                    groups[(e["family"], name, tuple(o["k_sha256_per_unit"]))].append(pid)
    summary["renaming_relation"] = {
        "note": "programs of one family whose K is bitwise identical on all 96 main-round units (deterministic "
                "programs only; atomic programs are not required to agree bitwise)",
        "identical_groups": [{"family": f, "output": n, "programs": ps} for (f, n, _), ps in groups.items() if len(ps) > 1]}
    # ---------------------------------------------------------------- failure classes and cost
    fc = Counter()
    for r in per_out:
        for c in r.get("failure_classes") or ([r["failure_class"]] if r.get("failure_class") else []):
            fc[c] += 1
    summary["failure_classes_main_B_outputs"] = dict(fc)
    cost = []
    for (rnd, mode, pid), j in sorted(jobs.items()):
        cost.append({"round": rnd, "mode": mode, "program": pid, "seconds": j.get("seconds"),
                     "timing": j.get("timing_seconds"), "units": len(j.get("executions") or [])})
    summary["cost"] = cost
    return d, summary, man


def report_md(d, s, man):
    L = []
    w = L.append
    c = s["completeness"]
    w(f"# 结构验收集 v1.1-rc1 运行 {s['run_id']}（冻结工具 general-v3.1）\n")
    w("> 本报告只汇总冻结规则下的测量，不是正式盲评分数：包状态 `ISSUER_RESEAL_REQUIRED`，新答案未签署，"
      "敏感度 / 特异度不计算（seal_status.json）。结论只按三种状态写：建立 / 未建立 / 无法判断。\n")
    w(f"- 工具提交 `{s['tool_commit'][:12]}`（src 树与 general-v3.1 相同），冻结记录 `RUN_FREEZE.json` 在测量前提交；"
      f"包 SHA256SUMS 哈希 `{s['package_sha256sums_sha256'][:16]}…`，协议哈希 `{s['protocol_sha256'][:16]}…`。")
    ex = s["execution"]
    w(f"- 作业：{ex['jobs_with_result']}/{ex['jobs_expected']} 有结果，状态 {ex['status']}；缺失 {len(ex['missing'])}。")
    w("- 汇总脚本 `scripts/acceptance/structure_v11_report.py` 写于冻结之后，只读作业文件，不做测量。\n")
    w("## 1. 完整率（主轮、模式 B）\n")
    a33, a32 = c["programs_all_outputs_established"]["of_33"], c["programs_all_outputs_established"]["of_32_regular"]
    w(f"- 程序级「全部必要输出已建立」：{a33[0]}/{a33[1]}（全部 33 项）；{a32[0]}/{a32[1]}（32 个常规项条件比例）。")
    w(f"- 输出级：{c['outputs_established'][0]}/{c['outputs_established'][1]} 个输出参照完整（kernel 级）。")
    w(f"- TTIR 覆盖完整的程序：{c['ttir_coverage_complete_programs'][0]}/{c['ttir_coverage_complete_programs'][1]}。")
    w("- 调用级完整：一律未建立（协议 §1：仅凭内容相同或全零不升级为调用级）；工具自身的调用级分类另列在 summary.json。")
    w(f"- 边界项：{c['boundary']}，不计入有限均值检出率，也不算作正确拒绝。\n")
    w("| 程序 | 结构 | 输出 | 状态 | 完整率 | 分辨率达标 | 宽度/ulp 中位 / p90 / max | 失败类别或原因 |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in c["per_output"]:
        wu = r.get("width_over_ulp") or {}
        why = ", ".join(r.get("failure_classes") or []) or (f"{r.get('failure_class')}: {r.get('reason')}"
                                                            if r.get("reason") else "")
        w(f"| {r['program']} | {r['family']} | {r['output']} | {'建立' if r['established'] else r['status']} | "
          f"{fmt(r.get('complete_rate'))} | {fmt(r.get('resolved_fraction'))} | "
          f"{fmt(wu.get('median'))} / {fmt(wu.get('p90'))} / {fmt(wu.get('max'))} | {why[:90]} |")
    w("\n## 2. 统计（Holm 按登记 family，缺格按 p = 1）\n")
    fams = s["families"]
    w(f"登记 family {fams['registered']} 个（两轮 × 33 程序 × 输出 × 4 个比较对象 × 2 个规则类）。各轮各对象的状态计数：\n")
    w("| 轮 / 对象 / 规则类 | 非零 | 未确认 | 无法判断 / 未建立 |")
    w("| --- | --- | --- | --- |")
    for k, v in fams["counts"].items():
        w(f"| {k} | {v.get('nonzero', 0)} | {v.get('not confirmed', 0)} | "
          f"{v.get('cannot judge / not established', 0) + v.get('not established (missing cell)', 0)} |")
    for cls in CLASSES:
        w(f"\n### {('固定向量均值（R1, R5）' if cls == 'fixed_mean' else '对齐作用（R2, R3）')}，主轮模式 B\n")
        w("| 程序 | 输出 | K−G | G−f | K−f | 普通参照 K−f64 |")
        w("| --- | --- | --- | --- | --- | --- |")
        rows = defaultdict(dict)
        for st in s["statistics"]:
            if st["round"] == "main" and st["mode"] == "B" and st["class"] == cls:
                rows[(st["program"], st["output"])][st["comparison"]] = short(st["summary"])
        for (pid, name), v in sorted(rows.items()):
            w(f"| {pid} | {name} | " + " | ".join(v.get(cp, "–") for cp in COMPS) + " |")
    w("\n规则级效应、原始与校正 p 值、逐单位投影端点在 `jobs/*.json`（`outputs.<name>.comparisons.<对象>.record`）。"
      "「未确认」不等于没有 bias；「非零」是对该投影平均作用的判断，不是逐元素契约违反。\n")
    w("### 非零判断的规则级明细（模式 B，两轮）\n")
    w("| 轮 | 程序 | 输出 | 对象 | 规则类 | 规则 | 判断 | 平均投影 | n | 原始 p | Holm p（登记 family） |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for st in s["statistics"]:
        if st["mode"] != "B":
            continue
        for r in st["rules"]:
            if r["judgment"].startswith("nonzero"):
                w(f"| {st['round']} | {st['program']} | {st['output']} | {st['comparison']} | {st['class']} | {r['rule']} | "
                  f"{r['judgment']} | {fmt(r['mean_projection'])} | {fmt(r['n'])} | {fmt(r['p_raw'])} | "
                  f"{fmt(r['p_holm_registered'])} |")
    certified = [r for r in c["per_output"] if r.get("e_sem_certified")]
    w(f"\nG 与 f 的包围不相交（e_sem 逐元素确证）的输出：{[(r['program'], r['output'], r['e_sem_certified']) for r in certified] or '无'}。"
      "这类元素要么是程序语义与任务规格的真实差异，要么是某一侧包围的反例，本轮不判定、不计分。\n")
    rp = s["replication"]
    w(f"## 3. 复现轮（种子 96–191，开发块 0–31 固定坐标与学习方向）\n\n两轮分别校正、不合并。主轮与复现轮摘要相同 "
      f"{rp['same_summary']} 个，不同 {rp['different_summary']} 个：\n")
    w("| 程序 | 输出 | 对象 | 规则类 | 主轮 | 复现轮 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for r in rp["rows"]:
        if r["main"] != r["replication"]:
            w(f"| {r['program']} | {r['output']} | {r['comparison']} | {r['class']} | {r['main']} | {r['replication']} |")
    bv = s["baseline_vs_F_total"]
    w(f"\n普通参照 + 相同统计（K − f64）与精确规格 K − f 的摘要相同 {bv['same_summary']}/{bv['compared']} 个格"
      + (f"；不同：{bv['different']}" if bv["different"] else "") + "。T1 的 K − f 与 K − f64 来自补跑 D1（DEVIATIONS.md）。")
    w("\n## 4. 一致性检查\n")
    bad_ab = [r for r in s["mode_A_vs_B"] if not r.get("atomic") and r.get("status_equal") is not None and not (
        r.get("status_equal") and r.get("K_bitwise_equal", True) and r.get("reference_equal", True)
        and r.get("e_num_summary_equal", True))]
    w(f"- 模式 A 与 B（主轮，非原子程序）：K 逐位、参照质量与 K−G 摘要不一致的输出 {len(bad_ab)} 个"
      + (f"：{[(r['program'], r.get('output')) for r in bad_ab]}" if bad_ab else "") + "。原子程序 K 随执行次序变化，单列。")
    for pid, r in s["entry_equivalence"].items():
        w(f"- 冻结入口（`acceptance_run.run_case` → `measure.run` + 基线）对照 {pid}：{json.dumps(r.get('outputs', r), ensure_ascii=False)}")
    w("\n## 5. E / F / P\n")
    w("| 轮 | 程序 | 输出 | E：K vs f64 相对 RMS | E：max/ulp | F：区间排除 0 的比例 | F：均值/ulp |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for r in s["E_F"]:
        if r["round"] != "main":
            continue
        E, F = r["E"] or {}, r["F"] or {}
        w(f"| {r['round']} | {r['program']} | {r['output']} | {fmt(E.get('relative_rms'))} | {fmt(E.get('max_over_ulp'))} | "
          f"{fmt(F.get('interval_excludes_zero_fraction'))} | {fmt(F.get('mean_mid_over_ulp'))} |")
    w("\nF 的区间排除 0 是样本差异证据，不自动是契约违反或均值非零（协议 §2）。P（性质关系）一律 `not_scored`：\n")
    w("| 程序 | 结构 | 关系 | max | / ulp(尺度) | 说明 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for p in s["P"]:
        if p["round"] != "main":
            continue
        if p["family"] == "T7":
            w(f"| {p['program']} | T7 | 8 次执行间逐位不同的元素比例 {fmt(p.get('fraction_elements_not_bitwise_identical_across_executions'))} | "
              f"spread {fmt(p.get('max_spread_over_ulp'))} ulp | 均值 {fmt(p.get('mean_spread_over_ulp'))} | 次序依赖证据 |")
        elif "relation" in p:
            w(f"| {p['program']} | {p['family']} | {p['relation'][:50]} | {fmt(p.get('max_abs'))} | "
              f"{fmt(p.get('max_abs_over_ulp_of_scale'))} | 原始残差 |")
    rn = s["renaming_relation"]["identical_groups"]
    w(f"\n重命名关系（确定性程序，主轮 96 个单位 K 全部逐位相同的同族程序组）：{[(g['family'], g['programs']) for g in rn] or '无'}。\n")
    w("## 6. 失败类别与成本\n")
    w(f"- 主轮模式 B 各输出的失败类别计数：{s['failure_classes_main_B_outputs']}。")
    tot = sum(x["seconds"] or 0 for x in s["cost"])
    w(f"- 全部作业墙钟秒数之和 {tot:.0f} s；逐作业的分阶段时间在 summary.json `cost`。\n")
    w("## 7. 没有做的事\n")
    w("- 正式盲评分数（等出题方重新封存）；等价判断（未事先声明 δ）；定位（预算 0，未尝试）；边界项 prog_26（未批准隔离运行）；"
      "调用级完整（协议 §1）。")
    (d / "REPORT.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    d, summary, man = build(a.run_id)
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    report_md(d, summary, man)
    print(json.dumps({k: summary[k] for k in ("execution",)}, indent=1)[:2000])
    print(json.dumps(summary["completeness"]["programs_all_outputs_established"]))


if __name__ == "__main__":
    main()
