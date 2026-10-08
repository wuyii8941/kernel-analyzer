#!/usr/bin/env python3
"""Section 3 family table (four numbers per family: complete reference rate, width over the output dtype's ulp,
fraction resolving the 1/8-ulp target, phase timings) for tool 3.0 against the tool-2.3 accumulation (gamma), and the
section 6 failure table (every not-established output or failed case in five classes).

Sources: results/general/fr_modeB_v3_1 and fr_modeB_gamma (tier-1 Triton candidates, mode B), results/general/
reference_quality_probe_{exact,gamma}.json (numerical-stream programs, G6 compositions, new operators),
results/general/demo/*.json (unified entry), results/general/calibration_five_distributions.json (statistics).
Writes results/general/quality_table.json, results/general/failure_table.json and the two markdown tables in
docs/general/.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.measure import FAILURE_CLASSES, classify_failure  # noqa: E402

G = ROOT / "results/general"


def merge_quality(qs):
    """element-weighted aggregation of reference_quality records (quantiles are reported as the weighted mean of the
    per-output quantiles and the overall max)."""
    n_all = sum(q["elements"] for q in qs)
    n_ok = sum(q["complete_finite"] for q in qs)
    res = sum((q["resolved_fraction"] or 0) * q["complete_finite"] for q in qs)
    med = [q["width_over_ulp"]["median"] for q in qs if q["width_over_ulp"]["median"] is not None]
    p90 = [q["width_over_ulp"]["p90"] for q in qs if q["width_over_ulp"]["p90"] is not None]
    mx = [q["width_over_ulp"]["max"] for q in qs if q["width_over_ulp"]["max"] is not None]
    return {"outputs": len(qs), "elements": n_all, "complete_rate": n_ok / n_all if n_all else None,
            "width_over_ulp_median_of_outputs": float(np.median(med)) if med else None,
            "width_over_ulp_p90_of_outputs": float(np.median(p90)) if p90 else None,
            "width_over_ulp_max": float(max(mx)) if mx else None,
            "resolved_fraction": res / n_ok if n_ok else None}


def fr_family(path):
    d = json.loads(Path(path).read_text())
    qs, timing = [], Counter()
    for r in d["cases"].values():
        if r.get("status") != "ok":
            continue
        for o in r["outputs"].values():
            if o.get("reference_quality"):
                qs.append(o["reference_quality"])
        for k, v in (r.get("timing") or {}).items():
            timing[k] += v or 0
    return qs, timing


def probe_groups(path):
    d = json.loads(Path(path).read_text())
    groups = defaultdict(lambda: ([], Counter()))
    for r in d["programs"].values():
        qs, tm = groups[r["group"]]
        qs.extend((r.get("outputs") or {}).values())
        for k, v in (r.get("timing_seconds") or {}).items():
            tm[k] += v or 0
    return groups


def phases(t):
    return {"capture": round(t.get("capture_first_seed", 0) + t.get("capture", 0) + t.get("setup_compile_warmup", 0), 1),
            "reference": round(t.get("reference", 0), 1), "statistics": round(t.get("statistics", 0) + t.get("specification", 0), 1)}


def quality_table():
    rows = []
    for mode, frdir, probe in (("3.1 (SumK / DotK, p_n last)", "fr_modeB_v3_1", "reference_quality_probe_exact.json"),
                               ("2.3 accumulation (gamma)", "fr_modeB_gamma", "reference_quality_probe_gamma.json")):
        for p in sorted(glob.glob(str(G / frdir / "*.json"))):
            qs, tm = fr_family(p)
            rows.append({"mode": mode, "family": Path(p).stem, "source": "tier-1 mode B (3 units)", **merge_quality(qs),
                         "seconds": phases(tm)})
        for grp, (qs, tm) in probe_groups(G / probe).items():
            rows.append({"mode": mode, "family": grp, "source": "quality probe (3 units)", **merge_quality(qs),
                         "seconds": phases(tm)})
    for p in sorted(glob.glob(str(G / "demo/*.json"))):
        d = json.loads(Path(p).read_text())
        for lv in d.get("levels", []):
            qs = [o["reference"] for o in (lv.get("outputs") or {}).values() if o.get("reference")]
            if qs:
                call = [q.get("complete_rate_call_level") for q in qs]
                rows.append({"mode": "3.1 (SumK / DotK, p_n last)", "family": f"unified entry: {Path(p).stem} {lv['level'] or ''}",
                             "source": "unified entry (96 units)", **merge_quality(qs),
                             "complete_rate_call_level": (sum(call) / len(call)) if all(c is not None for c in call) else None,
                             "seconds": phases(lv.get("timing_seconds") or {})})
    return rows


def failure_table():
    items = []
    for p in sorted(glob.glob(str(G / "fr_modeB_v3_1/*.json"))):
        fam = Path(p).stem
        for key, r in json.loads(Path(p).read_text())["cases"].items():
            if r.get("status") != "ok":
                why = r.get("reason", r.get("status"))
                cls = ("not counted (no f / clause pending: outside the measurement)" if r.get("status") in
                       ("no f", "clause pending", "not run") else
                       classify_failure(("call rejected the declared inputs: " + str(why)) if "Dynamo failed" in str(why)
                                        else why))
                items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": None, "reason": str(why)[:160], "class": cls})
                continue
            for o in r["notes"].get("outputs_not_written_by_triton") or []:
                items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": o, "reason": "not written by Triton",
                              "class": "binding"})
            for o, rs in (r["notes"].get("outputs_whose_writing_programs_aborted") or {}).items():
                items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": o, "reason": rs[0][:160],
                              "class": classify_failure(rs[0])})
            for o, v in r["outputs"].items():
                q = v.get("reference_quality") or {}
                if q.get("complete_rate") is not None and q["complete_rate"] < 1:
                    sv = v.get("special_values") or {}
                    missing = q["elements"] - q["complete_finite"]
                    # the missing elements are exactly those where f is special or undefined (NaN rows of the spec)
                    special = (int(sv.get("elements_with_special_f") or 0) > 0 and not int(sv.get("kr_vs_f_class_mismatch") or 0)) \
                        or (missing > 0 and missing == int(v.get("f_undefined_elements") or -1))
                    items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": o,
                                  "reason": ("missing elements are special values (NaN / inf: target undefined or infinite, "
                                             "contract class C)" if special else "incomplete reference"),
                                  "class": ("not counted (special values: target undefined or infinite)" if special
                                            else "enclosure too wide")})
                if q.get("resolved_fraction") is not None and q["resolved_fraction"] < 1:
                    items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": o,
                                  "reason": f"resolved fraction {q['resolved_fraction']:.4f} < 1", "class": "enclosure too wide"})
                if v.get("mixed_non_triton_sources"):
                    items.append({"source": f"fr_modeB_v3_1/{fam}", "case": key, "output": o,
                                  "reason": "reads a non-Triton intermediate: no semantic verdict (numerical only)",
                                  "class": "binding (semantic verdict only)"})
    for name, r in json.loads((G / "reference_quality_probe_exact.json").read_text())["programs"].items():
        for o in r.get("not_written_by_triton") or []:
            items.append({"source": f"probe/{r['group']}", "case": name, "output": o, "reason": "not written by Triton",
                          "class": "binding"})
        for o, rs in (r.get("aborted") or {}).items():
            items.append({"source": f"probe/{r['group']}", "case": name, "output": o, "reason": rs[0][:160],
                          "class": classify_failure(rs[0])})
        if r.get("status") == "error":
            items.append({"source": f"probe/{r['group']}", "case": name, "output": None, "reason": r["reason"][:160],
                          "class": classify_failure(r["reason"])})
    for p in sorted(glob.glob(str(G / "demo/*.json"))):
        d = json.loads(Path(p).read_text())
        for lv in d.get("levels", []):
            if lv.get("status") != "ok":
                items.append({"source": f"demo/{Path(p).stem}", "case": str(lv["level"]), "output": None,
                              "reason": str(lv.get("reason"))[:160], "class": lv.get("failure_class")})
            for o, v in (lv.get("outputs") or {}).items():
                for c in v.get("failure_classes") or []:
                    items.append({"source": f"demo/{Path(p).stem}", "case": str(lv["level"]), "output": o, "reason": c, "class": c})
    cal = json.loads((G / "calibration_five_distributions.json").read_text())
    for r in cal["rows"]:
        if r["cannot_judge_rate"] > 0 and r["rule_or_class"] in ("R1", "R2", "R3", "R5"):
            items.append({"source": "calibration", "case": f"{r['distribution']} effect {r['effect']}", "output": r["rule_or_class"],
                          "reason": f"cannot judge in {r['cannot_judge_rate']:.3f} of {r['replicates']} replicates",
                          "class": "statistics insufficient"})
    counts = Counter(i["class"] for i in items)
    return items, counts


def main():
    rows = quality_table()
    (G / "quality_table.json").write_text(json.dumps(rows, indent=1) + "\n")
    items, counts = failure_table()
    (G / "failure_table.json").write_text(json.dumps({"counts": counts, "items": items}, indent=1) + "\n")
    L = ["# 第 3 项：参照精度与成本的家族表（工具 3.1 对 2.3 累加方式）", "",
         "每行四个数：完整参照率；宽度 / 输出 dtype 在 |G| 处的 ulp（各输出中位数的中位数、各输出 90% 分位的中位数、最大值）；达到分辨"
         "目标（≤ 1/8 ulp）的元素比例；分段耗时（秒：捕获含编译预热 / 参照 / 统计含规格）。「2.3 累加方式」用同一代码、"
         "`KA_ACCUMULATION=gamma` 重跑，只换回 γₙ 求和与点积界。", "",
         "调用级完整率只对统一入口报告：上游非 Triton 值不是声明输入的复制时为 0（kernel 级参照）。", "",
         "| 方式 | 家族 | 来源 | 输出 | 完整率 | 调用级完整率 | 宽度/ulp 中位 | 90% | 最大 | 达到 1/8 ulp | 捕获 / 参照 / 统计 s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))  # noqa: E731
    for r in rows:
        s = r["seconds"]
        L.append(f"| {r['mode']} | {r['family']} | {r['source']} | {r['outputs']} | {fmt(r['complete_rate'])} | "
                 f"{fmt(r.get('complete_rate_call_level'))} | "
                 f"{fmt(r['width_over_ulp_median_of_outputs'])} | {fmt(r['width_over_ulp_p90_of_outputs'])} | "
                 f"{fmt(r['width_over_ulp_max'])} | {fmt(r['resolved_fraction'])} | {s['capture']} / {s['reference']} / {s['statistics']} |")
    (ROOT / "docs/general/quality_table.md").write_text("\n".join(L) + "\n")
    F = ["# 第 6 项：失败分类表（通用能力轮的全部运行）", "",
         f"五类计数：{dict(counts)}", "",
         "分类规则：`kernel_analyzer.measure.classify_failure`（原因字符串 → 类别，映射表随每份报告输出）。「binding (semantic verdict "
         "only)」是读到非 Triton 中间值的输出：数值差异照常测量，只是不出语义结论；「not counted」是规格不覆盖或条款待审阅的条件，"
         "不属于工具失败。", "", "| 来源 | 用例 | 输出 | 原因 | 类别 |", "|---|---|---|---|---|"]
    for i in items:
        F.append(f"| {i['source']} | {i['case']} | {i['output'] or '—'} | {i['reason']} | {i['class']} |")
    (ROOT / "docs/general/failure_table.md").write_text("\n".join(F) + "\n")
    print("quality rows", len(rows), "failure items", len(items), dict(counts))


if __name__ == "__main__":
    main()
