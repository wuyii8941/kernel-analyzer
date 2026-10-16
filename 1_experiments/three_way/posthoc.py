"""Post-hoc analyses P1-P4 and the fix segments of D6 / D8 (protocol.md, 事后分析; not registered).

    python posthoc.py classes units.jsonl > kernel_classes.json             # from the captured TTIR
    python posthoc.py segments fixes.jsonl units_for_fixes_D6.jsonl > fix_segments.json
    python posthoc.py report units.jsonl kernel_classes.json fixes.jsonl fix_segments.json \
        fixes_recheck.jsonl fixes_D8.jsonl units_prefix_D1.jsonl cost.jsonl > posthoc.json
    python posthoc.py tables posthoc.json > posthoc_tables.md
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize as S  # noqa: E402
from summarize_rest import h5_h6_fixes  # noqa: E402

REPS = 2000
CLASSES = ("matmul", "reduction", "scan", "elementwise")


# -- inputs -------------------------------------------------------------------------------------------------------

def kernel_classes(units: str) -> dict:
    """P1: matmul (tt.dot), scan (tt.scan), reduction (tt.reduce), elementwise (none of them), from the TTIR."""
    out = {}
    for line in open(units):
        u = json.loads(line)["unit"]
        t = (Path(u) / "ir" / "kernel.ttir").read_text()
        out[u] = ("matmul" if "tt.dot" in t else "scan" if "tt.scan" in t else
                  "reduction" if "tt.reduce" in t else "elementwise")
    return out


def fix_segments(fixes: str, d6_units: str) -> dict:
    """The main run's fixes: tf32 ones done before D6 (all of them), tf32 ones under the D6 cap, and the others."""
    d6 = {json.loads(l)["unit"] for l in open(d6_units) if l.strip()}
    seg = {}
    for line in open(fixes):
        f = json.loads(line)
        if (f.get("fix") or {}).get("mechanism") == "tf32 dot":
            seg[f"{f['unit']}|{f['output']}"] = "tf32 under the D6 cap" if f["unit"] in d6 else "tf32 before D6"
        else:
            seg[f"{f['unit']}|{f['output']}"] = "other"
    return seg


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if path and os.path.exists(path) else []


def records(rows: list):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    try:
        S.SINGLES.clear()
        return S.records(f.name)
    finally:
        os.unlink(f.name)


# -- classification with a threshold --------------------------------------------------------------------------------

def cls_z(a, se, z):
    if a is None or se is None or not math.isfinite(a):
        return None
    if se > 0:
        return 1 if a > z * se else -1 if a < -z * se else 0
    return 0 if a == 0 else (1 if a > 0 else -1)


def reclassify(recs, z):
    """Truth and every method classified with the same |a| > z * se rule (z = 3 is the registered one)."""
    out = []
    for r in recs:
        r = dict(r, truth=cls_z(r["a"], r["se"], z))
        for m in S.METHODS:
            if m == "A0" or r[m] is None:
                continue
            p, sp, _ = r[m]
            r[m] = (p, sp, cls_z(p, sp, z))
        out.append(r)
    return out


def z_cutoffs(recs):
    """Bonferroni (alpha 0.05) and Benjamini-Hochberg (q 0.05) cutoffs on |a| / se, two-sided normal p-values."""
    from statistics import NormalDist
    nd = NormalDist()
    recs = [r for r in recs if r["truth"] is not None]
    zs = sorted((abs(r["a"]) / r["se"] if r["se"] > 0 else (math.inf if r["a"] != 0 else 0.0)) for r in recs)
    n = len(zs)
    bonf = nd.inv_cdf(1 - 0.025 / n)
    p = sorted(2 * (1 - nd.cdf(z)) if math.isfinite(z) else 0.0 for z in zs)
    k = max((i + 1 for i in range(n) if p[i] <= 0.05 * (i + 1) / n), default=0)
    bh = nd.inv_cdf(1 - p[k - 1] / 2) if k and p[k - 1] > 0 else bonf
    return {"n_tests": n, "bonferroni_z": bonf, "bh_z": bh, "bh_rejections": k,
            "expected_false_at_3se": 2 * (1 - nd.cdf(3)) * sum(r["truth"] == 0 for r in recs)}


# -- rates with cluster bootstrap -----------------------------------------------------------------------------------

FIELDS = ("sig_n", "sign_agree", "size_agree", "null_n", "null_false_signal")


def cluster_counts(recs, m, key):
    g = collections.defaultdict(lambda: np.zeros(5))
    for r in recs:
        if not S.usable(r, m) or r["truth"] is None:
            continue
        c = g[key(r)]
        if r["truth"] != 0:
            c[0] += 1
            c[1] += r[m][2] == r["truth"]
            c[2] += r[m][2] == r["truth"] and S.agree_size(r, m)
        else:
            c[3] += 1
            c[4] += r[m][2] != 0
    return np.array(list(g.values())) if g else np.zeros((0, 5))


def boot(C, rng, reps=REPS):
    """Replicate totals: resample the clusters (rows of C) with replacement."""
    k = len(C)
    if k == 0:
        return np.zeros((reps, 5))
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=reps)
    return w @ C


def ratio(num, den):
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def summary(C, R):
    """Point rates and percentile intervals for sign, size (over sig_n) and false signals (over null_n)."""
    tot = C.sum(axis=0)
    out = {f: int(tot[i]) for i, f in enumerate(FIELDS)}
    for name, num, den in (("sign", 1, 0), ("size", 2, 0), ("false_signal", 4, 3)):
        rr = ratio(R[:, num], R[:, den])
        rr = rr[~np.isnan(rr)]
        out[name] = [float(tot[num] / tot[den]) if tot[den] else None,
                     *( [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))] if len(rr) else [None, None])]
    return out


def rates_ci(recs, m, key, seed=0):
    C = cluster_counts(recs, m, key)
    return summary(C, boot(C, np.random.default_rng(seed)))


def h1_diff(recs, key, seed=0):
    """C-model sign rate minus A-exec sign rate over outputs where both are usable, clusters resampled."""
    g = collections.defaultdict(lambda: np.zeros(3))
    for r in recs:
        if r["truth"] not in (None, 0) and S.usable(r, "C-model") and S.usable(r, "A-exec"):
            c = g[key(r)]
            c[0] += 1
            c[1] += r["C-model"][2] == r["truth"]
            c[2] += r["A-exec"][2] == r["truth"]
    C = np.array(list(g.values()))
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(C), np.full(len(C), 1.0 / len(C)), size=REPS)
    T = w @ C
    d = (T[:, 1] - T[:, 2]) / T[:, 0]
    tot = C.sum(axis=0)
    return {"outputs": int(tot[0]), "clusters": len(C), "diff": float((tot[1] - tot[2]) / tot[0]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def by_class(recs, cls_of, seed=0):
    """P1: per-class rates (units resampled within the class) and the equal-weight average over classes."""
    out, reps = {}, {}
    rng = np.random.default_rng(seed)
    groups = {"matmul": [r for r in recs if cls_of[r["unit"]] == "matmul"],
              "reduction": [r for r in recs if cls_of[r["unit"]] == "reduction"],
              "scan": [r for r in recs if cls_of[r["unit"]] == "scan"],
              "elementwise": [r for r in recs if cls_of[r["unit"]] == "elementwise"],
              "elementwise (mamba)": [r for r in recs if cls_of[r["unit"]] == "elementwise" and "mamba" in r["model"]],
              "elementwise (other models)": [r for r in recs if cls_of[r["unit"]] == "elementwise"
                                             and "mamba" not in r["model"]],
              "all but matmul": [r for r in recs if cls_of[r["unit"]] != "matmul"]}
    for name, rs in groups.items():
        out[name] = {"outputs": len(rs), "configs": dict(collections.Counter(r["config"] for r in rs))}
        for m in ("A-exec", "C-model", "C-exec"):
            C = cluster_counts(rs, m, lambda r: r["unit"])
            R = boot(C, rng)
            out[name][m] = summary(C, R)
            if name in CLASSES:
                reps[(name, m)] = (C.sum(axis=0), R)
    macro = {}
    for m in ("A-exec", "C-model", "C-exec"):
        present = [c for c in CLASSES if reps[(c, m)][0][0] > 0]
        point = np.mean([reps[(c, m)][0][1] / reps[(c, m)][0][0] for c in present])
        rr = np.nanmean(np.stack([ratio(reps[(c, m)][1][:, 1], reps[(c, m)][1][:, 0]) for c in present]), axis=0)
        macro[m] = {"classes": present, "sign": [float(point), float(np.percentile(rr, 2.5)),
                                                  float(np.percentile(rr, 97.5))]}
    out["equal-weight average over classes"] = macro
    return out


def main_table(recs, seed=0):
    t = {}
    for m in ("A0", "A-exec", "C-model", "C-exec"):
        t[m] = {"by_unit": rates_ci(recs, m, lambda r: r["unit"], seed),
                "by_model": rates_ci(recs, m, lambda r: r["model"], seed)}
    return t


def p5(recs, cls_of):
    """P5: the null group split into exactly zero (K = G element by element) and not significant; abstentions."""
    exact = [r for r in recs if r["truth"] == 0 and r["a"] == 0 and r["se"] == 0]
    rest = [r for r in recs if r["truth"] == 0 and not (r["a"] == 0 and r["se"] == 0)]
    out = {m: {"exact_zero": S.rates(exact, m), "not_significant": S.rates(rest, m)} for m in ("A-exec", "C-model")}
    for m in ("A-exec", "C-model"):
        reported = [r for r in rest if S.usable(r, m) and r[m][2] != 0]
        out[m]["reports_contradicted"] = sum(abs(r[m][0]) > 3 * r["se"] for r in reported)
        out[m]["reports_not_resolvable"] = len(reported) - out[m]["reports_contradicted"]
    ab = {}
    for name, rs in (("all", recs), ("matmul", [r for r in recs if cls_of[r["unit"]] == "matmul"]),
                     ("all but matmul", [r for r in recs if cls_of[r["unit"]] != "matmul"])):
        sig = [r for r in rs if r["truth"] not in (None, 0)]
        null = [r for r in rs if r["truth"] == 0]
        ab[name] = {"sig_n": len(sig), "sig_abstained": sum(not S.usable(r, "C-model") for r in sig),
                    "sign_agree": sum(S.usable(r, "C-model") and r["C-model"][2] == r["truth"] for r in sig),
                    "null_n": len(null), "null_abstained": sum(not S.usable(r, "C-model") for r in null)}
    out["C-model abstentions"] = ab
    return out


# -- fixes ------------------------------------------------------------------------------------------------------------

def fixes_report(fixes, segments, recheck, d8):
    re = {(r["unit"], r["output"]): r for r in recheck}
    compare = []
    rows = []
    for f in fixes:
        k = (f["unit"], f["output"])
        if k in re:
            old = h5_h6_fixes([f])["fixes"]
            new = h5_h6_fixes([re[k]])["fixes"]
            compare.append({"unit": f["unit"], "output": f["output"],
                            "original": old[0] if old else None, "recheck": new[0] if new else None})
            f = dict(re[k], segment_note="rechecked")
        rows.append(dict(f, segment=segments.get(f"{k[0]}|{k[1]}", "other")))
    seen = set()
    for f in d8:
        rank = f.get("D8_rank", -1)
        if "skipped" in f or rank in seen:      # a rank claimed by the other driver, or written twice
            continue
        seen.add(rank)
        rows.append(dict(f, segment="D8 random sample (first 60)" if 0 <= rank < 60 else
                         "D8 rest" if (f.get("fix") or {}).get("mechanism") == "tf32 dot" or rank < 395 else "D8 mamba"))

    def stats(rs):
        s = h5_h6_fixes(rs)
        e = s["not_registered_effect_within_10pct"]
        status = collections.Counter(r.get("status") or ("error" if r.get("error") else "?") for r in rs)
        return {"rows": len(rs), "status": dict(status), "fixed": s["n"], "localized": s["H5_localized"][0],
                "C_after_agrees": s["H6_C_after_agrees"][0], "A_after_agrees": s["A_after_agrees"][0],
                "effect_C": e["C"], "effect_A": e["A"],
                "by_mechanism": dict(collections.Counter(r["mechanism"] for r in s["fixes"]))}
    segs = collections.defaultdict(list)
    for r in rows:
        segs[r["segment"]].append(r)
    main = [r for r in rows if not r["segment"].startswith("D8")]
    return {"segments": {k: stats(v) for k, v in sorted(segs.items())},
            "main_run_with_recheck": stats(main), "main_run_and_D8": stats(rows),
            "tf32_all": stats([r for r in rows if (r.get("fix") or {}).get("mechanism") == "tf32 dot"]),
            "recheck": compare}


# -- D1 --------------------------------------------------------------------------------------------------------------

def d1_report(units_rows, prefix_rows):
    pre = {r["unit"]: r for r in prefix_rows}
    changed_verify, changed_values, errors, by_method = [], [], [], collections.Counter()
    for r in units_rows:
        p = pre.get(r["unit"])
        if p is None:
            continue
        if "error" in p:
            errors.append(r["unit"])
            continue
        if p.get("verify") != r.get("verify"):
            changed_verify.append({"unit": r["unit"], "before_fix": p.get("verify"), "after_fix": r.get("verify")})
        diff = set()
        for name, o in r.get("outputs", {}).items():
            q = p.get("outputs", {}).get(name, {})
            for m in ["truth", "n"] + list(S.METHODS[:2]) + [f"A-{x}" for x in S.PREMISES] + ["A-exec"]:
                if o.get(m) != q.get(m):
                    diff.add(m)
        if diff:
            changed_values.append(r["unit"])
            by_method.update(diff)
    merged = [pre.get(r["unit"], r) if "error" not in pre.get(r["unit"], {}) else r for r in units_rows]
    recs, units = records(merged)
    primary = [r for r in recs if r["bitwise"]]
    return {"rerun_units": len(pre), "rerun_errors": errors, "verify_changed": changed_verify,
            "units_with_changed_values": len(changed_values), "changed_value_units": changed_values,
            "changed_units_by_quantity": dict(by_method),
            "verify_before_fix": dict(collections.Counter(u["verify"] for u in units)),
            "primary_outputs_before_fix": len(primary),
            "primary_before_fix": {m: S.rates(primary, m) for m in ("A0", "A-exec", "C-model", "C-exec")},
            "H1_before_fix": S.bootstrap_h1(primary), "H3_before_fix": S.premise_attribution(primary)}


def cost_report(rows):
    """P6: medians and quartiles of each component's time, the number of rounding sites (= emulations for attribution)."""
    ok = [r for r in rows if "error" not in r]
    out = {"units": len(rows), "errors": len(rows) - len(ok)}
    for k in ("G_s", "C_exec_s", "C_model_s", "device_first_launch_s", "device_launch_s", "rounding_sites"):
        v = np.array([r[k] for r in ok], dtype=float)
        out[k] = [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))] if len(v) else None
    ratio = np.array([r["C_model_s"] / r["C_exec_s"] for r in ok if r["C_exec_s"] > 0])
    out["C_model_over_C_exec"] = [float(np.percentile(ratio, 25)), float(np.median(ratio)), float(np.percentile(ratio, 75))]
    out["model_faster_than_exec"] = int((ratio < 1).sum())
    out["programs_fraction"] = float(np.median([r["programs"] / r["programs_total"] for r in ok]))
    out["prediction_status"] = dict(collections.Counter(r["prediction_status"] for r in ok))
    return out


def report(units, classes, fixes, segments, recheck, d8, prefix, cost=None):
    cls_of = json.load(open(classes))
    rows = load(units)
    recs, _ = records(rows)
    primary = [r for r in recs if r["bitwise"]]
    assert all(r["config"] == "fp32tf32" for r in primary if cls_of[r["unit"]] == "matmul")
    cut = z_cutoffs(primary)
    out = {"main": main_table(primary),
           "H1_diff": {"by_unit": h1_diff(primary, lambda r: r["unit"]),
                       "by_model": h1_diff(primary, lambda r: r["model"])},
           "classes": by_class(primary, cls_of),
           "class_units": dict(collections.Counter(cls_of.values())),
           "H2_false_signals": {
               "by_class": {c: S.rates([r for r in primary if cls_of[r["unit"]] == c], "C-model") for c in CLASSES},
               "mamba": S.rates([r for r in primary if "mamba" in r["model"]], "C-model"),
               "without_mamba": S.rates([r for r in primary if "mamba" not in r["model"]], "C-model")},
           "significance": cut,
           "P5": p5(primary, cls_of),
           "sensitivity": {name: {m: S.rates(reclassify(primary, z), m) for m in ("A0", "A-exec", "C-model", "C-exec")}
                           for name, z in (("3se (registered)", 3.0), ("Benjamini-Hochberg", cut["bh_z"]),
                                           ("Bonferroni", cut["bonferroni_z"]))},
           "fixes": fixes_report(load(fixes), json.load(open(segments)), load(recheck), load(d8))}
    from fixes_D8 import order
    done = {f["D8_rank"] for f in load(d8) if "skipped" not in f}
    out["fixes"]["D8_not_done"] = [{"rank": i, "unit": u, "output": o}
                                   for i, (u, o) in enumerate(order(units, fixes)) if i not in done]
    # D7: the units the first directory glob missed (index >= 100); the registered rates without them
    early = [r for r in primary if int(Path(r["unit"]).name.split("_", 1)[0]) < 100]
    out["D7"] = {"missed_units": sum(int(Path(r["unit"]).name.split("_", 1)[0]) >= 100 for r in rows),
                 "without_missed": {m: S.rates(early, m) for m in ("A-exec", "C-model")}}
    if prefix and os.path.exists(prefix):
        out["D1"] = d1_report(rows, load(prefix))
    if cost and os.path.exists(cost):
        out["P6"] = cost_report(load(cost))
    return out


# -- markdown ----------------------------------------------------------------------------------------------------------

def _p(k, n, ci=None):
    if not n:
        return "—"
    t = f"{k}/{n}（{100 * k / n:.0f}%"
    if ci and ci[1] is not None:
        t += f"，{100 * ci[1]:.0f}–{100 * ci[2]:.0f}"
    return t + "）"


CLASS_NAMES = {"matmul": "矩阵乘（全部是 tf32 模板）", "reduction": "归约", "scan": "扫描（`tt.scan`）",
               "elementwise": "逐元素", "elementwise (mamba)": "　其中 mamba", "elementwise (other models)": "　其中其余模型",
               "all but matmul": "矩阵乘以外合计"}
METHOD = {"A0": "A0 概率模型", "A-exec": "A 替代执行假设下的重放", "C-model": "C 事前预测器", "C-exec": "C 仿真器（真实执行假设）"}


def tables(path):
    r = json.load(open(path))
    out = ["### 主表与聚类区间（P2）", "", "括号内是 95% 区间：按单元（一次 kernel 启动）重抽 / 按模型重抽。", "",
           "| 方法 | 有显著偏差：符号对 | 按模型 | 符号与大小都对 | 未显著：报出显著 | 按模型 |", "|---|---|---|---|---|---|"]
    for m in ("A0", "A-exec", "C-model", "C-exec"):
        u, g = r["main"][m]["by_unit"], r["main"][m]["by_model"]
        out.append(f"| {METHOD[m]} | {_p(u['sign_agree'], u['sig_n'], u['sign'])} | {100 * g['sign'][1]:.0f}–{100 * g['sign'][2]:.0f} | "
                   f"{_p(u['size_agree'], u['sig_n'], u['size'])} | {_p(u['null_false_signal'], u['null_n'], u['false_signal'])} | "
                   f"{100 * g['false_signal'][1]:.0f}–{100 * g['false_signal'][2]:.0f} |")
    h = r["H1_diff"]
    out += ["", f"H1 的差（C-model − A-exec，{h['by_unit']['outputs']} 个输出）：{100 * h['by_unit']['diff']:.0f} 个百分点，"
            f"按单元 {100 * h['by_unit']['ci95'][0]:.0f}–{100 * h['by_unit']['ci95'][1]:.0f}，"
            f"按模型 {100 * h['by_model']['ci95'][0]:.0f}–{100 * h['by_model']['ci95'][1]:.0f}。"]
    out += ["", "### 按 kernel 结构分层（P1）", "",
            "| 类别 | 输出 | A 符号对 | C 预测器符号对 | C 预测器：未显著上报出 | A：未显著上报出 |", "|---|---|---|---|---|---|"]
    for k, v in r["classes"].items():
        if k.startswith("equal"):
            continue
        a, c = v["A-exec"], v["C-model"]
        out.append(f"| {CLASS_NAMES[k]} | {v['outputs']} | {_p(a['sign_agree'], a['sig_n'], a['sign'])} | "
                   f"{_p(c['sign_agree'], c['sig_n'], c['sign'])} | {_p(c['null_false_signal'], c['null_n'], c['false_signal'])} | "
                   f"{_p(a['null_false_signal'], a['null_n'], a['false_signal'])} |")
    eq = r["classes"]["equal-weight average over classes"]
    out.append(f"| 各类等权平均（{'、'.join(CLASS_NAMES[c].split('（')[0] for c in eq['A-exec']['classes'])}） | | "
               f"{100 * eq['A-exec']['sign'][0]:.0f}%（{100 * eq['A-exec']['sign'][1]:.0f}–{100 * eq['A-exec']['sign'][2]:.0f}） | "
               f"{100 * eq['C-model']['sign'][0]:.0f}%（{100 * eq['C-model']['sign'][1]:.0f}–{100 * eq['C-model']['sign'][2]:.0f}） | | |")
    sig = r["significance"]
    out += ["", "### 显著性判定的敏感性（P3）", "",
            f"{sig['n_tests']} 个输出各自检验。按登记的 3·se，真值无偏差时期望约 {sig['expected_false_at_3se']:.1f} 个输出被误判为显著。"
            f"Bonferroni 的门槛是 |a|/se > {sig['bonferroni_z']:.2f}；Benjamini–Hochberg 的门槛是 {sig['bh_z']:.2f}"
            "（真信号多，BH 比 3·se 宽）。真值与各方法用同一门槛重新分类：", "",
            "| 门槛 | 显著输出 | A 符号对 | C 预测器符号对 | C 预测器：未显著上报出 |", "|---|---|---|---|---|"]
    labels = {"3se (registered)": "3·se（登记）", "Benjamini-Hochberg": "Benjamini–Hochberg（q = 0.05）",
              "Bonferroni": "Bonferroni（α = 0.05）"}
    for name, v in r["sensitivity"].items():
        out.append(f"| {labels[name]} | {v['A-exec']['sig_n']} | {_p(v['A-exec']['sign_agree'], v['A-exec']['sig_n'])} | "
                   f"{_p(v['C-model']['sign_agree'], v['C-model']['sig_n'])} | "
                   f"{_p(v['C-model']['null_false_signal'], v['C-model']['null_n'])} |")
    q = r["P5"]
    out += ["", "### 「无显著偏差」的两部分与预测器弃权（P5）", "",
            "| 方法 | 精确为零（K = G 逐元素）：报出显著 | 其余未显著：报出显著 | 其中与测量矛盾（预测的 |a| > 真值的 3·se） | 其中测量分辨不了 |",
            "|---|---|---|---|---|"]
    for m in ("A-exec", "C-model"):
        e, n = q[m]["exact_zero"], q[m]["not_significant"]
        out.append(f"| {METHOD[m]} | {_p(e['null_false_signal'], e['null_n'])} | {_p(n['null_false_signal'], n['null_n'])} | "
                   f"{q[m]['reports_contradicted']} | {q[m]['reports_not_resolvable']} |")
    out += ["", "| 子集 | 有显著偏差 | 预测器弃权 | 符号对（弃权不计） | 符号对（弃权算错） | 未显著输出中弃权 |", "|---|---|---|---|---|---|"]
    for k, v in q["C-model abstentions"].items():
        given = v["sig_n"] - v["sig_abstained"]
        out.append(f"| {CLASS_NAMES.get(k, '全部')} | {v['sig_n']} | {v['sig_abstained']} | {_p(v['sign_agree'], given)} | "
                   f"{_p(v['sign_agree'], v['sig_n'])} | {v['null_abstained']}/{v['null_n']} |")
    fx = r["fixes"]
    out += ["", "### 改法验证分段（D6、D8）", "",
            "| 段 | 改了并重放 | 偏差减半 | C 改后值一致 | 效果：C 对 | 效果：A 对 |", "|---|---|---|---|---|---|"]
    names = {"tf32 before D6": "tf32，D6 之前（全量）", "tf32 under the D6 cap": "tf32，D6 限量（每组前 4 个）",
             "D8 random sample (first 60)": "tf32，D8 随机补抽前 60 个", "D8 rest": "tf32，D8 其余",
             "D8 mamba": "mamba 未完成的输出（D8）", "other": "近似除法与 rsqrt"}
    for k in ("tf32 before D6", "tf32 under the D6 cap", "D8 random sample (first 60)", "D8 rest", "other", "D8 mamba"):
        v = fx["segments"].get(k)
        if not v:
            continue
        n = v["fixed"]
        out.append(f"| {names[k]} | {n} | {_p(v['localized'], n)} | {_p(v['C_after_agrees'], n)} | {_p(v['effect_C'], n)} | "
                   f"{_p(v['effect_A'], n)} |")
    for k, label in (("main_run_with_recheck", "主运行合计"), ("main_run_and_D8", "加上 D8")):
        v = fx[k]
        n = v["fixed"]
        out.append(f"| **{label}** | {n} | {_p(v['localized'], n)} | {_p(v['C_after_agrees'], n)} | {_p(v['effect_C'], n)} | "
                   f"{_p(v['effect_A'], n)} |")
    if "D1" in r:
        d = r["D1"]
        out += ["", "### D1：仿真器修补前后（P4）", "",
                f"用修补前的仿真器重跑 {d['rerun_units']} 个含归约的单元（出错 {len(d['rerun_errors'])} 个）：逐位核对状态变化的 "
                f"{len(d['verify_changed'])} 个，任一方法的数值变化的 {d['units_with_changed_values']} 个。", "",
                "| 方法 | 修补前：符号对 | 修补前：未显著上报出 | 修补后：符号对 | 修补后：未显著上报出 |", "|---|---|---|---|---|"]
        for m in ("A-exec", "C-model", "C-exec"):
            b, a = d["primary_before_fix"][m], r["main"][m]["by_unit"]
            out.append(f"| {METHOD[m]} | {_p(b['sign_agree'], b['sig_n'])} | {_p(b['null_false_signal'], b['null_n'])} | "
                       f"{_p(a['sign_agree'], a['sig_n'])} | {_p(a['null_false_signal'], a['null_n'])} |")
    if "P6" in r:
        c = r["P6"]
        f = lambda v, k=1: f"{v[1]:.{k}f}（{v[0]:.{k}f}–{v[2]:.{k}f}）"
        out += ["", f"### 成本（P6，{c['units']} 个单元，出错 {c['errors']} 个）", "",
                "中位数（四分位数）。G、C-exec、C-model 只算抽中的 program（中位数占全部 program 的 "
                f"{100 * c['programs_fraction']:.0f}%）；设备上的启动算全部 program。", "",
                "| 项目 | 秒 |", "|---|---|",
                f"| G 的求值 | {f(c['G_s'])} |", f"| C-exec 仿真 | {f(c['C_exec_s'])} |", f"| C-model 预测 | {f(c['C_model_s'])} |",
                f"| 设备：重新编译并启动 | {f(c['device_first_launch_s'])} |",
                f"| 设备：编译之后单次启动 | {f(c['device_launch_s'], 5)} |",
                f"| 舍入位点数（归因要做的仿真次数） | {f(c['rounding_sites'], 0)} |",
                "", f"C-model 与 C-exec 的耗时比：{f(c['C_model_over_C_exec'], 2)}；C-model 更快的单元 "
                f"{c['model_faster_than_exec']}/{c['units'] - c['errors']}。"]
    d7 = r["D7"]
    out += ["", f"### D7：补上漏掉的 {d7['missed_units']} 个单元之前与之后", "",
            "| 方法 | 补之前：符号对 | 补之前：未显著上报出 | 补之后：符号对 | 补之后：未显著上报出 |", "|---|---|---|---|---|"]
    for m in ("A-exec", "C-model"):
        b, a = d7["without_missed"][m], r["main"][m]["by_unit"]
        out.append(f"| {METHOD[m]} | {_p(b['sign_agree'], b['sig_n'])} | {_p(b['null_false_signal'], b['null_n'])} | "
                   f"{_p(a['sign_agree'], a['sig_n'])} | {_p(a['null_false_signal'], a['null_n'])} |")
    print("\n".join(out))


def main():
    cmd = sys.argv[1]
    if cmd == "classes":
        print(json.dumps(kernel_classes(sys.argv[2]), indent=0))
    elif cmd == "segments":
        print(json.dumps(fix_segments(sys.argv[2], sys.argv[3]), indent=0))
    elif cmd == "report":
        print(json.dumps(report(*sys.argv[2:10]), indent=1, default=str))
    elif cmd == "tables":
        tables(sys.argv[2])


if __name__ == "__main__":
    main()
