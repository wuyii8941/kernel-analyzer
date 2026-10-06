#!/usr/bin/env python3
"""Tables for the external controlled corpus (docs/external_eval_protocol_20261006.md): RQ1 (reference reliability and
cost), RQ2 (specification differences, tool against B1-B3 with defaults and the pre-registered sweeps), RQ3 (rules on
e_num and on the total K - f, information-difference counts), ablations A1 and A2, black-box entries.

    python scripts/external_corpus_summary.py      # -> results/external/gpuemu/summary.json, summary.md
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_spec_summary import sem_bin  # noqa: E402

OUT = ROOT / "results/external/gpuemu"
TIERS = {1: (range(0, 2), range(2, 8)), 2: (range(0, 32), range(32, 96))}
B1_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
B2_GRID = [1e-1, 1.0, 10.0, 1e2, 1e3]
B3_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 64.0]
DEFAULTS = {"B1": 1.0, "B2": 1.0, "B3": 2.0}
TOLS = {}  # entry -> float32 tolerance of meta.json (from the manifest)


def label(entry, dims):
    """Protocol section 4: True where the planted change makes the declared semantics differ from f."""
    if not entry.endswith("buggy"):
        return False
    if entry == "attention_triton_buggy":
        return dims["N"] > 1
    if entry == "flash_attention_triton_buggy":
        return dims["N"] > max(8, 1 << (min(dims["N"], 32) - 1).bit_length())
    if entry == "matmul_triton_buggy":
        return dims["K"] > 1
    if entry == "softmax_triton_buggy":
        return dims["H"] & (dims["H"] - 1) != 0
    if entry == "softmax_llm_buggy":
        return dims["H"] % 128 != 0
    return True


def control_of(entry):
    if entry == "softmax_llm_buggy":
        return "softmax"
    return entry[: -len("_buggy")] if entry.endswith("_buggy") else entry


def load_reports(sub):
    out = {}
    for p in sorted(glob.glob(str(OUT / sub / "*.json"))):
        r = json.loads(Path(p).read_text())
        out[Path(p).stem] = r
    return out


def load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(str(OUT / pattern))):
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return rows


def detected(rec):
    """Any rule detects, with the skew policy of the statistics calibration (section 4): a suspect t approximation is
    replaced by the bootstrap-t companion; |skew| >= 2 with fewer than 64 confirmation units is unresolved."""
    if not rec or "rules" not in rec:
        return None
    hits = []
    for r in rec["rules"]:
        v = str(r.get("verdict", ""))
        if str(r.get("t_approximation", "ok")).startswith("suspect"):
            if abs(r.get("unit_skewness") or 0) >= 2 and (r.get("n") or 0) < 64:
                v = "UNRESOLVED_SKEW"
            elif r.get("robust"):
                v = str(r["robust"].get("verdict", v))
        hits.append((r["rule"], v))
    return hits


def any_detected(rec):
    h = detected(rec)
    return None if h is None else any(v.startswith("DETECTED") for _, v in h)


def tool_rq2(report):
    """Three outcomes (protocol section 4) and the e_sem bin."""
    if "error" in report:
        return "not established", "error"
    o = report["outputs"].get("out")
    if o is None:
        return "not established", "no output"
    rc = o["reference_classes"]
    if rc["complete_fraction"] < 1.0 or o.get("aborted_programs_seed0"):
        return "not established", "incomplete reference"
    b = sem_bin(o.get("semantic"), total=(o.get("total") or {}).get("relative_rms"),
                elementwise=o.get("semantic_elementwise"))
    if b == "candidate":
        return "inconsistent", b
    if b in ("none", "constant-level", "small"):
        return "equality not ruled out", b
    if b == "unresolved":
        return "not established", b
    return "other bin", b


def baseline_flags(row, method, param, seeds):
    key = {"B1": "b1_max_abs_vs_rounded_f", "B2": "b2_max_ratio", "B3": "b3_ratio"}[method]
    thr = param * row["tolerance"] if method == "B1" else param
    return any(p[key] > thr for p in row["per_seed"] if p["seed"] in seeds)


def working_points(rows, tier):
    """Per control: the strictest grid value with no flag on that control's development seeds (all conditions)."""
    dev = set(TIERS[tier][0])
    wp = {}
    for method, grid in (("B1", B1_GRID), ("B2", B2_GRID), ("B3", B3_GRID)):
        by_ctrl = defaultdict(list)
        for r in rows:
            if not r["entry"].endswith("buggy"):
                by_ctrl[r["entry"]].append(r)
        for ctrl, rs in by_ctrl.items():
            ok = [g for g in grid if not any(baseline_flags(r, method, g, dev) for r in rs)]
            wp[(method, ctrl)] = min(ok) if ok else None
    return wp


def rate(xs):
    xs = list(xs)
    return {"k": int(sum(xs)), "n": len(xs), "rate": (sum(xs) / len(xs)) if xs else None}


def rq2(reports, rows, tier=1, black_box=False):
    conf = set(TIERS[tier][1])
    base = {(r["entry"], r["condition"]): r for r in rows}
    wp = working_points(rows, tier)
    table = defaultdict(lambda: defaultdict(list))  # method -> group -> [flag]
    per_entry = defaultdict(Counter)
    bins = Counter()
    for name, rep in reports.items():
        entry, cond = rep["condition"]["entry"], rep["condition"]["name"]
        lab = label(entry, rep["condition"]["dims"])
        group = "positive" if lab else ("negative (control)" if not entry.endswith("buggy") else "negative (planted, inactive)")
        if black_box:
            o = (rep.get("outputs") or {}).get("out")
            rel = ((o or {}).get("total_black_box") or {}).get("scale", {}).get("relative_rms") if o else None
            outcome = "not established" if rel is None else ("inconsistent" if rel >= 1e-5 else "equality not ruled out")
            b = "error" if rel is None else ("candidate" if rel >= 1e-5 else "below 1e-5")
        else:
            outcome, b = tool_rq2(rep)
        bins[(group, b)] += 1
        per_entry[entry][outcome] += 1
        table["tool"][group].append(outcome == "inconsistent")
        table["tool: not established"][group].append(outcome == "not established")
        row = base.get((entry, cond))
        if row is None:
            continue
        for method in ("B1", "B2", "B3"):
            table[f"{method} default"][group].append(baseline_flags(row, method, DEFAULTS[method], conf))
            w = wp.get((method, control_of(entry)))
            table[f"{method} swept"][group].append(w is not None and baseline_flags(row, method, w, conf))
    return {"rates": {m: {g: rate(v) for g, v in gs.items()} for m, gs in table.items()},
            "bins": {f"{g} | {b}": n for (g, b), n in sorted(bins.items())},
            "per_entry": {e: dict(c) for e, c in sorted(per_entry.items())},
            "working_points": {f"{m} {c}": v for (m, c), v in sorted(wp.items())}}


def rq1(reports, declared):
    per = defaultdict(lambda: {"conditions": 0, "errors": 0, "incomplete": 0, "max_rel_width": 0.0,
                               "timing": defaultdict(list), "wall": []})
    for rep in reports.values():
        e = rep["condition"]["entry"]
        d = per[e]
        d["conditions"] += 1
        if "error" in rep:
            d["errors"] += 1
            continue
        o = rep["outputs"]["out"]
        if o["reference_classes"]["complete_fraction"] < 1.0:
            d["incomplete"] += 1
        sc = o["numerical"].get("scale") or {}
        if sc.get("rms_reference"):
            d["max_rel_width"] = max(d["max_rel_width"], sc["max_K_R_width"] / sc["rms_reference"])
        for k, v in rep.get("timing_seconds", {}).items():
            d["timing"][k].append(v)
        d["wall"].append(rep.get("wall_seconds", 0))
    dec = defaultdict(lambda: {"checked": 0, "violations": 0, "not_complete": 0, "K_matches_returned": True})
    for r in declared:
        x = dec[r["entry"]]
        for k in ("checked", "violations", "not_complete"):
            x[k] += r[k]
        x["K_matches_returned"] &= r["K_matches_returned"]
    out = {}
    for e, d in sorted(per.items()):
        out[e] = {"conditions": d["conditions"], "errors": d["errors"], "incomplete_reference": d["incomplete"],
                  "max_K_R_width_over_rms": d["max_rel_width"],
                  "median_seconds": {k: float(np.median(v)) for k, v in d["timing"].items()},
                  "total_wall_seconds": float(np.sum(d["wall"])), "declared_check": dict(dec.get(e, {}))}
    return out


def holm_family(reports, totals):
    """Holm over every rule test of the RQ3 table (entries x {e_num, total} x rules), on the conservative two-sided
    p-values; a test whose t approximation is suspect must also be confirmed by its bootstrap-t companion."""
    sys.path.insert(0, str(ROOT / "src"))
    from kernel_analyzer.reference_eval.analysis import holm_adjusted

    tests = []
    for name, rep in reports.items():
        if "error" in rep:
            continue
        recs = {"e_num": rep["outputs"]["out"]["numerical"]}
        t = totals.get(name)
        if t and "error" not in t:
            recs["total"] = t["outputs"]["out"]["total_black_box"]
        for q, rec in recs.items():
            for r in rec.get("rules", []):
                if "p_value_two_sided_conservative" in r:
                    tests.append(((name, q, r["rule"]), r))
    adj = holm_adjusted([r["p_value_two_sided_conservative"] for _, r in tests])
    out = {}
    for ((name, q, rule), r), a in zip(tests, adj):
        ok = a <= 0.05 and str(r.get("verdict", "")).startswith("DETECTED")
        if ok and str(r.get("t_approximation", "ok")).startswith("suspect"):
            ok = str((r.get("robust") or {}).get("verdict", "")).startswith("DETECTED")
        out[(name, q, rule)] = r["verdict"] if ok else "NOT_CONFIRMED"
    return out, len(tests)


def rq3(reports, totals, rows):
    conf = set(TIERS[2][1])
    base = {(r["entry"], r["condition"]): r for r in rows}
    holm, n_tests = holm_family(reports, totals)
    out, counts, counts_raw = {}, Counter(), Counter()
    for name, rep in sorted(reports.items()):
        entry, cond = rep["condition"]["entry"], rep["condition"]["name"]
        if "error" in rep:
            out[entry] = {"error": rep["error"]}
            continue
        o = rep["outputs"]["out"]
        num = detected(o["numerical"])
        tot_rep = totals.get(name)
        tot = detected(((tot_rep or {}).get("outputs") or {}).get("out", {}).get("total_black_box")) if tot_rep else None
        row = base.get((entry, cond))
        b1 = baseline_flags(row, "B1", 1.0, conf) if row else None
        raw_hit = any(v.startswith("DETECTED") for _, v in (num or []) + (tot or []))
        h_num = [(r, holm.get((name, "e_num", r), "NOT_CONFIRMED")) for r, _ in num or []]
        h_tot = [(r, holm.get((name, "total", r), "NOT_CONFIRMED")) for r, _ in tot or []]
        hit = any(v.startswith("DETECTED") for _, v in h_num + h_tot)
        out[entry] = {"condition": cond, "e_num": num, "e_num_holm": h_num,
                      "e_num_relative_rms": (o["numerical"].get("scale") or {}).get("relative_rms"),
                      "total": tot, "total_holm": h_tot, "total_relative_rms": (o.get("total") or {}).get("relative_rms"),
                      "e_sem_relative_rms": ((o.get("semantic") or {}).get("scale") or {}).get("relative_rms"),
                      "B1_default_flag": b1, "planted": entry.endswith("buggy")}
        for c, h in ((counts, hit), (counts_raw, raw_hit)):
            if b1 is False and h:
                c["passes tolerance, mean effect detected"] += 1
            if b1 is True and not h:
                c["fails tolerance, no mean effect confirmed"] += 1
        if b1 is False and hit:
            out[entry]["information_difference"] = "passes tolerance, mean effect detected"
        if b1 is True and not hit:
            out[entry]["information_difference"] = "fails tolerance, no mean effect confirmed"
    return {"per_entry": out, "holm_family_size": n_tests, "information_difference_counts": dict(counts),
            "information_difference_counts_without_holm": dict(counts_raw)}


def ablations(t1, t2):
    res = {}
    for tier, reps in ((1, t1), (2, t2)):
        a1 = load_reports(f"ablation_a1_tier{tier}")
        agree = Counter()
        for name, a in a1.items():
            rep = reps.get(name)
            if not rep or "error" in rep:
                continue
            tool = any_detected(rep["outputs"]["out"]["numerical"])
            abl = any_detected(a["numerical_fp64_rerun"])
            agree[f"tool {'detects' if tool else 'no'} / fp64 rerun {'detects' if abl else 'no'}"] += 1
        res[f"A1 tier{tier}"] = dict(agree)
        a2 = Counter()
        for name, rep in reps.items():
            if "error" in rep:
                continue
            o = rep["outputs"]["out"]
            # A2: the mean test replaced by a magnitude decision at B1's default tolerance (protocol section 6)
            mag = (o["numerical"].get("scale") or {}).get("max_abs_residual", 0) > TOLS[rep["condition"]["entry"]]
            mean = any_detected(o["numerical"])
            a2[f"e_num: magnitude {'flags' if mag else 'passes'} / mean test {'detects' if mean else 'no'}"] += 1
        res[f"A2 tier{tier}"] = dict(a2)
    return res


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    TOLS.update({e["entry"]: e["tolerance_float32"] for e in manifest["entries"]})
    t1, t2 = load_reports("tier1"), load_reports("tier2")
    s = {"protocol": "docs/external_eval_protocol_20261006.md", "corpus_commit": manifest["commit"],
         "tier1_conditions_reported": len(t1), "tier1_conditions_planned": manifest["conditions_triton"],
         "rq1": rq1(t1, load_jsonl("declared_check*.jsonl")),
         "rq2_tier1": rq2(t1, load_jsonl("baselines_tier1*.jsonl")),
         "rq3_tier2": rq3(t2, load_reports("tier2_total"), load_jsonl("baselines_tier2*.jsonl")),
         "ablations": ablations(t1, t2),
         "black_box_tier1": rq2(load_reports("blackbox_tier1"), load_jsonl("baselines_blackbox_tier1*.jsonl"), black_box=True)}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
    # compact per-condition table (the tier-1 reports themselves stay local, see .gitignore)
    with open(OUT / "conditions_tier1.jsonl", "w") as fh:
        for name, rep in sorted(t1.items()):
            outcome, b = tool_rq2(rep)
            o = (rep.get("outputs") or {}).get("out") or {}
            fh.write(json.dumps({
                "entry": rep["condition"]["entry"], "condition": rep["condition"]["name"],
                "label_positive": label(rep["condition"]["entry"], rep["condition"]["dims"]), "tool": outcome, "bin": b,
                "complete_fraction": (o.get("reference_classes") or {}).get("complete_fraction"),
                "e_num_relative_rms": ((o.get("numerical") or {}).get("scale") or {}).get("relative_rms"),
                "e_sem_relative_rms": ((o.get("semantic") or {}).get("scale") or {}).get("relative_rms"),
                "e_sem_certified_frac": (o.get("semantic_elementwise") or {}).get("certified_frac"),
                "total_relative_rms": (o.get("total") or {}).get("relative_rms"),
                "max_K_R_width": ((o.get("numerical") or {}).get("scale") or {}).get("max_K_R_width"),
                "timing_seconds": rep.get("timing_seconds"), "error": rep.get("error")}, default=float) + "\n")
    print(json.dumps({k: s[k] for k in ("tier1_conditions_reported", "tier1_conditions_planned")}))
    print(json.dumps(s["rq2_tier1"]["rates"], indent=1))


if __name__ == "__main__":
    main()
