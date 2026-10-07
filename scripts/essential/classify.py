#!/usr/bin/env python3
"""Phase-1 classification (protocol sections 5-7): F, E, P and FR per (condition, candidate), forward and backward.

    python scripts/essential/classify.py --family pool      # -> results/essential/phase1/classification_pool.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidates as K  # noqa: E402
import common  # noqa: E402
import run_phase1 as R  # noqa: E402
import spec_index_scatter as six  # noqa: E402
import spec_pooling as spo  # noqa: E402

OUT = common.ROOT / "results/essential/phase1"
SEM_REL = 2.0 ** -20                       # protocol section 12 entry 6
NIGHTLY = ["nightly_eager_cuda32", "nightly_inductor_cuda32"]
EREF = {"eager_cpu32": "eager_cpu64", "eager_cuda32": "eager_cpu64", "eager_cuda_bf16": "eager_cpu64",
        "inductor_cuda32": "eager_cuda32", "inductor_cuda_bf16": "eager_cuda_bf16",
        "nightly_eager_cuda32": "eager_cpu64", "nightly_inductor_cuda32": "nightly_eager_cuda32"}
PHASE = {"ce": {"loss": "fwd", "grad": "bwd"}, "pool": {"out": "fwd", "indices": "fwd", "grad": "bwd"},
         "index": {"out": "fwd", "grad_self": "bwd", "grad_source": "bwd"}}


def base_name(cand):
    return cand[len("nightly_"):] if cand.startswith("nightly_") else cand


def cand_dtype(cand):
    return K.DTYPE_NAME[K.CANDIDATES[base_name(cand)][1]]


def eps_kind(family, cond, cand):
    compiled = K.CANDIDATES[base_name(cand)][2]
    return "f32" if (family == "ce" and cond["eps"] != 0 and cand_dtype(cand) != "float64" and compiled) else "f64"


def readings_of(family, cond):
    if family == "ce":
        return ["R_A", "R_B", "R_C"]
    if family == "pool":
        return ["R2", "R1"] if cond["op"] == "avg_pool" else ["R_prop", "R_ignore"]
    if cond.get("reduce") in ("amax", "amin"):
        return ["R_prop", "R_ignore"]
    return ["main", "error_variant_mean_divides_by_contrib_count"] if cond.get("reduce") == "mean" and cond.get("include_self") else ["main"]


def main_of(family, cond):
    return readings_of(family, cond)[0]


def exact_inputs(family, cond, phase="fwd"):
    """every intermediate exact in the candidate's dtype (protocol 6: then K = f is required).  Backward of amax /
    amin splits the gradient as v/N at ties, which is not exact, so only the forward is exact there."""
    if family == "pool":
        return cond["op"] == "max_pool"
    if family == "index":
        ops = ("sum", "prod", "amax", "amin") if phase == "fwd" else ("sum", "prod")
        return cond["values"] in ("ints", "special_ints") and (cond["op"] == "index_add" or cond.get("reduce") in ops)
    return False


def doc_split(cond):
    """protocol section 12 entry 4: MaxPool1d ceil_mode length formula differs between the 2.10 and main docs."""
    if cond.get("op") != "max_pool" or cond["nd"] != 1 or not cond["ceil_mode"]:
        return False
    L, k, s, p, d = cond["size"][0], cond["kernel"][0], cond["stride"][0], cond["padding"][0], cond["dilation"][0]
    num = L + 2 * p - d * (k - 1) - 1

    def fix(out):
        return out - 1 if (out - 1) * s >= L + p else out
    return fix(math.ceil((num + s - 1) / s) + 1) != fix(math.floor((num + s - 1) / s) + 1)


# ------------------------------------------------------------------------------------------------ spec values per output

def spec_outputs(family, cond, spec, reading):
    """{output name: (lo, hi)} of the specification under one reading (None where it has no value)."""
    if spec is None or spec.get("status") != "ok":
        return None
    rd = spec["readings"].get(reading)
    if rd is None or rd.get("status") != "ok":
        return None
    if family == "ce":
        return {"loss": rd["loss"], "grad": rd["grad"]}
    if family == "pool":
        out = {"out": rd["out"]}
        if rd.get("grad") is not None:
            out["grad"] = rd["grad"]
        return out
    out = {"out": rd["out"]}
    g = spec.get("grad")
    if g is not None:
        for nm, key in (("grad_self", "self"), ("grad_source", "source")):
            lo, hi = g[key]
            none = np.isnan(lo) | np.isnan(hi)                         # ties: no strict value
            out[nm] = (np.where(none, -np.inf, lo), np.where(none, np.inf, hi))
    return out


def _same_harness(family, cond, seed, r, sp):
    """guard: the candidate used the deterministic upstream the specification used, and received the same inputs."""
    inp = R.FAMILIES[family][1](cond, seed)
    if family == "ce":
        v = K.ce_upstream(cond, seed)
    elif family == "pool":
        v = K.pool_upstream(cond, inp, tuple(np.asarray(r["upstream"]).shape))
    else:
        v = K.index_upstream(inp, tuple(np.asarray(inp["self"]).shape))
    if not np.array_equal(np.asarray(r["upstream"], dtype=np.float64), np.asarray(v, dtype=np.float64)):
        return False
    rx = sp.get("received", {})
    for k, val in r["received"].items():
        if k in rx and isinstance(val, np.ndarray) and not np.array_equal(val, np.asarray(rx[k]), equal_nan=True):
            return False
    return True


def output_deviation(k, enc, dtype, exact):
    lo, hi = enc
    k = np.asarray(k, dtype=np.float64).reshape(np.shape(lo))
    dev, judged = common.deviation(k, lo, hi, dtype, exact)
    return dev, judged


# ------------------------------------------------------------------------------------------------ per (condition, candidate)

def classify_one(family, cond, cand, raw, specs):
    """returns {phase: record} for one condition over its seeds."""
    dtype = cand_dtype(cand)
    rds = readings_of(family, cond)
    res = {}
    for phase in ("fwd", "bwd"):
        exact = exact_inputs(family, cond, phase)
        names = [n for n, ph in PHASE[family].items() if ph == phase and n != "indices"]
        rec = {"seeds": []}
        for seed in R.C.SEEDS:
            r = raw.get((cond["id"], seed), {}).get("base")
            sp = specs.get(seed)
            srec = {"seed": seed}
            if r is None:
                srec["status"] = "missing"
            elif r["status"] != "ok":
                srec["status"] = r["status"]
                srec["reason"] = r.get("reason")
            elif sp is None or sp.get("status") not in ("ok",):
                srec["status"] = "spec_" + (sp or {}).get("status", "missing")
                srec["reason"] = (sp or {}).get("reason")
            elif not _same_harness(family, cond, seed, r, sp):
                srec["status"] = "harness_mismatch"
            else:
                main = spec_outputs(family, cond, sp, main_of(family, cond))
                if main is None:
                    srec["status"] = "spec_main_" + sp["readings"].get(main_of(family, cond), {}).get("status", "missing")
                    srec["candidate_values"] = {n: np.asarray(r["outputs"][n]).reshape(-1)[:8].tolist() for n in names if n in r["outputs"]}
                else:
                    srec["status"] = "ok"
                    per_reading, distinct = {}, False
                    judged_any = compared_any = False
                    for rd in rds:
                        so = spec_outputs(family, cond, sp, rd)
                        if so is None:
                            per_reading[rd] = None
                            continue
                        devs = {}
                        for n in names:
                            if n not in so or n not in r["outputs"]:
                                continue
                            dev, judged = output_deviation(r["outputs"][n], so[n], dtype, exact)
                            judged_any |= bool(judged.any())
                            compared_any = True
                            devs[n] = int(dev.sum())
                        per_reading[rd] = devs
                        if rd != rds[0] and so is not None:
                            for n in names:
                                if n in so and n in main:
                                    a, b = so[n], main[n]
                                    fin = np.isfinite(a[0]) & np.isfinite(b[0])
                                    tol = 2.0 ** -30 * (1 + np.abs(np.where(fin, b[0], 0)))
                                    if (fin & (np.abs(np.where(fin, a[0] - b[0], 0)) > tol)).any() or \
                                       (np.isnan(a[0]) != np.isnan(b[0])).any():
                                        distinct = True
                    srec["deviations"] = per_reading
                    srec["readings_distinct"] = distinct
                    srec["judged"] = judged_any
                    srec["compared"] = compared_any
            rec["seeds"].append(srec)
        oks = [s for s in rec["seeds"] if s["status"] == "ok"]
        statuses = sorted({s["status"] for s in rec["seeds"]})
        if not oks:
            rec["class"] = statuses[0] if len(statuses) == 1 else "mixed:" + ",".join(statuses)
        elif not any(s.get("compared") for s in oks):
            rec["class"] = "set_checks_only"             # no strict spec value (max-pool ties, NaN windows)
        elif not any(s["judged"] for s in oks):
            rec["class"] = "recorded_only"
            rec["max_deviating_elements_main"] = max(sum((s["deviations"].get(rds[0]) or {}).values()) for s in oks)
        else:
            compat = [rd for rd in rds if all(s["deviations"].get(rd) is not None and sum(s["deviations"][rd].values()) == 0 for s in oks)]
            rec["compatible_readings"] = compat
            rec["readings_distinct"] = any(s["readings_distinct"] for s in oks)
            if rds[0] in compat:
                rec["class"] = "compatible"
            elif compat:
                rec["class"] = ("error_variant:" if compat[0].startswith("error_variant") else "reading:") + compat[0]
            else:
                rec["class"] = "deviates"
            if len(oks) < len(rec["seeds"]):
                rec["class"] += " (some seeds: " + ",".join(sorted({s["status"] for s in rec["seeds"] if s["status"] != "ok"})) + ")"
        res[phase] = rec
    return res


def e_deviation(family, cond, cand, raw, raw_ref):
    """E: candidate against its eager reference (same device and dtype where defined), per phase."""
    dtype = cand_dtype(cand)
    out = {}
    for phase in ("fwd", "bwd"):
        exact = exact_inputs(family, cond, phase)
        names = [n for n, ph in PHASE[family].items() if ph == phase and n != "indices"]
        devs, judged_any, compared = 0, False, 0
        for seed in R.C.SEEDS:
            a = raw.get((cond["id"], seed), {}).get("base")
            b = raw_ref.get((cond["id"], seed), {}).get("base") if raw_ref else None
            if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                continue
            for n in names:
                if n in a["outputs"] and n in b["outputs"]:
                    kb = np.asarray(b["outputs"][n], dtype=np.float64)
                    ka = np.asarray(a["outputs"][n], dtype=np.float64)
                    if ka.shape != kb.shape:
                        devs += 1
                        judged_any = True
                        continue
                    d, j = common.deviation(ka, kb, kb, dtype, exact)
                    devs += int(d.sum())
                    judged_any |= bool(j.any())
                    compared += 1
        out[phase] = {"deviating_elements": devs, "judged": judged_any, "compared_outputs": compared}
    return out


def combine(c, e, eref_class):
    """protocol 6.6 categories from the candidate's class, the E comparison and the eager reference's class."""
    def dev(x):
        return x.startswith(("deviates", "error_variant"))
    if not (c.startswith(("compatible", "deviates", "reading", "error_variant"))):
        return c                                         # recorded_only, set_checks_only, statuses
    if eref_class is None:
        return "eager_itself:" + c
    if not eref_class.startswith(("compatible", "deviates", "reading", "error_variant")):
        return c + " | eager_ref:" + eref_class
    e_dev = e["deviating_elements"] > 0
    if c.startswith("compatible") and eref_class.startswith("compatible"):
        return "compatible"
    if dev(c) and dev(eref_class) and not e_dev:
        return "shared_deviation"
    if dev(c) and eref_class.startswith("compatible"):
        return "candidate_error"
    if c.startswith("compatible") and dev(eref_class):
        return "candidate_compatible_eager_deviates"
    if dev(c) and dev(eref_class):
        return "both_deviate_differently"
    if c.startswith("reading") or eref_class.startswith("reading"):
        same = c == eref_class
        return f"reading_difference ({'same as eager' if same else 'eager: ' + eref_class}; candidate: {c})"
    return c + " | eager_ref:" + eref_class


# ------------------------------------------------------------------------------------------------ properties (P)

def _close(a, b, dtype, exact):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    d, _ = common.deviation(a, b, b, dtype if dtype in common.TAU else "float32", exact)
    return not d.any()


def properties(family, cond, cand, raw, specs):
    """reference-free checks per seed; bf16 candidates are recorded (judged = False)."""
    dtype = cand_dtype(cand)
    exact = exact_inputs(family, cond)
    judged = dtype in common.TAU or exact
    out, examples = {}, {}
    for seed in R.C.SEEDS:
        rec = raw.get((cond["id"], seed), {})
        b = rec.get("base")
        if not b or b["status"] != "ok":
            continue
        res = {}
        o = b["outputs"]
        if family == "ce":
            for v in ("shift", "perm"):
                t = rec.get(v)
                if t and t["status"] == "ok":
                    g2 = t["outputs"]["grad"][:, ::-1] if v == "perm" else t["outputs"]["grad"]
                    res[f"{v}_invariance"] = _close(t["outputs"]["loss"], o["loss"], dtype, False) and _close(g2, o["grad"], dtype, False)
            g = np.asarray(o["grad"])
            sp = specs.get(seed)
            main = (sp or {}).get("readings", {}).get("R_A", {})
            defined = main.get("status") == "ok"          # precondition: the gradient is defined (not CE-A3)
            if defined and np.isfinite(g).all():
                rs = np.abs(g.sum(1))
                res["grad_row_sums_zero"] = bool((rs <= common.TAU.get(dtype, 2 ** -7) * (1 + np.abs(g).sum(1))).all())
            if defined and cond["target"] == "index" and np.isfinite(g).all():
                ign = np.array([t == cond["ignore_index"] for t in R.FAMILIES["ce"][1](cond, seed)["target"]])
                res["ignored_rows_zero_grad"] = bool((g[ign] == 0).all()) if ign.any() else None
        elif family == "pool":
            x = b["received"]["x"]
            if cond["op"] == "avg_pool":
                y, gx, v = o["out"], o["grad"], b["upstream"]
                lhs, rhs = float((y * v).sum()), float((x * gx).sum())
                scale = float(np.abs(y * v).sum() + np.abs(x * gx).sum()) + 1.0
                res["adjoint"] = abs(lhs - rhs) <= common.TAU.get(dtype, 2 ** -7) * scale
            else:
                t = rec.get("shift")
                if t and t["status"] == "ok":       # exact only when x + 1 is exact (integer inputs)
                    res["shift"] = _close(np.asarray(t["outputs"]["out"]) - 1.0, o["out"], dtype, cond["values"] == "ties")
                sp = specs.get(seed)
                main = sp["readings"].get("R_prop") if sp and sp.get("status") == "ok" else None
                if main and main.get("status") == "ok":
                    sets = main["sets"]
                    v = np.asarray(b["upstream"]).reshape(-1)
                    nonempty = np.array([len(s) > 0 for s in sets])
                    if np.isfinite(o["grad"]).all():
                        res["grad_mass"] = float(np.asarray(o["grad"]).sum()) == float(v[nonempty].sum()) if exact else \
                            abs(float(np.asarray(o["grad"]).sum()) - float(v[nonempty].sum())) <= 1e-6 * (1 + np.abs(v).sum())
                    if "indices" in o:
                        spatial = tuple(x.shape[1:])
                        idx = np.asarray(o["indices"]).reshape(-1)
                        per_c = len(sets) // x.shape[0]
                        bad, size = 0, int(np.prod(spatial))
                        for j, (ii, s) in enumerate(zip(idx, sets)):
                            if not s:
                                continue
                            if not (0 <= int(ii) < size) or tuple(int(q) for q in np.unravel_index(int(ii), spatial)) not in s:
                                bad += 1
                                ex = examples.setdefault("indices_not_in_argmax_set", [])
                                if len(ex) < 3:
                                    win_vals = [float(x[(j // per_c,) + tuple(q)]) for q in sorted(s)]
                                    ex.append({"seed": seed, "window": j, "returned_index": int(ii), "argmax_set": sorted(s),
                                               "values_at_set": win_vals})
                        res["indices_in_argmax_set"] = bad == 0
                    k, s_, p, d = cond["kernel"], cond["stride"], cond["padding"], cond["dilation"]
                    nonoverlap = all(s_[i] >= d[i] * (k[i] - 1) + 1 for i in range(len(k)))
                    ties = any(len(s) > 1 for s in sets)
                    if nonoverlap and ties and np.isfinite(o["grad"]).all() and not np.isnan(x).any():
                        shp = tuple(np.asarray(b["upstream"]).shape)
                        nested_sets = np.empty(len(sets), dtype=object)
                        for j, s in enumerate(sets):
                            nested_sets[j] = s

                        def nest(flat, shape):
                            arr = np.array(flat, dtype=object).reshape(shape)
                            return arr.tolist()
                        try:
                            viol = spo.check_max_subgradient(nest([Fraction(float(q)) for q in np.asarray(b["upstream"]).reshape(-1)], shp),
                                                             nest([Fraction(float(q)) for q in np.asarray(o["grad"]).reshape(-1)], x.shape),
                                                             nest(list(sets), shp), list(k), list(s_), list(p), list(d), cond["ceil_mode"])
                            res["tie_subgradient_set"] = len(viol) == 0
                        except spo.SpecInputError:
                            pass
        else:
            inp = R.FAMILIES["index"][1](cond, seed)
            touched = set()
            sp = specs.get(seed)
            selfv = np.asarray(b["received"]["self"])
            pos_map = R.received_inputs  # noqa: F841
            groups = six._groups_index(selfv.tolist(), inp["dim"], inp["index"], np.asarray(b["received"]["source"]).tolist()) \
                if cond["op"] != "scatter_reduce" else six._groups_scatter(selfv.tolist(), inp["dim"], inp["index"], np.asarray(b["received"]["source"]).tolist())
            touched = set(groups)
            out_arr = np.asarray(o["out"])
            mask = np.ones(selfv.shape, dtype=bool)
            for pos in touched:
                mask[pos] = False
            res["untouched_unchanged"] = bool(np.array_equal(out_arr[mask], selfv[mask], equal_nan=True))
            t = rec.get("ones")
            if t and t["status"] == "ok":
                exp = np.ones(selfv.shape)
                red = cond.get("reduce", "sum")
                for pos, contrib in groups.items():
                    n = len(contrib)
                    if cond["op"] == "index_add":
                        exp[pos] = 1 + cond["alpha"] * n
                    elif red == "sum":
                        exp[pos] = (1 + n) if cond["include_self"] else n
                    else:
                        exp[pos] = 1.0
                res["counting_on_ones"] = bool(np.array_equal(np.asarray(t["outputs"]["out"]), exp))
            t = rec.get("reverse")
            if t and t["status"] == "ok":
                res["order_invariance"] = _close(t["outputs"]["out"], o["out"], dtype, exact)
            if sp and sp.get("status") == "ok" and sp.get("ties"):
                okt = True
                gs, gsrc = np.asarray(o["grad_self"]), np.asarray(o["grad_source"])
                for tie in sp["ties"]:
                    grads = ([gs[tie["target"]]] if tie["self_included"] else []) + [gsrc[q] for q in tie["source_positions"]]
                    vals = [v if cond["reduce"] == "amax" else -v for v in tie["values"]]
                    reason = ("non-finite gradient" if not np.isfinite(grads).all() else
                              six.check_extremum_subgradient([Fraction(v) for v in vals], [Fraction(float(g)) for g in grads], tie["upstream"]))
                    if reason is not None:
                        okt = False
                        ex = examples.setdefault("tie_subgradient", [])
                        if len(ex) < 3:
                            ex.append({"seed": seed, "target": tie["target"], "values": tie["values"], "upstream": tie["upstream"],
                                       "grads": [float(g) for g in grads], "reason": reason})
                res["tie_subgradient_set"] = okt
        out[seed] = res
    summary = {}
    for seed, res in out.items():
        for k, v in res.items():
            if v is None:
                continue
            s = summary.setdefault(k, {"checked": 0, "violated": 0})
            s["checked"] += 1
            s["violated"] += int(not v)
    return {"judged": judged, "per_property": summary, "examples": examples}


# ------------------------------------------------------------------------------------------------ FR

def fr_assess(family, cond, cand, fr, specs_f32):
    if fr is None:
        return None
    if fr["status"] != "ok":
        return {"status": "error", "reason": fr.get("reason")}
    rds = readings_of(family, cond)
    out = {"status": "ok", "notes": fr["notes"], "mixed": fr["mixed"], "outputs": {}}
    for name, per in fr["keep"].items():
        o = {"phase": PHASE[family].get(name), "seeds": len(per), "ok_elements": 0, "e_num_excludes_zero": 0,
             "readings": {}}
        for rd in rds:
            sem = iface = 0
            usable = True
            for p, seed in zip(per, R.C.SEEDS):
                so = spec_outputs(family, cond, specs_f32.get(seed), rd)
                if so is None or name not in so:
                    usable = False
                    break
                lo, hi = (np.asarray(a, dtype=np.float64).reshape(-1) for a in so[name])
                okm = p["ok"] & np.isfinite(lo) & np.isfinite(hi)
                s_lo, s_hi = p["r_lo"] - hi, p["r_hi"] - lo
                excl = ((s_lo > 0) | (s_hi < 0)) & okm
                mid = np.abs(0.5 * (s_lo + s_hi))
                big = mid > SEM_REL * (1 + np.abs(np.where(np.isfinite(lo), lo, 0)))
                sem += int((excl & big).sum())
                iface += int((excl & ~big).sum())
            o["readings"][rd] = {"semantic_elements": sem, "interface_elements": iface} if usable else None
        for p in per:
            o["ok_elements"] += int(p["ok"].sum())
            n_lo, n_hi = p["k"] - p["r_hi"], p["k"] - p["r_lo"]
            o["e_num_excludes_zero"] += int((((n_lo > 0) | (n_hi < 0)) & p["ok"]).sum())
        main = o["readings"].get(rds[0])
        o["semantic_vs_main"] = None if main is None else main["semantic_elements"] > 0
        o["compatible_readings"] = [rd for rd, v in o["readings"].items() if v is not None and v["semantic_elements"] == 0]
        out["outputs"][name] = o
    return out


# ------------------------------------------------------------------------------------------------ driver

def load_raw(family, cand):
    p = R.CACHE / "raw" / family / f"{cand}.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(R.FAMILIES), required=True)
    a = ap.parse_args()
    fam = a.family
    conds = [c for c in R.FAMILIES[fam][0]() if c["op"] != "flce"]
    cands = K.FAMILY_CANDIDATES[fam] + NIGHTLY
    raws = {c: load_raw(fam, c) for c in cands}
    frs = {}
    for c in ("inductor_cuda32", "inductor_cuda_bf16"):
        p = R.CACHE / "fr" / fam / f"{c}.pkl"
        if p.exists():
            frs[c] = pickle.loads(p.read_bytes())
    records = []
    for cond in conds:
        held = doc_split(cond)
        per_cand = {}
        for cand in cands:
            raw = raws.get(cand)
            if raw is None:
                continue
            dtype = cand_dtype(cand)
            specs = {s: R.load_spec(fam, cond["id"], s, "base", dtype, eps_kind(fam, cond, cand)) for s in R.C.SEEDS}
            cls = classify_one(fam, cond, cand, raw, specs)
            ref = EREF.get(cand)
            e = e_deviation(fam, cond, cand, raw, raws.get(ref)) if ref else None
            # W8 baseline: the FP64 reference (eager CPU float64) with the same verdict rule
            e64 = e_deviation(fam, cond, cand, raw, raws.get("eager_cpu64")) if cand != "eager_cpu64" else None
            per_cand[cand] = {"class": cls, "E": e, "E64": e64, "P": properties(fam, cond, cand, raw, specs)}
            if cand in frs:
                specs32 = {s: R.load_spec(fam, cond["id"], s, "base", dtype, "f32" if (fam == "ce" and cond["eps"] != 0) else "f64")
                           for s in R.C.SEEDS}
                per_cand[cand]["FR"] = fr_assess(fam, cond, cand, frs[cand].get(cond["id"]), specs32)
        for cand, rec in per_cand.items():
            ref = EREF.get(cand)
            for phase in ("fwd", "bwd"):
                c = rec["class"][phase]["class"]
                rc = per_cand[ref]["class"][phase]["class"] if ref in per_cand else None
                rec["class"][phase]["category"] = combine(c, rec["E"][phase] if rec["E"] else None, rc) if ref else combine(c, None, None)
        records.append({"condition": cond, "held_doc_version_split": held, "candidates": per_cand})
    # counts with denominators
    counts = {}
    for r in records:
        for cand, rec in r["candidates"].items():
            for phase in ("fwd", "bwd"):
                cat = "held_doc_version_split" if r["held_doc_version_split"] else rec["class"][phase]["category"]
                cat = cat.split(" (")[0] if cat.startswith(("compatible", "deviates", "reading:", "error_variant")) else cat
                d = counts.setdefault(cand, {}).setdefault(phase, {})
                d[cat] = d.get(cat, 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    import gzip
    with gzip.open(OUT / f"classification_{fam}.json.gz", "wt") as fh:     # per-seed detail: compressed
        json.dump({"family": fam, "conditions": len(conds), "counts": counts, "records": records}, fh,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    for cand, ph in counts.items():
        for phase, d in ph.items():
            print(fam, cand, phase, sum(d.values()), dict(sorted(d.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
