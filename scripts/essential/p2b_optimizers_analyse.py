"""Analysis of the 2b optimizer runs (p2b_optimizers.py): E and P, pre-registered vs post-hoc.

E: K - K_eager on the same device, K_eager = torch for_loop (single-tensor) of the same torch build; τ32 = 2^-12 (1 + |K_eager|)
per element of every parameter and state tensor after every step ("step" counters exactly).  Library candidates are compared
on parameters (and bnb state1/state2 mapped to exp_avg/exp_avg_sq); HF Adafactor implements a different algorithm from
torch.optim.Adafactor (relative step and parameter-scale terms), so E is not applicable to it.

P, pre-registered (section 3.2 registry):
  P_impl     for_loop == foreach == fused -- the same comparison as E for the impl candidates, reported once under E;
  P_mirror   maximize(g) == minimize(-g): bitwise (negation is exact, the documented algorithms negate g first);
  P_none     grad None skips the parameter: its value and state are bitwise unchanged across the step, while a zero
             gradient advances the step counter (documented difference);
  P_restore  save -> fresh optimizer -> load -> continue == uninterrupted, bitwise;
  P_amsgrad  max_exp_avg_sq non-decreasing and >= exp_avg_sq (exact comparisons).
State sequence "skipped (non-finite) -> next normal" (registered): GradScaler skips the step; snapshot unchanged at the
skip and the remaining trajectory equals the run without that step (unscaling is by powers of two, exact).

P, post-hoc (reported apart): step counter = number of applied steps; Adam == AdamW on groups with weight decay 0
(bitwise, both skip the decay branch); SGD first-step momentum buffer = effective gradient (documented b_1 = g_1, group 1,
wd 0, bitwise); Adafactor factored second moments agree in mean (row vs column, τ32); everything finite.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import p2b_optimizers as O

REF = {"cpu": "torch_for_loop_cpu", "cuda": "torch_for_loop_cuda"}
STATE_MAP = {"bnb": {"state1": "exp_avg", "state2": "exp_avg_sq"}}


def load():
    out = {}
    for p in sorted(O.CACHE.glob("*.pkl")):
        if p.stem.endswith("_pilot"):
            continue
        out[p.stem] = pickle.loads(p.read_bytes())
    return out


def _num(v):
    return np.asarray(v, dtype=np.float64)


def tau_viol(k, r):
    k, r = _num(k), _num(r)
    if k.shape != r.shape:
        return np.ones(1, bool), np.inf
    fin = np.isfinite(r) & np.isfinite(k)
    bad = ~fin & ~((np.isnan(k) & np.isnan(r)) | (k == r))
    d = np.abs(k - r)
    lim = O.TAU32 * (1 + np.abs(r))
    bad |= fin & (d > lim)
    ratio = float(np.max(np.where(fin, d / lim, 0), initial=0.0))
    return bad, ratio


def traj_equal(a, b):
    """bitwise equality of two trajectories (params + state + lr)."""
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        for x, y in zip(sa["params"], sb["params"]):
            if not np.array_equal(x, y, equal_nan=True):
                return False
        if sa["state"].keys() != sb["state"].keys():
            return False
        for pi in sa["state"]:
            if sa["state"][pi].keys() != sb["state"][pi].keys():
                return False
            for k in sa["state"][pi]:
                if not np.array_equal(_num(sa["state"][pi][k]), _num(sb["state"][pi][k]), equal_nan=True):
                    return False
    return True


def first_diff(a, b):
    for t, (sa, sb) in enumerate(zip(a, b)):
        for i, (x, y) in enumerate(zip(sa["params"], sb["params"])):
            if not np.array_equal(x, y, equal_nan=True):
                return {"step": t, "where": f"param {i}", "max_abs": float(np.nanmax(np.abs(x - y)))}
        for pi in sa["state"]:
            for k in sa["state"][pi]:
                x, y = _num(sa["state"][pi][k]), _num(sb["state"].get(pi, {}).get(k, np.nan))
                if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                    return {"step": t, "where": f"state {pi}.{k}"}
    return {"step": None, "where": "length" if len(a) != len(b) else "?"}


# ------------------------------------------------------------------------------------------------ E

def e_compare(cand_meta, kt, rt):
    """per-condition E over a trajectory; returns (violating elements, compared elements, max ratio, first violation)."""
    lib = cand_meta["lib"]
    nbad = ncmp = 0
    worst, first = 0.0, None
    for t, (sk, sr) in enumerate(zip(kt, rt)):
        pairs = [(f"param{i}", x, y) for i, (x, y) in enumerate(zip(sk["params"], sr["params"]))]
        if lib == "torch":
            for pi in sr["state"]:
                for k, v in sr["state"][pi].items():
                    pairs.append((f"state{pi}.{k}", sk["state"].get(pi, {}).get(k), v))
        elif lib in STATE_MAP:
            for pi in sr["state"]:
                for kk, rk in STATE_MAP[lib].items():
                    if kk in sk["state"].get(pi, {}) and rk in sr["state"][pi]:
                        pairs.append((f"state{pi}.{rk}", sk["state"][pi][kk], sr["state"][pi][rk]))
        for name, x, y in pairs:
            if x is None:
                nbad += 1
                ncmp += 1
                first = first or {"step": t, "where": name, "why": "missing in candidate"}
                continue
            if name.endswith(".step"):
                bad = _num(x).ravel() != _num(y).ravel()
                ratio = 0.0
            else:
                bad, ratio = tau_viol(x, y)
            nbad += int(bad.sum())
            ncmp += int(bad.size)
            worst = max(worst, ratio)
            if bad.any() and first is None:
                first = {"step": t, "where": name}
    if len(kt) != len(rt):
        nbad += 1
        first = first or {"why": "trajectory length differs"}
    return nbad, ncmp, worst, first


def bitwise_traj_params(kt, rt):
    return all(np.array_equal(a, b, equal_nan=True) for sk, sr in zip(kt, rt) for a, b in zip(sk["params"], sr["params"]))


# ------------------------------------------------------------------------------------------------ P

def p_props(cond, rec):
    """pre-registered and post-hoc properties of one (condition, seed) record of one candidate."""
    pre, post = {}, {}
    b = rec.get("base")
    if not b or b["status"] != "ok":
        return pre, post
    bt = b["traj"]
    m = rec.get("mirror")
    if m and m["status"] == "ok":
        pre["P_mirror"] = traj_equal(bt, m["traj"])
    u = rec.get("uninterrupted")
    if u and u["status"] == "ok":
        pre["P_restore"] = traj_equal(bt, u["traj"])
    st = cond["state"]
    if st == "grad_none":
        s0, s1 = bt[O.EVENT], bt[O.EVENT + 1]
        same = np.array_equal(s0["params"][1], s1["params"][1]) and all(
            np.array_equal(_num(s0["state"][1][k]), _num(s1["state"][1][k])) for k in s0["state"].get(1, {}))
        pre["P_none"] = bool(same)
    if st == "zero_grad":
        s0, s1 = bt[O.EVENT], bt[O.EVENT + 1]
        if "step" in s0["state"].get(1, {}):
            pre["P_none"] = bool(_num(s1["state"][1]["step"]) == _num(s0["state"][1]["step"]) + 1)
    if cond["opt"] == "adam_amsgrad":
        ok = True
        for t in range(1, len(bt)):
            for pi, s in bt[t]["state"].items():
                if "max_exp_avg_sq" not in s:
                    continue
                mx, v = _num(s["max_exp_avg_sq"]), _num(s["exp_avg_sq"])
                prev = bt[t - 1]["state"].get(pi, {}).get("max_exp_avg_sq")
                ok &= bool(np.all(mx >= v))
                if prev is not None:
                    ok &= bool(np.all(mx >= _num(prev)))
        pre["P_amsgrad"] = ok
    if st == "nonfinite_skip":
        r = rec.get("removed")
        if r and r["status"] == "ok":
            same_at_skip = traj_equal([bt[O.EVENT]], [bt[O.EVENT + 1]])
            rest = traj_equal(bt[:O.EVENT + 1] + bt[O.EVENT + 2:], r["traj"])
            pre["P_skip_sequence"] = bool(same_at_skip and rest)
            pre["_skip_detail"] = {"unchanged_at_skip": bool(same_at_skip), "continues_as_removed": bool(rest)}
    # post-hoc
    applied = defaultdict(int)
    ok_step = True
    grads = O.grad_sequence(cond, 0, "base")          # only the None / skip pattern is used, which is seed-independent
    for t in range(1, len(bt)):
        skipped = st == "nonfinite_skip" and t - 1 == O.EVENT
        for pi, s in bt[t]["state"].items():
            if not skipped and grads[t - 1][pi] is not None:
                applied[pi] += 1
            if "step" in s:
                ok_step &= int(_num(s["step"])) == applied[pi]
    post["H_step_count"] = ok_step
    post["H_finite"] = all(np.isfinite(x).all() for s in bt for x in s["params"])
    if cond["opt"] == "sgd_dampening" or cond["opt"] == "sgd_nesterov":
        s1 = bt[1]["state"].get(2, {})
        if "momentum_buffer" in s1:
            g = O.grad_sequence(cond, rec["_seed"], "base")[0][2]
            g = np.asarray(np.float32(-g if cond["maximize"] else g), np.float64)
            post["H_sgd_first_buffer"] = bool(np.array_equal(_num(s1["momentum_buffer"]), g))
    if cond["opt"] == "adafactor":
        ok = True
        for s in bt[1:]:
            for pi, d in s["state"].items():
                rk = "row_var" if "row_var" in d else ("exp_avg_sq_row" if "exp_avg_sq_row" in d else None)
                ck = "col_var" if "col_var" in d else ("exp_avg_sq_col" if "exp_avg_sq_col" in d else None)
                if rk and ck and d[rk] is not None and d[ck] is not None and np.ndim(d[rk]) and np.ndim(d[ck]):
                    rv, cv = _num(d[rk]), _num(d[ck])
                    if rk == "row_var":                   # torch keeps the reduced axis (..., m, 1) / (..., 1, n)
                        rv, cv = rv.squeeze(-1), cv.squeeze(-2)
                    rm, cm = rv.mean(axis=-1), cv.mean(axis=-1)
                    bad, _ = tau_viol(rm, cm)
                    ok &= not bad.any()
        post["H_adafactor_factor_means"] = ok
    return pre, post


def main():
    data = load()
    conds = {c["id"]: c for c in O.conditions()}
    out = {"conditions": len(conds), "seeds": list(O.SEEDS), "candidates": {}, "E": {}, "P": {}, "posthoc": {},
           "cross": {}, "examples": defaultdict(list)}
    for name, d in data.items():
        meta = d["meta"]
        cand = meta["candidate"]
        st = defaultdict(int)
        for rec in d["res"].values():
            for r in rec.values():
                st[r["status"]] += 1
        out["candidates"][name] = {"meta": {k: v for k, v in meta.items() if k != "candidate"}, "statuses": dict(st)}
        # E
        ref = REF.get(cand["device"])
        e = None
        if cand["lib"] == "hf":
            e = {"status": "not applicable", "why": "HF Adafactor is a different algorithm from torch.optim.Adafactor"}
        elif name != ref and ref in data:
            per = {"compared": 0, "violating_conditions": 0, "bitwise_param_trajectories": 0, "max_ratio": 0.0,
                   "unsupported": 0, "error": 0}
            for key, rec in d["res"].items():
                cid, seed = key
                kb, rb = rec.get("base"), data[ref]["res"][key].get("base")
                if kb["status"] != "ok":
                    per[kb["status"]] += 1
                    continue
                nbad, ncmp, worst, first = e_compare(cand, kb["traj"], rb["traj"])
                per["compared"] += 1
                per["max_ratio"] = max(per["max_ratio"], worst)
                per["bitwise_param_trajectories"] += int(bitwise_traj_params(kb["traj"], rb["traj"]))
                if nbad:
                    per["violating_conditions"] += 1
                    if len(out["examples"]["E"]) < 20:
                        out["examples"]["E"].append({"candidate": name, "condition": cid, "seed": seed, "violating": nbad,
                                                     "of": ncmp, "first": first})
            e = dict(per, status="executed", reference=ref)
        elif name == ref:
            e = {"status": "reference (K_eager)"}
        out["E"][name] = e
        # P
        pre_t, post_t = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for (cid, seed), rec in d["res"].items():
            rec = dict(rec, _seed=seed)
            pre, post = p_props(conds[cid], rec)
            for tally, res, kind in ((pre_t, pre, "P"), (post_t, post, "posthoc")):
                for k, v in res.items():
                    if k.startswith("_"):
                        continue
                    tally[k][0] += 1
                    tally[k][1] += int(not v)
                    if not v and len(out["examples"][kind]) < 30:
                        ex = {"candidate": name, "condition": cid, "seed": seed, "property": k}
                        if k == "P_skip_sequence":
                            ex["detail"] = pre.get("_skip_detail")
                        if k in ("P_mirror", "P_restore"):
                            other = rec["mirror" if k == "P_mirror" else "uninterrupted"]["traj"]
                            ex["first_difference"] = first_diff(rec["base"]["traj"], other)
                        out["examples"][kind].append(ex)
        out["P"][name] = {k: {"checked": v[0], "violated": v[1]} for k, v in pre_t.items()}
        out["posthoc"][name] = {k: {"checked": v[0], "violated": v[1]} for k, v in post_t.items()}
        # post-hoc cross-condition: Adam == AdamW on groups with weight decay 0
        tally = [0, 0]
        for c in conds.values():
            if c["opt"] != "adam":
                continue
            twin = c["id"].replace("opt_adam_", "opt_adamw_", 1)
            if twin not in conds:
                continue
            for seed in O.SEEDS:
                a, w = d["res"].get((c["id"], seed), {}).get("base"), d["res"].get((twin, seed), {}).get("base")
                if not a or not w or a["status"] != "ok" or w["status"] != "ok":
                    continue
                idx = [0, 1, 2] if c["weight_decay"] == 0 else [2]
                same = all(np.array_equal(sa["params"][i], sw["params"][i]) for sa, sw in zip(a["traj"], w["traj"]) for i in idx)
                tally[0] += 1
                tally[1] += int(not same)
                if not same and len(out["examples"]["posthoc"]) < 30:
                    out["examples"]["posthoc"].append({"candidate": name, "condition": c["id"], "seed": seed,
                                                       "property": "H_adam_eq_adamw_wd0"})
        if tally[0]:
            out["posthoc"][name]["H_adam_eq_adamw_wd0"] = {"checked": tally[0], "violated": tally[1]}
    out["examples"] = dict(out["examples"])
    O.OUT.mkdir(parents=True, exist_ok=True)
    (O.OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for name in data:
        e = out["E"][name] or {}
        print(f"{name:28s} E: {e.get('status')} {e.get('violating_conditions', '')}/{e.get('compared', '')} "
              f"bitwise {e.get('bitwise_param_trajectories', '')} max {e.get('max_ratio', 0):.3g}")
        print("   P :", {k: f"{v['violated']}/{v['checked']}" for k, v in out["P"][name].items()})
        print("   H :", {k: f"{v['violated']}/{v['checked']}" for k, v in out["posthoc"][name].items()})
