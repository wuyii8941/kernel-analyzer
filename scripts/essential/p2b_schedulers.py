#!/usr/bin/env python3
"""2b family "schedulers" (protocol v2 section 3; registry tier1.schedulers): E and P on learning-rate sequences.  F stays
closed until the spec is delivered.

Conditions: the coverage plan (10 distinct combinations of sched, resume, steps) x 3 base learning rates (the "seeds":
0.1, 0.03, 1.0).  ``steps`` is the schedule length: warmup_cosine_hf = transformers get_cosine_schedule_with_warmup with
num_training_steps = steps and num_warmup_steps = max(steps // 10, 1) ... clipped to steps; cosine_annealing = CosineAnnealingLR
with T_max = steps, eta_min = base / 10; one_cycle = OneCycleLR with total_steps = steps, max_lr = base; linear_warmup_torch =
LinearLR with start_factor 0.1 over total_iters = steps.  Each run records get_last_lr() after construction and after every
step, for 2 x steps + 2 steps (one_cycle: until it refuses -- it is defined for total_steps steps).  ``resume``: state_dict
after half of the steps (at least 1), a fresh optimizer and scheduler load it and continue.

P (pre-registered): the chained (step) values equal the scheduler's own closed form (``_get_closed_form_lr``, where the class
has one: CosineAnnealingLR, LinearLR); resume == continuous run (bitwise); values at phase boundaries as documented
(OneCycleLR: max_lr / div_factor at the start, max_lr at pct_start x total, max_lr / (div_factor x final_div_factor) at the
end; warmup-cosine: 0 at step 0, base at the end of warmup, 0 at the end; LinearLR: start_factor x base at 0, base from
total_iters on; CosineAnnealingLR: base at 0, eta_min at T_max).

    python scripts/essential/p2b_schedulers.py run --env ka_main
    /data1/tzh/envs/pt_nightly_cpu/bin/python scripts/essential/p2b_schedulers.py run --env nightly
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_schedulers.py run --env liger
    python scripts/essential/p2b_schedulers.py analyse
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/schedulers"
OUT = ROOT / "results/essential/phase2b/schedulers"
BASES = (0.1, 0.03, 1.0)
LIB = {"warmup_cosine_hf": "hf", "cosine_annealing": "torch", "one_cycle": "torch", "linear_warmup_torch": "torch"}


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["schedulers"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = (c["sched"], c["resume"], c["steps"])
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = dict(c, high_risk=bool(c.get("high_risk")), id=f"sch_{c['sched']}_{'resume' if c['resume'] else 'cont'}_n{c['steps']}")
        seen[key] = d
        out.append(d)
    return out


def warmup_of(steps):
    return min(max(steps // 10, 1), steps)


def make(cond, base):
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=base)
    n, s = cond["steps"], cond["sched"]
    if s == "warmup_cosine_hf":
        from transformers import get_cosine_schedule_with_warmup
        sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_of(n), num_training_steps=n)
    elif s == "cosine_annealing":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n, eta_min=base / 10)
    elif s == "one_cycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=base, total_steps=n)
    else:
        sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=n)
    return opt, sch


def closed(sch):
    f = getattr(sch, "_get_closed_form_lr", None)
    return None if f is None else [float(x) for x in f()]


def run_seq(cond, base, resume):
    n = cond["steps"]
    total = 2 * n + 2
    opt, sch = make(cond, base)
    lrs, cf, notes = [float(sch.get_last_lr()[0])], [closed(sch)], []
    cut = max(n // 2, 1) if resume else None
    for t in range(1, total + 1):
        if cut is not None and t == cut + 1:
            state = sch.state_dict()
            ostate = opt.state_dict()
            opt, sch = make(cond, base)
            opt.load_state_dict(ostate)
            sch.load_state_dict(state)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt.step()
                sch.step()
        except Exception as exc:  # noqa: BLE001
            notes.append({"step": t, "stopped": f"{type(exc).__name__}: {str(exc)[:200]}"})
            break
        lrs.append(float(sch.get_last_lr()[0]))
        cf.append(closed(sch))
    return {"lrs": lrs, "closed": cf, "notes": notes}


def run(env):
    CACHE.mkdir(parents=True, exist_ok=True)
    lib = "hf" if env == "liger" else "torch"
    name = {"ka_main": "torch_lr_scheduler", "nightly": "torch_nightly_lr_scheduler", "liger": "hf_schedulers"}[env]
    res, t0 = {}, time.time()
    for cond in conditions():
        if LIB[cond["sched"]] != lib:
            continue
        for i, base in enumerate(BASES):
            rec = {}
            for var in ("base", "resumed") if cond["resume"] else ("base",):
                try:
                    rec[var] = {"status": "ok", **run_seq(cond, base, var == "resumed")}
                except Exception as exc:  # noqa: BLE001
                    rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
            res[(cond["id"], i)] = rec
    meta = {"candidate": name, "torch": torch.__version__, "seconds": round(time.time() - t0, 2), "env": env}
    if env == "liger":
        import transformers
        meta["transformers"] = transformers.__version__
    (CACHE / f"{name}.pkl").write_bytes(pickle.dumps({"meta": meta, "res": res}))
    st = defaultdict(int)
    for rec in res.values():
        for r in rec.values():
            st[r["status"]] += 1
    print(name, dict(st), meta)


def boundaries(cond, base, lrs):
    """documented values at phase boundaries: list of (step, expected, label)."""
    n, s = cond["steps"], cond["sched"]
    if s == "one_cycle":
        pct = 0.3
        div, fdiv = 25.0, 1e4
        peak = float(pct * n) - 1                     # OneCycleLR phase end (documented as pct_start * total_steps)
        out = [(0, base / div, "start = max_lr / div_factor"), (n - 1, base / div / fdiv, "end = initial_lr / final_div_factor")]
        if peak >= 0 and float(peak).is_integer():
            out.append((int(peak), base, "pct_start x total_steps - 1 = max_lr"))
        return out
    if s == "warmup_cosine_hf":
        w = warmup_of(n)
        return [(0, 0.0, "0 at step 0"), (w, base, "base at the end of warmup") if w < n else (w, 0.0, "warmup == total"),
                (n, 0.0, "0 at num_training_steps")]
    if s == "linear_warmup_torch":
        return [(0, 0.1 * base, "start_factor x base"), (n, base, "base at total_iters"), (2 * n, base, "constant after")]
    return [(0, base, "base at 0"), (n, base / 10, "eta_min at T_max")]


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"candidates": {}, "examples": []}
    for name, d in data.items():
        P = defaultdict(lambda: [0, 0])
        E = [0, 0]
        stops = []
        for (cid, i), rec in d["res"].items():
            cond, base = conds[cid], BASES[i]
            b = rec.get("base")
            if not b or b["status"] != "ok":
                out["examples"].append({"candidate": name, "condition": cid, "base": base, "error": b and b.get("reason")})
                continue
            lrs = b["lrs"]
            if b["notes"]:
                stops.append({"condition": cid, "base": base, "after_steps": len(lrs) - 1, "note": b["notes"][0]["stopped"][:120]})
            cfs = [(t, c[0]) for t, c in enumerate(b["closed"]) if c is not None]
            if cfs:
                bad = [(t, lrs[t], v) for t, v in cfs if abs(lrs[t] - v) > 1e-12 * (1 + abs(v))]
                P["P_chained_equals_closed_form"][0] += 1
                P["P_chained_equals_closed_form"][1] += int(bool(bad))
                if bad and len(out["examples"]) < 40:
                    out["examples"].append({"candidate": name, "condition": cid, "base": base, "property": "P_chained_equals_closed_form",
                                            "first": bad[0], "count": len(bad)})
            r = rec.get("resumed")
            if r and r["status"] == "ok":
                same = r["lrs"] == lrs
                P["P_resume_equals_continuous"][0] += 1
                P["P_resume_equals_continuous"][1] += int(not same)
                if not same and len(out["examples"]) < 40:
                    k = next((t for t, (a, c) in enumerate(zip(lrs, r["lrs"])) if a != c), min(len(lrs), len(r["lrs"])))
                    out["examples"].append({"candidate": name, "condition": cid, "base": base, "property": "P_resume_equals_continuous",
                                            "first_difference_step": k, "continuous": lrs[k:k + 3], "resumed": r["lrs"][k:k + 3]})
            for t, v, label in boundaries(cond, base, lrs):
                if t >= len(lrs):
                    P["P_phase_boundaries"][0] += 1
                    P["P_phase_boundaries"][1] += 1
                    out["examples"].append({"candidate": name, "condition": cid, "base": base, "property": "P_phase_boundaries",
                                            "label": label, "step": t, "why": "schedule stopped before this step"})
                    continue
                ok = abs(lrs[t] - v) <= 1e-12 * (1 + abs(v))
                P["P_phase_boundaries"][0] += 1
                P["P_phase_boundaries"][1] += int(not ok)
                if not ok and len(out["examples"]) < 60:
                    out["examples"].append({"candidate": name, "condition": cid, "base": base, "property": "P_phase_boundaries",
                                            "label": label, "step": t, "got": lrs[t], "expected": v})
            if name == "torch_nightly_lr_scheduler" and "torch_lr_scheduler" in data:
                o = data["torch_lr_scheduler"]["res"].get((cid, i), {}).get("base")
                if o and o["status"] == "ok":
                    E[0] += 1
                    E[1] += int(o["lrs"] != lrs)
        out["candidates"][name] = {"meta": d["meta"], "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()},
                                   "E_vs_2.10": f"{E[1]}/{E[0]}" if E[0] else None, "stopped_early": stops}
        print(name, out["candidates"][name]["P"], "E", out["candidates"][name]["E_vs_2.10"])
        for s in stops:
            print("   stopped:", s)
    for e in out["examples"]:
        print("  ", e)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.env) if a.stage == "run" else analyse()
