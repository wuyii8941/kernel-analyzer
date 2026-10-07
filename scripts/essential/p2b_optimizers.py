#!/usr/bin/env python3
"""2b family "optimizers" (docs/protocol_essential_bugs_phase2_20261007.md section 3; task book G2, G5): E, P and state
sequences.  F and mode-B FR stay closed until the reviewer's optimizer spec is delivered.

Conditions are the pre-registered coverage plan (results/essential/phase2b/coverage_plan.json, tier1.optimizers) with the
factor ``impl`` projected out -- every implementation path is a candidate and runs every condition -- plus the three
registered state sequences (section 3.1 registry: init -> normal -> zero-grad -> restore; init -> steps -> save/restore ->
continue; init -> skipped (non-finite) -> next normal).  Each run records, after every step, the parameters, every state
tensor and the learning rates.  Two parameter groups: group 0 takes the condition's weight decay, group 1 has lr x 2 and no
weight decay (parameter-group differences).

P variants (auxiliary runs of the same candidate, as the phase-1 P variants): ``mirror`` (maximize flipped, gradients
negated), ``uninterrupted`` (save/restore conditions without the save/restore), ``removed`` (non-finite skip without the
skipped step).  Gradients depend on (state, seed) only, so different optimizers see the same sequence.

    python scripts/essential/p2b_optimizers.py run --env ka_main [--only torch_compiled_step_cuda] [--limit N]
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_optimizers.py run --env liger
    python scripts/essential/p2b_optimizers.py analyse
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/optimizers"
OUT = ROOT / "results/essential/phase2b/optimizers"
SEEDS = (0, 1, 2)
SHAPES = [(5, 3), (7,), (2, 2, 3)]          # group 0: params 0, 1; group 1: param 2
N_STEPS = 8
EVENT = 4                                   # step index of the zero gradient / None / non-finite / restore event
TAU32 = 2.0 ** -12

OPTS = {
    "adamw": ("AdamW", dict(lr=1e-2, betas=(0.9, 0.99), eps=1e-8)),
    "adam": ("Adam", dict(lr=1e-2, betas=(0.9, 0.99), eps=1e-8)),
    "adam_amsgrad": ("Adam", dict(lr=1e-2, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)),
    "sgd_nesterov": ("SGD", dict(lr=1e-2, momentum=0.9, nesterov=True)),
    "sgd_dampening": ("SGD", dict(lr=1e-2, momentum=0.9, dampening=0.5)),
    "rmsprop_centered": ("RMSprop", dict(lr=1e-2, alpha=0.9, eps=1e-8, momentum=0.5, centered=True)),
    "adafactor": ("Adafactor", dict(lr=1e-2)),
}
FUSED_OK = {"Adam", "AdamW", "SGD"}          # torch 2.10: classes accepting fused=True


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["optimizers"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = (c["opt"], c["maximize"], c["weight_decay"], c["state"])
        if key in seen:
            seen[key]["planned_impls"].append(c["impl"])
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        cid = f"opt_{c['opt']}_{c['state']}_{'max' if c['maximize'] else 'min'}_wd{c['weight_decay']}"
        d = {"id": cid, "opt": c["opt"], "maximize": c["maximize"], "weight_decay": c["weight_decay"], "state": c["state"],
             "planned_impls": [c["impl"]], "high_risk": bool(c.get("high_risk")), "source": "coverage_plan"}
        seen[key] = d
        out.append(d)
    for opt in OPTS:                         # registered state sequence 1 for every optimizer, base settings
        out.append({"id": f"opt_{opt}_seq_zero_restore_min_wd0.0", "opt": opt, "maximize": False, "weight_decay": 0.0,
                    "state": "seq_zero_restore", "planned_impls": [], "high_risk": False, "source": "state_sequence"})
    return out


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def grad_sequence(cond, seed, variant):
    """list of steps; a step is a list (per parameter) of float64 arrays or None (grad = None)."""
    rng = _rng(f"opt-grads/{cond['state']}/{seed}")
    steps = [[rng.normal(0, 1, s) for s in SHAPES] for _ in range(N_STEPS)]
    st = cond["state"]
    if st == "cold":
        steps = steps[:3]
    elif st == "zero_grad":
        steps[EVENT] = [np.zeros(s) for s in SHAPES]
    elif st == "grad_none":
        steps[EVENT][1] = None
    elif st == "seq_zero_restore":
        steps[2] = [np.zeros(s) for s in SHAPES]
    elif st == "nonfinite_skip":
        if variant == "removed":
            del steps[EVENT]
        else:
            steps[EVENT][0][1, 2] = np.inf
    if variant == "mirror":
        steps = [[None if g is None else -g for g in gs] for gs in steps]
    return steps


def initial_params(seed):
    rng = _rng(f"opt-params/{seed}")
    return [rng.normal(0, 1, s) for s in SHAPES]


# ------------------------------------------------------------------------------------------------ candidates

def candidates_for(env):
    if env == "ka_main":
        out = [{"id": f"torch_{impl}_{dev}", "lib": "torch", "impl": impl, "device": dev}
               for dev in ("cpu", "cuda") for impl in ("for_loop", "foreach", "fused")]
        return out + [{"id": "torch_compiled_step_cuda", "lib": "torch", "impl": "compiled", "device": "cuda"}]
    if env == "nightly":                     # version check of findings only (not a registered candidate)
        return [{"id": f"nightly_torch_{impl}_{dev}", "lib": "torch", "impl": impl, "device": dev}
                for dev in ("cpu", "cuda") for impl in ("for_loop", "foreach", "fused")]
    if env == "liger":
        return [{"id": "bnb_adamw32bit_cuda", "lib": "bnb", "device": "cuda"},
                {"id": "torchao_adamw_fp32_cuda", "lib": "torchao", "device": "cuda"},
                {"id": "hf_adafactor_cuda", "lib": "hf", "device": "cuda"}]
    raise KeyError(env)


def make_optimizer(cand, params, cond, maximize):
    cls_name, kw = OPTS[cond["opt"]]
    kw = dict(kw, weight_decay=cond["weight_decay"])
    groups = [{"params": params[:2]}, {"params": params[2:], "lr": 2 * kw["lr"], "weight_decay": 0.0}]
    lib, impl = cand["lib"], cand.get("impl")
    if lib == "torch":
        if maximize:
            kw["maximize"] = True
        if impl == "fused":
            if cls_name not in FUSED_OK:
                raise NotImplementedError(f"torch.optim.{cls_name} has no fused implementation")
            kw["fused"] = True
        elif impl == "foreach":
            kw["foreach"] = True
        elif impl == "for_loop":
            kw["foreach"] = False
        return getattr(torch.optim, cls_name)(groups, **kw)
    if maximize:
        raise NotImplementedError(f"{lib}: no maximize option")
    if lib == "bnb":
        import bitsandbytes as bnb
        if cls_name != "AdamW" or kw.get("amsgrad"):
            raise NotImplementedError("bnb candidate: AdamW32bit without amsgrad only")
        return bnb.optim.AdamW32bit(groups, **{x: kw[x] for x in ("lr", "betas", "eps", "weight_decay")})
    if lib == "torchao":
        import torchao.optim as ao
        if cls_name != "AdamW" or kw.get("amsgrad"):
            raise NotImplementedError("torchao candidate: _AdamW (fp32 state) without amsgrad only")
        return ao._AdamW(groups, **{x: kw[x] for x in ("lr", "betas", "eps", "weight_decay")})
    if lib == "hf":
        from transformers.optimization import Adafactor
        if cls_name != "Adafactor":
            raise NotImplementedError("HF candidate: Adafactor only")
        return Adafactor(groups, lr=kw["lr"], weight_decay=kw["weight_decay"], scale_parameter=False, relative_step=False,
                         warmup_init=False)
    raise KeyError(lib)


def _np(v):
    return v.detach().double().cpu().numpy().copy() if torch.is_tensor(v) else v


def snapshot(opt, params):
    pid = {id(p): i for i, p in enumerate(params)}
    st = {}
    for g in opt.param_groups:
        for p in g["params"]:
            st[pid[id(p)]] = {k: _np(v) for k, v in opt.state.get(p, {}).items()}
    return {"params": [_np(p) for p in params], "lr": [float(g["lr"]) for g in opt.param_groups], "state": st}


class Runner:
    """one optimizer instance; ``step`` goes through torch.compile for the compiled candidate."""

    def __init__(self, cand, params, cond, maximize):
        self.cand, self.params, self.cond, self.maximize = cand, params, cond, maximize
        self.opt = make_optimizer(cand, params, cond, maximize)
        if cand.get("impl") == "compiled":
            self.opt.step = torch.compile(self.opt.step)

    def reload(self):
        buf = io.BytesIO()
        torch.save(self.opt.state_dict(), buf)
        buf.seek(0)
        self.__init__(self.cand, self.params, self.cond, self.maximize)
        self.opt.load_state_dict(torch.load(buf, weights_only=False))


def run_one(cand, cond, seed, variant):
    device, dtype = cand["device"], torch.float32
    maximize = cond["maximize"] != (variant == "mirror")
    params = [torch.tensor(a, dtype=dtype, device=device, requires_grad=True) for a in initial_params(seed)]
    run = Runner(cand, params, cond, maximize)
    scaler = torch.amp.GradScaler(device) if cond["state"] == "nonfinite_skip" else None
    restore_at = {"save_restore": EVENT, "seq_zero_restore": EVENT}.get(cond["state"]) if variant != "uninterrupted" else None
    traj, notes = [snapshot(run.opt, params)], []
    for t, gs in enumerate(grad_sequence(cond, seed, variant)):
        if restore_at is not None and t == restore_at:
            run.reload()
        for p, g in zip(params, gs):
            if g is None:
                p.grad = None
            else:
                gt = torch.tensor(g, dtype=dtype, device=device)
                p.grad = scaler.scale(gt) if scaler is not None else gt
        if scaler is not None:
            scaler.step(run.opt)
            notes.append({"step": t, "scale": float(scaler.get_scale())})
            scaler.update()
        else:
            run.opt.step()
        traj.append(snapshot(run.opt, params))
    return traj, notes


def variants_for(cond):
    v = ["base", "mirror"]
    if cond["state"] in ("save_restore", "seq_zero_restore"):
        v.append("uninterrupted")
    if cond["state"] == "nonfinite_skip":
        v.append("removed")
    return v


def run(env, only=None, limit=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    conds = conditions()[:limit] if limit else conditions()
    for cand in candidates_for(env):
        if only and cand["id"] not in only:
            continue
        path = CACHE / f"{cand['id']}{'_pilot' if limit else ''}.pkl"
        if path.exists():
            continue
        res, t0, slowest = {}, time.time(), (0.0, None)
        for cond in conds:
            for seed in SEEDS:
                rec = {}
                for var in variants_for(cond):
                    t1 = time.time()
                    if cand.get("impl") == "compiled":
                        torch._dynamo.reset()
                        torch._dynamo.utils.counters.clear()
                    try:
                        traj, notes = run_one(cand, cond, seed, var)
                        rec[var] = {"status": "ok", "traj": traj, "notes": notes}
                        if cand.get("impl") == "compiled":
                            c = torch._dynamo.utils.counters
                            rec[var]["dynamo"] = {"unique_graphs": int(c["stats"].get("unique_graphs", 0)),
                                                  "graph_breaks": int(sum(c["graph_break"].values())),
                                                  "calls_captured": int(c["stats"].get("calls_captured", 0))}
                    except NotImplementedError as exc:
                        rec[var] = {"status": "unsupported", "reason": str(exc)}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    dt = time.time() - t1
                    if dt > slowest[0]:
                        slowest = (dt, f"{cond['id']}/{seed}/{var}")
                res[(cond["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
                "slowest": [round(slowest[0], 2), slowest[1]], "conditions": len(conds), "env": env}
        if cand["lib"] != "torch":
            import importlib.metadata as md
            meta["lib_version"] = md.version({"bnb": "bitsandbytes", "torchao": "torchao", "hf": "transformers"}[cand["lib"]])
        path.write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = {}
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] = st.get(r["status"], 0) + 1
        print(cand["id"], st, meta["seconds"], "s, slowest", meta["slowest"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.stage == "run":
        run(a.env, a.only, a.limit)
    else:
        import p2b_optimizers_analyse
        p2b_optimizers_analyse.main()
