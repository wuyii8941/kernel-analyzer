#!/usr/bin/env python3
"""2b FR in mode A (protocol v2 section 3: F and mode-B FR stay closed until a family's spec is delivered): the tool
evaluates the Triton kernels of a compiled candidate on their own TTIR, e_num = K - K_R.  No task semantics are checked.

Family "optimizers": one compiled ``opt.step()`` (Inductor, default foreach path under compile) from a warm state -- the
parameters, state and gradients of step 3 of the torch for_loop CUDA trajectory of the same condition and seed -- for the
distinct (optimizer, maximize, weight decay) settings of the optimizer conditions.  Outputs: every parameter and every state
tensor.  Verdict per output: elements with a complete finite K_R whose K lies outside [K_R lo, K_R hi] widened by
τ32 = 2^-12 (1 + |K_R|).

    python scripts/essential/p2b_fr_modeA.py optimizers
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import p2b_optimizers as O  # noqa: E402

OUT = O.ROOT / "results/essential/phase2b"
WARM_STEP = 2                      # snapshot index after two eager steps; the compiled step is step 3


def optimizer_cases():
    seen, out = set(), []
    for c in O.conditions():
        key = (c["opt"], c["maximize"], c["weight_decay"])
        if c["state"] == "cold" and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def optimizer_case(cond, ref):
    st = {}
    cand = {"lib": "torch", "impl": None, "device": "cuda"}

    def load(seed):
        snap = ref[(cond["id"], seed)]["base"]["traj"][WARM_STEP]
        grads = O.grad_sequence(cond, seed, "base")[WARM_STEP]
        with torch.no_grad():
            for i, p in enumerate(st["params"]):
                p.copy_(torch.as_tensor(snap["params"][i], dtype=torch.float32))
                p.grad = torch.as_tensor(grads[i], dtype=torch.float32, device="cuda")
                for k, v in st["opt"].state[p].items():
                    if torch.is_tensor(v):
                        v.copy_(torch.as_tensor(np.asarray(snap["state"][i][k]), dtype=v.dtype))

    def setup():
        torch._dynamo.reset()
        params = [torch.zeros(s, dtype=torch.float32, device="cuda", requires_grad=True) for s in O.SHAPES]
        opt = O.make_optimizer(cand, params, cond, cond["maximize"])
        st.update(params=params, opt=opt, step=torch.compile(opt.step))
        for p in params:
            p.grad = torch.zeros_like(p)
        st["step"]()                    # state initialisation graph
        load(0)
        st["step"]()                    # steady-state graph, compiled outside the recorder
        torch.cuda.synchronize()

    def inputs(seed):
        load(seed)
        return {"_seed": seed}

    def launch(_t):
        st["step"]()
        outs = {f"param{i}": p for i, p in enumerate(st["params"])}
        for i, p in enumerate(st["params"]):
            for k, v in st["opt"].state[p].items():
                if torch.is_tensor(v) and v.is_floating_point() and v.is_cuda and v.dim() > 0:
                    outs[f"state{i}.{k}"] = v
        return outs

    return setup, inputs, launch


def verdict(keep):
    out = {}
    for name, rows in keep.items():
        n_ok = n_out = 0
        worst = 0.0
        for r in rows:
            ok = np.asarray(r["ok"], bool)
            k, lo, hi = (np.asarray(r[x], np.float64)[ok] for x in ("k", "r_lo", "r_hi"))
            mid = 0.5 * (lo + hi)
            tol = O.TAU32 * (1 + np.abs(mid))
            dist = np.maximum(lo - k, k - hi).clip(min=0)
            n_ok += int(ok.sum())
            n_out += int((dist > tol).sum())
            if ok.any():
                worst = max(worst, float((dist / tol).max()))
        out[name] = {"ok_elements": n_ok, "beyond_tau32": n_out, "max_dist_over_tau32": worst}
    return out


def run_optimizers():
    ref = pickle.loads((O.CACHE / "torch_for_loop_cuda.pkl").read_bytes())["res"]
    res, t0 = {}, time.time()
    for cond in optimizer_cases():
        try:
            setup, inputs, launch = optimizer_case(cond, ref)
            rep, keep = common.fr_run(f"optimizers/{cond['id']}/torch_compiled_step_cuda", setup, inputs, launch,
                                      lambda _t: None)
            res[cond["id"]] = {"status": "ok", "verdict": verdict(keep),
                               "notes": {k: rep.get(k) for k in (
                                   "outputs_not_written_by_triton", "outputs_binding_not_established",
                                   "outputs_at_address_of_another_recorded_storage", "outputs_modified_after_last_triton_write",
                                   "outputs_whose_writing_programs_aborted", "ttir_coverage_complete", "tool_version")},
                               "launches": len(rep.get("launches") or []),
                               "special": {k: v.get("special_values") for k, v in rep.get("outputs", {}).items()},
                               "complete_fraction": {k: v["reference_classes"]["finite_complete_fraction"]
                                                     for k, v in rep.get("outputs", {}).items()},
                               "seconds": rep.get("seconds"), "timing": rep.get("timing_seconds")}
        except Exception as exc:  # noqa: BLE001
            res[cond["id"]] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400],
                               "trace": traceback.format_exc()[-1500:]}
        r = res[cond["id"]]
        if r["status"] == "ok":
            tot = sum(v["ok_elements"] for v in r["verdict"].values())
            bad = sum(v["beyond_tau32"] for v in r["verdict"].values())
            print(cond["id"], f"launches {r['launches']} ok elements {tot} beyond τ32 {bad} "
                              f"not-Triton {r['notes']['outputs_not_written_by_triton']} "
                              f"unbound {r['notes']['outputs_binding_not_established']}", flush=True)
        else:
            print(cond["id"], r["reason"], flush=True)
    path = OUT / "optimizers" / "fr_modeA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "cases": res}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    {"optimizers": run_optimizers}[sys.argv[1]]()
