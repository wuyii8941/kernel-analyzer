#!/usr/bin/env python3
"""2b FR in mode A (protocol v2 section 3: F and mode-B FR stay closed until a family's spec is delivered): the tool
evaluates the Triton kernels of a compiled candidate on their own TTIR, e_num = K - K_R.  No task semantics are checked.

Family "optimizers": one compiled ``opt.step()`` (Inductor, default foreach path under compile) from a warm state -- the
parameters, state and gradients of step 3 of the torch for_loop CUDA trajectory of the same condition and seed -- for the
distinct (optimizer, maximize, weight decay) settings of the optimizer conditions.  Outputs: every parameter and every state
tensor.  Verdict per output: elements with a complete finite K_R whose K lies outside [K_R lo, K_R hi] widened by
τ32 = 2^-12 (1 + |K_R|).

Family "attention": flex_attention and the compiled manual attention (Inductor), forward and backward, every condition
except head_dim 72 for flex (2.10 fails to compile it: #164931); outputs out, dq, dk, dv.  The block mask is built in setup.

    python scripts/essential/p2b_fr_modeA.py optimizers
    python scripts/essential/p2b_fr_modeA.py attention
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
import p2b_attention as A  # noqa: E402
import p2b_normalization as NM  # noqa: E402
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


def attention_case(cand_id, cond):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    st = {}
    a = A.allowed(cond)
    scale = cond["scale"] if cond["scale"] is not None else 1.0 / cond["head_dim"] ** 0.5

    def setup():
        torch._dynamo.reset()
        mask_t = None if cond["mask"] == "none" else torch.as_tensor(a, device="cuda")
        if cand_id == "flex_attention_float32":
            bm = None
            if mask_t is not None:
                def mask_mod(b, h, qi, ki):
                    return mask_t[b, qi, ki]
                lq, lk = A.LENS[cond["q_len_k_len"]]
                bm = create_block_mask(mask_mod, A.B, None, lq, lk, device="cuda")
            f = torch.compile(flex_attention, dynamic=False)
            st["fn"] = lambda q, k, v: f(q, k, v, block_mask=bm, scale=cond["scale"], enable_gqa=cond["gqa"])
        else:
            f = torch.compile(A.manual_attention, dynamic=False)
            st["fn"] = lambda q, k, v: f(q, k, v, mask_t, scale, q.shape[1] // k.shape[1])
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        d = A.make_inputs(cond, seed, "base")
        t = {x: torch.tensor(d[x], dtype=torch.float32, device="cuda", requires_grad=x != "g") for x in ("q", "k", "v", "g")}
        t["_seed"] = seed
        return t

    def launch(t):
        for x in ("q", "k", "v"):
            t[x].grad = None
        o = st["fn"](t["q"], t["k"], t["v"])
        o.backward(t["g"])
        return {"out": o, "dq": t["q"].grad, "dk": t["k"].grad, "dv": t["v"].grad}

    return setup, inputs, launch


def run_attention():
    res, t0 = {}, time.time()
    for cand_id in ("flex_attention_float32", "inductor_attention_float32"):
        for cond in A.conditions():
            key = f"{cand_id}/{cond['id']}"
            if cand_id == "flex_attention_float32" and cond["head_dim"] == 72:
                res[key] = {"status": "not run", "reason": "2.10 does not compile flex decoding for head_dim 72 (#164931)"}
                continue
            try:
                setup, inputs, launch = attention_case(cand_id, cond)
                rep, keep = common.fr_run(f"attention/{key}", setup, inputs, launch, lambda _t: None)
                res[key] = {"status": "ok", "verdict": verdict(keep),
                            "notes": {k: rep.get(k) for k in (
                                "outputs_not_written_by_triton", "outputs_binding_not_established",
                                "outputs_at_address_of_another_recorded_storage", "outputs_modified_after_last_triton_write",
                                "outputs_whose_writing_programs_aborted", "ttir_coverage_complete", "tool_version")},
                            "launches": [l["kernel"][:60] for l in rep.get("launches") or []],
                            "special": {k: v.get("special_values") for k, v in rep.get("outputs", {}).items()},
                            "complete_fraction": {k: v["reference_classes"]["finite_complete_fraction"]
                                                  for k, v in rep.get("outputs", {}).items()},
                            "seconds": rep.get("seconds"), "timing": rep.get("timing_seconds")}
            except Exception as exc:  # noqa: BLE001
                res[key] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400],
                            "trace": traceback.format_exc()[-1500:]}
            r = res[key]
            if r["status"] == "ok":
                v = r["verdict"]
                print(key, "ok-el", {k: x["ok_elements"] for k, x in v.items()}, "beyond", sum(x["beyond_tau32"] for x in v.values()),
                      "complete", {k: round(x, 2) for k, x in r["complete_fraction"].items()},
                      "not-Triton", r["notes"]["outputs_not_written_by_triton"], flush=True)
            else:
                print(key, r["status"], r.get("reason", "")[:200], flush=True)
    path = OUT / "attention" / "fr_modeA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "cases": res}, indent=1, default=str) + "\n")


def normalization_case(dtype, cond):
    st = {}
    dt = getattr(torch, dtype)

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(NM.norm_fn(cond), dynamic=False)
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        d = NM.make_inputs(cond, seed, "base")
        t = {k: torch.tensor(d[k], dtype=dt, device="cuda", requires_grad=k in ("x", "w", "b")) for k in ("x", "w", "b", "g")}
        t["rm"], t["rv"] = (torch.tensor(d[k], dtype=dt, device="cuda") for k in ("rm", "rv"))
        t["_seed"] = seed
        return t

    def launch(t):
        for k in ("x", "w", "b"):
            t[k].grad = None
        y = st["fn"](t["x"], t["w"], t["b"], t["rm"], t["rv"])
        y.backward(t["g"])
        out = {"y": y, "dx": t["x"].grad}
        if cond["affine"]:
            out["dw"] = t["w"].grad
            if cond["op"] != "rms_norm":
                out["db"] = t["b"].grad
        if cond["op"] == "batch_norm_train":
            out["rm"], out["rv"] = t["rm"], t["rv"]
        return out

    return setup, inputs, launch


def kr_vs_reference(keep, ref_outputs, seed_of_row=None):
    """E applied to K_R: max |mid(K_R) - K_ref| / τ32(1 + |K_ref|) over elements with a complete finite K_R, and the count
    beyond τ32.  K_ref is the float64 eager run on the same inputs; K_R is the exact evaluation of the candidate's own TTIR,
    so a K_R close to K_ref while K is not attributes the deviation to rounding (e_num)."""
    out = {}
    for name, rows in keep.items():
        n_bad = n = 0
        worst = 0.0
        for r in rows:
            ref = ref_outputs.get(r["seed"], {}).get(name)
            if ref is None:
                continue
            ok = np.asarray(r["ok"], bool)
            mid = 0.5 * (np.asarray(r["r_lo"], float) + np.asarray(r["r_hi"], float))
            refv = np.asarray(ref, float).reshape(-1)
            if refv.size != mid.size:
                continue
            tol = O.TAU32 * (1 + np.abs(refv))
            d = np.abs(mid - refv)
            n += int(ok.sum())
            n_bad += int((ok & (d > tol)).sum())
            if ok.any():
                worst = max(worst, float((d / tol)[ok].max()))
        out[name] = {"elements": n, "kr_beyond_tau32_vs_float64": n_bad, "max_kr_dev_over_tau32": worst}
    return out


def float64_on_rounded(cond, seed, dt):
    d = NM.make_inputs(cond, seed, "base")
    r = {k: torch.tensor(d[k], dtype=dt).double() for k in ("x", "w", "b", "rm", "rv", "g")}
    x, w, b = (r[k].clone().requires_grad_(True) for k in ("x", "w", "b"))
    y = NM.norm_fn(cond)(x, w, b, r["rm"], r["rv"])
    y.backward(r["g"])
    out = {"y": y.detach().numpy(), "dx": x.grad.numpy()}
    if cond["affine"]:
        out["dw"] = w.grad.numpy()
        if cond["op"] != "rms_norm":
            out["db"] = b.grad.numpy()
    if cond["op"] == "batch_norm_train":
        out["rm"], out["rv"] = r["rm"].numpy(), r["rv"].numpy()
    return out


def run_normalization():
    res, t0 = {}, time.time()
    ref = pickle.loads((NM.CACHE / "eager_cpu_float64.pkl").read_bytes())["res"]
    for dtype in ("float32", "bfloat16"):
        for cond in NM.conditions():
            key = f"inductor_cuda_{dtype}/{cond['id']}"
            try:
                setup, inputs, launch = normalization_case(dtype, cond)
                rep, keep = common.fr_run(f"normalization/{key}", setup, inputs, launch, lambda _t: None)
                if dtype == "float32":
                    refs = {sd: ref[(cond["id"], sd)]["base"]["outputs"] for sd in (0, 1, 2)}
                else:                                         # float64 eager on the bfloat16-rounded inputs the candidate received
                    refs = {sd: float64_on_rounded(cond, sd, torch.bfloat16) for sd in (0, 1, 2)}
                res[key] = {"status": "ok", "verdict": verdict(keep), "kr_vs_float64_eager": kr_vs_reference(keep, refs),
                            "notes": {k: rep.get(k) for k in (
                                "outputs_not_written_by_triton", "outputs_binding_not_established",
                                "outputs_at_address_of_another_recorded_storage", "outputs_modified_after_last_triton_write",
                                "outputs_whose_writing_programs_aborted", "ttir_coverage_complete", "tool_version")},
                            "launches": [l["kernel"][:60] for l in rep.get("launches") or []],
                            "special": {k: v.get("special_values") for k, v in rep.get("outputs", {}).items()},
                            "complete_fraction": {k: v["reference_classes"]["finite_complete_fraction"]
                                                  for k, v in rep.get("outputs", {}).items()},
                            "seconds": rep.get("seconds")}
            except Exception as exc:  # noqa: BLE001
                res[key] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-1500:]}
            r = res[key]
            if r["status"] == "ok":
                v = r["verdict"]
                kr = r["kr_vs_float64_eager"]
                print(key, "ok-el", sum(x["ok_elements"] for x in v.values()), "e_num beyond", sum(x["beyond_tau32"] for x in v.values()),
                      "| K_R vs f64 beyond", sum(x["kr_beyond_tau32_vs_float64"] for x in kr.values()),
                      "max", round(max([x["max_kr_dev_over_tau32"] for x in kr.values()] or [0]), 3),
                      "complete", {k: round(x, 2) for k, x in r["complete_fraction"].items()},
                      "not-Triton", r["notes"]["outputs_not_written_by_triton"], "unbound", r["notes"]["outputs_binding_not_established"],
                      "modified", r["notes"]["outputs_modified_after_last_triton_write"], flush=True)
            else:
                print(key, r["status"], r.get("reason", "")[:200], flush=True)
    path = OUT / "normalization" / "fr_modeA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "cases": res}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    {"optimizers": run_optimizers, "attention": run_attention, "normalization": run_normalization}[sys.argv[1]]()
