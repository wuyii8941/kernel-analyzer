#!/usr/bin/env python3
"""FR in mode B (closure task book section 7: K_R against the phase-2 specs) for the Triton candidates of the families
with a delivered spec.  The tool (kernel_analyzer.check.run through common.fr_run) evaluates the candidate's own TTIR on
the inputs it received (K_R) and, given f, reports e_num = K - K_R and e_sem = K_R - f through the frozen statistics layer.

Semantic judgment (protocol v3 section 7, "有 TTIR 时 K_R 对 f_r 精确比较"): element-wise, over the elements with a
complete finite K_R and a defined f, K_R and f_r differ as real numbers iff their enclosures are disjoint.  Both are
rigorous except where a declared approximation enters (spec_base_ops.gelu_erf uses mpmath erf, declared; the TTIR mapping's
transcendental bounds are declared in ttir_mapping): those elements are reported as conditional.  Elements where the spec
declines (undefined / not established) carry NaN and are excluded and counted.

Cases reuse scripts/essential/p2b_fr_modeA.py (same compiled programs, same inputs) and the spec evaluations of
scripts/closure/f_eval_2b.py; outputs without f (most gradients) are not returned to the tool here (mode A covered their
e_num).  Output: results/closure/fr_modeB/<family>.json.

    python scripts/closure/fr_modeB.py matmul_linear reductions ...
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("specs/phase2", "specs/phase1", "scripts/essential", "scripts/closure"):
    sys.path.insert(0, str(ROOT / p))
import contract_v3 as CV  # noqa: E402
import f_eval_2b as FE  # noqa: E402
from f_eval import bounds, norm_spec, spec_arrays  # noqa: E402

# closure records (tool 2.3) stay in results/closure/fr_modeB; tool-3.0 reruns go to results/general/
OUT = ROOT / ("results/general/fr_modeB_v3" if __import__("os").environ.get("KA_ACCUMULATION", "exact") == "exact"
              else "results/general/fr_modeB_gamma")
DTYPES = {}
TAU32 = 2.0 ** -12
NOTE_KEYS = ("outputs_not_written_by_triton", "outputs_binding_not_established", "outputs_at_address_of_another_recorded_storage",
             "outputs_modified_after_last_triton_write", "outputs_whose_writing_programs_aborted", "ttir_coverage_complete",
             "tool_version")


# ------------------------------------------------------------------------------------------------ helpers

def filtered(case, keys_of_seed):
    """wrap (setup, inputs, launch): launch returns only the outputs that have f, and remembers their shapes."""
    setup, inputs, launch = case
    shapes = {}

    def launch2(t):
        outs = launch(t)
        keys = keys_of_seed(t)
        sel = {k: v for k, v in outs.items() if k in keys and v is not None}
        shapes.clear()
        shapes.update({k: tuple(v.shape) for k, v in sel.items()})
        DTYPES.update({k: str(v.dtype).replace("torch.", "") for k, v in sel.items()})
        return sel
    return setup, inputs, launch2, shapes


def make_spec(fas, shapes):
    def spec(t):
        fa = fas[t["_seed"]]
        out = {}
        for k, shp in shapes.items():
            lo, hi = (np.asarray(a, np.float64).reshape(shp) for a in fa[k])
            out[k] = (lo, hi)
        return out
    return spec


def elementwise(keep, fas, name):
    """element-wise K_R vs f over the ok elements: (certified disjoint, max relative gap)."""
    n_sem, worst = 0, 0.0
    for r in keep[name]:
        f_lo, f_hi = (np.asarray(a, np.float64).ravel() for a in fas[r["seed"]][name])
        ok = np.asarray(r["ok"], bool) & np.isfinite(f_lo)
        lo, hi = np.asarray(r["r_lo"], np.float64), np.asarray(r["r_hi"], np.float64)
        with np.errstate(invalid="ignore"):
            gap = np.maximum(lo - f_hi, f_lo - hi)
        sem = ok & (gap > 0)
        n_sem += int(sem.sum())
        if sem.any():
            worst = max(worst, float((gap / (1 + np.abs(0.5 * (f_lo + f_hi))))[sem].max()))
    return n_sem, worst


def summarise(rep, keep, fas, fas_alt=None):
    """e_num beyond τ32 and the element-wise semantic comparison K_R vs f, per output; fas_alt: f with the hyperparameters
    rounded to float32 as the kernel received them (attribution of certified differences)."""
    res = {}
    for name, rows in keep.items():
        n_ok = n_num = n_sem = n_undef = n_knf = 0
        worst_num = worst_sem = 0.0
        for r in rows:
            fa = fas[r["seed"]][name]
            f_lo, f_hi = (np.asarray(a, np.float64).ravel() for a in fa)
            ok = np.asarray(r["ok"], bool)
            k, lo, hi = (np.asarray(r[x], np.float64) for x in ("k", "r_lo", "r_hi"))
            n_undef += int(np.isnan(f_lo).sum())
            mid = 0.5 * (lo + hi)
            dist = np.maximum(lo - k, k - hi).clip(min=0)
            tol = TAU32 * (1 + np.abs(mid))
            n_ok += int(ok.sum())
            kfin = np.isfinite(k)
            n_num += int((ok & kfin & (dist > tol)).sum())
            n_knf += int((ok & ~kfin).sum())
            with np.errstate(invalid="ignore"):
                gap = np.maximum(lo - f_hi, f_lo - hi)            # > 0: enclosures disjoint
            sem = ok & (gap > 0)
            n_sem += int(sem.sum())
            if (ok & kfin).any():
                worst_num = max(worst_num, float((dist / tol)[ok & kfin].max()))
                fm = 0.5 * (f_lo + f_hi)
                worst_sem = max(worst_sem, float(np.where(sem, gap / (1 + np.abs(fm)), 0).max()))
        out = rep.get("outputs", {}).get(name, {})
        sem_rec = out.get("semantic") or {}
        rules = sem_rec.get("rules")
        rules = {r.get("rule", str(i)): r for i, r in enumerate(rules)} if isinstance(rules, list) else (rules or {})
        from kernel_analyzer.measure import reference_quality
        quality = reference_quality(rows, DTYPES.get(name, "float32"), 0.125)
        res[name] = {"ok_elements": n_ok, "e_num_beyond_tau32": n_num, "max_e_num_over_tau32": worst_num,
                     "reference_quality": quality,
                     "k_nonfinite_with_finite_kr_column4": n_knf,
                     "e_sem_certified_elements": n_sem, "max_e_sem_gap_relative": worst_sem, "f_undefined_elements": n_undef,
                     "mixed_non_triton_sources": out.get("depends_on_non_triton_intermediates") or [],
                     "e_sem_mean_effect": {k: CV.statistical_judgment(v) for k, v in rules.items() if isinstance(v, dict)}
                     or CV.statistical_judgment(sem_rec) if sem_rec else None,
                     "e_sem_elementwise_tool": out.get("semantic_elementwise"),
                     "special_values": out.get("special_values"),
                     "complete_fraction": (out.get("reference_classes") or {}).get("finite_complete_fraction")}
        if fas_alt is not None and name in fas_alt[0]:
            n_alt, w_alt = elementwise(keep, fas_alt, name)
            res[name]["e_sem_certified_with_received_hyperparameters"] = n_alt
            res[name]["max_gap_with_received_hyperparameters"] = w_alt
    return res


def run_case(key, case, fas, keys_of_seed=None, fas_alt=None):
    import common
    keys_of_seed = keys_of_seed or (lambda t: set(fas[t["_seed"]]))
    setup, inputs, launch, shapes = filtered(case, keys_of_seed)
    t0 = time.time()
    rep, keep = common.fr_run(key, setup, inputs, launch, make_spec(fas, shapes))
    return {"status": "ok", "outputs": summarise(rep, keep, fas, fas_alt), "notes": {k: rep.get(k) for k in NOTE_KEYS},
            "launches": len(rep.get("launches") or []), "seconds": round(time.time() - t0, 1),
            "timing": rep.get("timing_seconds")}


def guarded(key, fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-1500:]}


def precompute(job, args):
    with ProcessPoolExecutor(max_workers=24) as ex:
        return list(ex.map(job, args))


# ------------------------------------------------------------------------------------------------ families

def run_basic(family):
    import p2b_basic as BA
    import p2b_fr_modeA as MA
    job = {"matmul_linear": FE.matmul_job, "reductions": FE.reduction_job, "activations": FE.activation_job,
           "gather_layout": FE.gather_job}[family]
    conds = BA.FAMILIES[family].conditions()
    specs = {}
    for cid, seed, fa, status in precompute(job, [(c["id"], s) for c in conds for s in (0, 1, 2)]):
        if fa is not None:
            fa.pop("da_set", None)
        specs[(cid, seed)] = (fa, status)
    res = {}
    for c in conds:
        key = f"inductor_cuda_float32/{c['id']}"
        st = [specs[(c["id"], s)] for s in (0, 1, 2)]
        if any(fa is None for fa, _ in st):
            res[key] = {"status": "no f", "reason": st[0][1]}
            continue
        fas = {s: st[s][0] for s in (0, 1, 2)}
        alt = None
        if family == "activations" and c["op"] in ("gelu_erf", "gelu_tanh", "geglu"):
            alt = {s: gelu_rounded_constants(c, s) for s in (0, 1, 2)}
        res[key] = guarded(key, lambda: run_case(f"{family}/{key}", MA.basic_case(family, c), fas, fas_alt=alt))
        report_line(key, res[key])
    return res


def gelu_rounded_constants(c, seed):
    """attribution only (not f): the GELU formulas with the constants as float32 literals, as Inductor's TTIR holds them
    (sqrt(2/pi), 0.044715, 1/sqrt(2) rounded to float32; 0.5 exact)."""
    import p2b_basic as BA
    import rigorous_mp as R
    from spec_base_ops import I
    inp = BA.FAMILIES["activations"].inputs(c, seed, "base")
    a, b = (np.asarray(inp[k], np.float64).ravel() for k in ("a", "b"))
    c1, c2, c3 = f32(0.7978845608028654), f32(0.044715), f32(0.7071067811865476)
    out = []
    for i, x in enumerate(a):
        x = Fr(float(x))
        if c["op"] == "gelu_tanh":
            v = I(x) * (1 + R.tanh(I(x + c2 * x ** 3) * I(c1))) / 2
        else:
            v = I(x) * (1 + R.erf_approx(I(x * c3))) / 2
            if c["op"] == "geglu":
                v = v * I(Fr(float(b[i])))
        out.append(v)
    return {"out": spec_arrays(out)}


def run_normalization():
    import p2b_fr_modeA as MA
    import p2b_normalization as NM
    res = {}
    for c in NM.conditions():
        key = f"inductor_cuda_float32/{c['id']}"
        try:
            fas = {s: {k: spec_arrays(v) for k, v in norm_spec(c, NM.make_inputs(c, s, "base")).items()} for s in (0, 1, 2)}
            alt = {s: {k: spec_arrays(v) for k, v in norm_spec_f32_hp(c, NM.make_inputs(c, s, "base")).items()} for s in (0, 1, 2)}
        except Exception as exc:  # noqa: BLE001
            res[key] = {"status": "no f", "reason": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        res[key] = guarded(key, lambda: run_case(f"normalization/{key}", MA.normalization_case("float32", c), fas, fas_alt=alt))
        report_line(key, res[key])
    return res


def f32(v):
    return Fr(float(np.float32(float(v))))


def norm_spec_f32_hp(cond, inp):
    """norm_spec with eps and momentum as the float32 kernel receives them (attribution only; f itself is norm_spec)."""
    import spec_normalization as SN
    eps = f32(1e-5)
    x = np.asarray(inp["x"], dtype=np.float64)
    w = [Fr(float(v)) for v in inp["w"]] if cond["affine"] else None
    b = [Fr(float(v)) for v in inp["b"]] if cond["affine"] else None
    op, out = cond["op"], {}
    if op in ("layer_norm", "rms_norm"):
        out["y"] = [SN.layer_norm([Fr(float(v)) for v in x[n, r]], w, b, eps) if op == "layer_norm"
                    else SN.rms_norm([Fr(float(v)) for v in x[n, r]], w, eps) for n in range(x.shape[0]) for r in range(x.shape[1])]
    elif op == "group_norm":
        out["y"] = [SN.group_norm([[Fr(float(v)) for v in x[n, c]] for c in range(x.shape[1])], 3, w, b, eps) for n in range(x.shape[0])]
    else:
        st = SN.BNState(x.shape[1])
        st.running_mean = [Fr(float(v)) for v in inp["rm"]]
        st.running_var = [Fr(float(v)) for v in inp["rv"]]
        xl = [[[Fr(float(v)) for v in x[n, c]] for c in range(x.shape[1])] for n in range(x.shape[0])]
        out["y"] = SN.batch_norm(xl, st, w, b, eps, momentum=f32(0.1), training=op == "batch_norm_train")
        if op == "batch_norm_train":
            out["rm"], out["rv"] = st.running_mean, st.running_var
    return out


def run_attention():
    import p2b_attention as A
    import p2b_fr_modeA as MA
    conds = A.conditions()
    specs = {(cid, seed): fa for cid, seed, _r, fa, _u in
             precompute(FE.attention_job, [(c["id"], s, "float32") for c in conds for s in (0, 1, 2)])}
    res = {}
    for cand in ("flex_attention_float32", "inductor_attention_float32"):
        for c in conds:
            key = f"{cand}/{c['id']}"
            if cand == "flex_attention_float32" and c["head_dim"] == 72:
                res[key] = {"status": "not run", "reason": "2.10 does not compile flex decoding for head_dim 72 (#164931)"}
                continue
            fas = {s: specs[(c["id"], s)] for s in (0, 1, 2)}
            res[key] = guarded(key, lambda: run_case(f"attention/{key}", MA.attention_case(cand, c), fas))
            report_line(key, res[key])
    return res


def opt_warm_spec(cond, seed, ref, hp=lambda v: Fr(v)):
    """one spec step from the received warm state: parameters and state of step WARM_STEP of the torch for_loop CUDA
    trajectory (float32 values) and the float32-rounded gradients of that step."""
    import p2b_fr_modeA as MA
    import p2b_optimizers as OP
    import spec_optimizers as SO
    cls, kw = OP.OPTS[cond["opt"]]
    snap = ref[(cond["id"], seed)]["base"]["traj"][MA.WARM_STEP]
    grads = OP.grad_sequence(cond, seed, "base")[MA.WARM_STEP]
    out = {}
    for pi in range(3):
        lr = hp(kw["lr"] * (2 if pi == 2 else 1))
        wd = hp(cond["weight_decay"]) if pi < 2 else Fr(0)
        theta = [Fr(float(v)) for v in np.asarray(snap["params"][pi], np.float64).ravel()]
        g = [Fr(float(v)) for v in FE.F32(grads[pi]).ravel()]
        s = {k: np.asarray(v, np.float64).ravel() for k, v in snap["state"][pi].items()}
        n = len(theta)
        if cls in ("Adam", "AdamW"):
            st = SO.AdamState(n)
            st.m = [Fr(float(v)) for v in s["exp_avg"]]
            st.v = [Fr(float(v)) for v in s["exp_avg_sq"]]
            if "max_exp_avg_sq" in s:
                st.vmax = [Fr(float(v)) for v in s["max_exp_avg_sq"]]
            st.t = int(s["step"][0])
            new = SO.adam_step(theta, g, st, lr=lr, beta1=hp(kw["betas"][0]), beta2=hp(kw["betas"][1]), eps=hp(kw["eps"]),
                               weight_decay=wd, amsgrad=kw.get("amsgrad", False), maximize=cond["maximize"],
                               decoupled=cls == "AdamW")
            states = {"exp_avg": st.m, "exp_avg_sq": st.v}
            if kw.get("amsgrad"):
                states["max_exp_avg_sq"] = st.vmax
        elif cls == "SGD":
            st = SO.SGDState(n)
            st.b = [Fr(float(v)) for v in s["momentum_buffer"]]
            new = SO.sgd_step(theta, g, st, lr=lr, momentum=hp(kw["momentum"]), dampening=hp(kw.get("dampening", 0)),
                              weight_decay=wd, nesterov=kw.get("nesterov", False), maximize=cond["maximize"])
            states = {"momentum_buffer": st.b}
        else:
            st = SO.RMSpropState(n)
            st.v = [Fr(float(v)) for v in s["square_avg"]]
            if "grad_avg" in s:
                st.gave = [Fr(float(v)) for v in s["grad_avg"]]
            if "momentum_buffer" in s:
                st.b = [Fr(float(v)) for v in s["momentum_buffer"]]
            new = SO.rmsprop_step(theta, g, st, lr=lr, alpha=hp(kw["alpha"]), eps=hp(kw["eps"]), weight_decay=wd,
                                  momentum=hp(kw["momentum"]), centered=kw.get("centered", False), maximize=cond["maximize"])
            states = {"square_avg": st.v}
            if kw.get("centered"):
                states["grad_avg"] = st.gave
            if kw["momentum"]:
                states["momentum_buffer"] = st.b
        out[f"param{pi}"] = spec_arrays(new)
        for k, v in states.items():
            out[f"state{pi}.{k}"] = spec_arrays(v)
    return out


def run_optimizers():
    import p2b_fr_modeA as MA
    import p2b_optimizers as OP
    from contract_v2 import PENDING, classify_condition
    ref = pickle.loads((OP.CACHE / "torch_for_loop_cuda.pkl").read_bytes())["res"]
    res = {}
    for c in MA.optimizer_cases():
        key = f"torch_compiled_step_cuda/{c['id']}"
        cls = classify_condition("optimizers", c)["classes"]
        if cls[0].startswith("X"):
            res[key] = {"status": "no f", "reason": "contract v2: " + cls[0]}
            continue
        if PENDING["optimizers"](c):
            res[key] = {"status": "clause pending", "reason": "O-D2 (SGD maximize placement), docs/spec_issues_phase2.md"}
            continue
        try:
            fas = {s: opt_warm_spec(c, s, ref) for s in (0, 1, 2)}
            alt = {s: opt_warm_spec(c, s, ref, hp=f32) for s in (0, 1, 2)}
        except Exception as exc:  # noqa: BLE001
            res[key] = {"status": "no f", "reason": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        res[key] = guarded(key, lambda: run_case(f"optimizers/{key}", MA.optimizer_case(c, ref), fas, fas_alt=alt))
        report_line(key, res[key])
    return res


def embedding_case(cond):
    import torch
    import p2b_embedding as EM
    st = {}

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(EM.emb_fn(cond, "base"), dynamic=False)
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        d = EM.make_inputs(cond, seed)
        t = {"w": torch.tensor(d["w"], dtype=torch.float32, device="cuda", requires_grad=True),
             "idx": torch.tensor(d["idx"], dtype=torch.long, device="cuda"),
             "off": None if d["offsets"] is None else torch.tensor(d["offsets"], dtype=torch.long, device="cuda"),
             "psw": torch.tensor(d["psw"], dtype=torch.float32, device="cuda", requires_grad=True),
             "g": torch.tensor(d["g"], dtype=torch.float32, device="cuda"), "_seed": seed}
        return t

    def launch(t):
        t["w"].grad = None
        o = st["fn"](t["w"], t["idx"], t["off"], t["psw"])
        o.backward(t["g"])
        return {"out": o, "dweight": t["w"].grad, "weight_after": t["w"]}

    return setup, inputs, launch


def run_embedding():
    import p2b_embedding as EM
    conds = EM.conditions()
    specs = {(cid, seed): (fa, st) for cid, seed, fa, st in precompute(FE.embedding_job, [(c["id"], s) for c in conds for s in (0, 1, 2)])}
    res = {}
    for c in conds:
        key = f"inductor_cuda_float32/{c['id']}"
        st = [specs[(c["id"], s)] for s in (0, 1, 2)]
        if any(fa is None for fa, _ in st):
            res[key] = {"status": "no f", "reason": st[0][1]}
            continue
        fas = {s: st[s][0] for s in (0, 1, 2)}
        res[key] = guarded(key, lambda: run_case(f"embedding/{key}", embedding_case(c), fas))
        if c["max_norm"] and res[key].get("outputs", {}).get("weight_after"):
            res[key]["outputs"]["weight_after"]["clause_pending"] = "E-D2 (max_norm scope)"
        report_line(key, res[key])
    return res


def packing_case(cond):
    import torch
    import p2b_packing as PK
    st = {"cache": {}}
    cand = {"id": "flex_block_mask", "kind": "flex", "compiled": True}

    def setup():
        torch._dynamo.reset()
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        rng = PK._rng(f"pack/{cond['id']}/{seed}")
        t_ = sum(cond["doc_lengths"])
        q, k, v, g = (rng.normal(0, 1, (1, PK.H, t_, PK.D)) for _ in range(4))
        t = {n: torch.tensor(a, dtype=torch.float32, device="cuda", requires_grad=n != "g") for n, a in zip("qkvg", (q, k, v, g))}
        t["_seed"] = seed
        return t

    def launch(t):
        for n in "qkv":
            t[n].grad = None
        o = PK.attn_call(cand, t["q"], t["k"], t["v"], PK.doc_mask(cond), st["cache"])
        o.backward(t["g"])
        return {"out": o}

    return setup, inputs, launch


def run_packing():
    import p2b_packing as PK
    conds = PK.conditions()
    specs = {(cid, seed): fa for cid, seed, _r, fa in precompute(FE.packing_job, [(c["id"], s, "float32") for c in conds for s in (0, 1, 2)])}
    res = {}
    for c in conds:
        key = f"flex_block_mask/{c['id']}"
        fas = {s: specs[(c["id"], s)] for s in (0, 1, 2)}
        res[key] = guarded(key, lambda: run_case(f"packing/{key}", packing_case(c), fas))
        report_line(key, res[key])
    return res


def moe_case(cond, sets_of_seed, fas):
    import torch
    import p2b_small_families as SM
    st = {}

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(SM.moe_vectorised)
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        xn, ln, Wn, cap = SM.moe_inputs(cond, seed)
        t = {n: torch.tensor(a, dtype=torch.float32, device="cuda") for n, a in zip(("x", "logits", "W"), (xn, ln, Wn))}
        t["cap"], t["_seed"] = cap, seed
        return t

    def launch(t):
        out, order, *_ = st["fn"](cond, t["x"], t["logits"], t["W"], t["cap"])
        chosen = [sorted(int(e) for e in row) for row in order.cpu().numpy()]
        valid = sets_of_seed[t["_seed"]]
        lo, hi = fas[t["_seed"]]["out"]
        lo, hi = lo.reshape(len(chosen), -1).copy(), hi.reshape(len(chosen), -1).copy()
        for i, (ch, v) in enumerate(zip(chosen, valid)):    # another valid tie set (MOE-C2): f of that token not compared
            if ch != v[0]:
                lo[i], hi[i] = np.nan, np.nan
        fas[t["_seed"]]["out"] = (lo.ravel(), hi.ravel())
        return {"out": out}

    return setup, inputs, launch


def run_moe():
    import p2b_small_families as SM
    conds = SM.plan("moe", SM.MO_KEYS)
    pre = {(cid, seed): (fa, sets) for cid, seed, fa, sets, _st in precompute(FE.moe_job, [(c["id"], s) for c in conds for s in (0, 1, 2)])}
    res = {}
    for c in conds:
        key = f"vectorised_compiled/{c['id']}"
        fas = {s: pre[(c["id"], s)][0] for s in (0, 1, 2)}
        sets = {s: pre[(c["id"], s)][1] for s in (0, 1, 2)}
        if any(v is None for v in fas.values()):
            res[key] = {"status": "no f", "reason": "spec declined"}
            continue
        res[key] = guarded(key, lambda: run_case(f"moe/{key}", moe_case(c, sets, fas), fas))
        report_line(key, res[key])
    return res


def report_line(key, r):
    if r["status"] != "ok":
        print(key, r["status"], r.get("reason", "")[:200], flush=True)
        return
    o = r["outputs"]
    print(key, "ok", {k: v["ok_elements"] for k, v in o.items()}, "e_num>τ32", sum(v["e_num_beyond_tau32"] for v in o.values()),
          "e_sem certified", sum(v["e_sem_certified_elements"] for v in o.values()),
          "undef f", sum(v["f_undefined_elements"] for v in o.values()),
          "not-Triton", r["notes"]["outputs_not_written_by_triton"], "unbound", r["notes"]["outputs_binding_not_established"],
          flush=True)


FAMILIES = {"matmul_linear": lambda: run_basic("matmul_linear"), "reductions": lambda: run_basic("reductions"),
            "activations": lambda: run_basic("activations"), "gather_layout": lambda: run_basic("gather_layout"),
            "normalization": run_normalization, "attention": run_attention, "optimizers": run_optimizers,
            "embedding": run_embedding, "packing": run_packing, "moe": run_moe}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in sys.argv[1:]:
        t0 = time.time()
        cases = FAMILIES[fam]()
        tally = {}
        for r in cases.values():
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        out = {"family": fam, "seconds": round(time.time() - t0, 1), "status_counts": tally, "cases": cases}
        (OUT / f"{fam}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"== {fam}: {tally} ({out['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
