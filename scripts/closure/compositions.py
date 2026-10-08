#!/usr/bin/env python3
"""Three composition checks of the closure (task book section 6; protocol v3 section 6), through the production tool
(kernel_analyzer.check.run via common.fr_run, detector 2.2) and the closure judgment code (contract_v3, contract_v2).

C1 pure compute       linear -> GELU(tanh) -> residual, Inductor float32, mode B (f from spec_base_ops: linear, gelu_tanh,
                      exact addition), 1 development + 32 confirmation units.  Declared variants on the same program:
                      recorder keep-alive off (arrow 0), 8 confirmation units (arrow 3 sample), a query with an
                      unregistered direction on its residuals (arrow 2).
C2 saved value/branch RMSNorm under activation checkpoint -> RoPE (cos / sin rows gathered by position id) -> causal
                      attention, Inductor float32, forward and backward (the checkpoint recomputes the RMSNorm in the
                      backward), mode A, 1 + 32 units with a per-unit log-normal input scale (declared: unit residual
                      means spread over decades).  Declared variant: the same block with LayerNorm in bfloat16 over a
                      1027-wide hidden state projected to the attention width (the normalisation reduces with a Welford
                      combiner).
C3 state update       AdamW, compiled optimizer step, sequence init -> normal -> zero grad -> grad=None (parameter 1) ->
                      state_dict save / reload -> step, mode B per step with f from spec_optimizers on the state the step
                      received; A -> B -> A: the reloaded run against the uninterrupted run.  Declared variant: a
                      persistent fused AdamW Triton kernel that takes its tiles from an atomic ticket counter (the
                      atomic's returned value selects the addresses).

Every record goes to results/closure/compositions/<name>.json; docs/composition_checklists_20261008.md cites them.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for p in ("specs/phase2", "scripts/essential", "scripts/closure", "src"):
    sys.path.insert(0, str(ROOT / p))
import contract_v3 as CV  # noqa: E402
from f_eval import spec_arrays  # noqa: E402

OUT = ROOT / "results/closure/compositions"
TAU32 = 2.0 ** -12
NOTE_KEYS = ("outputs_not_written_by_triton", "outputs_binding_not_established", "outputs_at_address_of_another_recorded_storage",
             "outputs_modified_after_last_triton_write", "outputs_whose_writing_programs_aborted", "ttir_coverage_complete",
             "tool_version")


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")
    print("saved", name, flush=True)


def f32(a):
    return np.asarray(a, np.float64).astype(np.float32).astype(np.float64)


def frs(a):
    a = np.asarray(a, np.float64)
    return Fr(float(a)) if a.ndim == 0 else [frs(x) for x in a]


def judge_records(out_rec):
    """the frozen statistics records of one output (numerical, semantic) -> v3 judgments per rule."""
    res = {}
    for key in ("numerical", "semantic"):
        rec = out_rec.get(key)
        if not rec:
            continue
        rules = rec.get("rules")
        if isinstance(rules, list) and rules:
            res[key] = {r.get("rule", str(i)): CV.statistical_judgment(r) for i, r in enumerate(rules) if isinstance(r, dict)}
        elif isinstance(rules, dict) and rules:
            res[key] = {r: CV.statistical_judgment(v) for r, v in rules.items() if isinstance(v, dict)}
        else:
            res[key] = {"record": CV.statistical_judgment(rec)}
    return res


def kappa_of(out_rec):
    """composition containment per output from the tool's reference classes: complete (every written element has a
    complete finite K_R; 'assumed:' entries are premises the tool checked, e.g. PTX zero fill), conditional (part of the
    elements excluded: pinned loads / undecided branches), unestablished (none)."""
    rc = out_rec.get("reference_classes") or {}
    frac = rc.get("finite_complete_fraction", 0) or 0
    reasons = out_rec.get("not_established_reasons_seed0") or {}
    other = {k: v for k, v in reasons.items() if not str(k).startswith("assumed:")}
    if frac >= 1.0:
        return "complete" + (" (checked premises: " + ", ".join(sorted({str(k).split("@")[0] for k in reasons})) + ")" if reasons else "")
    if frac > 0:
        return "conditional (fraction %.3f; " % frac + ", ".join(f"{k}: {v}" for k, v in list(other.items())[:4]) + ")"
    return "unestablished (" + ", ".join(f"{k}: {v}" for k, v in list(other.items())[:4]) + ")"


def elementwise_sem(keep, fas, name):
    n_ok = n_sem = 0
    worst = 0.0
    for r in keep.get(name, []):
        lo_f, hi_f = (np.asarray(a, np.float64).ravel() for a in fas[r["seed"]][name])
        ok = np.asarray(r["ok"], bool) & np.isfinite(lo_f)
        lo, hi = np.asarray(r["r_lo"], np.float64), np.asarray(r["r_hi"], np.float64)
        with np.errstate(invalid="ignore"):
            gap = np.maximum(lo - hi_f, lo_f - hi)
        sem = ok & (gap > 0)
        n_ok += int(ok.sum())
        n_sem += int(sem.sum())
        if sem.any():
            worst = max(worst, float((gap / (1 + np.abs(0.5 * (lo_f + hi_f))))[sem].max()))
    return {"ok_elements": n_ok, "e_sem_certified": n_sem, "max_gap_relative": worst}


def f_black_box(keep, fas, name, special):
    """F (K - f, no K_R): K outside f's enclosure widened by τ32 (1 + |mid|); K special values counted apart (the keep
    frame stores them as 0, the tool's special_values count them)."""
    n = beyond = 0
    for r in keep.get(name, []):
        lo_f, hi_f = (np.asarray(a, np.float64).ravel() for a in fas[r["seed"]][name])
        k = np.asarray(r["k"], np.float64)
        fin = np.isfinite(lo_f)
        mid = 0.5 * (lo_f + hi_f)
        dist = np.maximum(np.maximum(lo_f - k, k - hi_f), 0.0)
        n += int(fin.sum())
        beyond += int((fin & (dist > TAU32 * (1 + np.abs(mid)))).sum())
    knf = int(special.get("k_vs_f_class_mismatch") or 0)
    return {"elements": n, "beyond_tau32": max(0, beyond - knf), "K_special_vs_f": knf}


def report_summary(rep, keep, fas=None):
    outs = {}
    for name, o in rep.get("outputs", {}).items():
        e = {"kappa": kappa_of(o), "mixed_non_triton_sources": o.get("depends_on_non_triton_intermediates") or [],
             "complete_fraction": (o.get("reference_classes") or {}).get("finite_complete_fraction"),
             "not_established_reasons_seed0": o.get("not_established_reasons_seed0"),
             "statistics_v3": judge_records(o), "special_values": o.get("special_values")}
        if fas is not None and name in keep:
            e["elementwise_semantic"] = elementwise_sem(keep, fas, name)
            e["F_black_box"] = f_black_box(keep, fas, name, o.get("special_values") or {})
        outs[name] = e
    for name in rep.get("outputs_not_written_by_triton") or []:
        outs.setdefault(name, {"kappa": "not applicable", "ok_elements": 0,
                               "conclusion": "not established (arrow 4: ok_elements = 0, output not written by Triton)"})
    for name in rep.get("outputs_binding_not_established") or []:
        outs.setdefault(name, {"conclusion": "not established (arrow 0: pairing not provable)"})
    return {"notes": {k: rep.get(k) for k in NOTE_KEYS}, "launches": [l.get("kernel", "")[:70] for l in rep.get("launches") or []],
            "outputs": outs, "seconds": rep.get("seconds")}


def fr(name, setup, inputs, launch, spec, seeds):
    import common
    rep, keep = common.fr_run(name, setup, inputs, launch, spec, seeds=seeds)
    return rep, keep


# ------------------------------------------------------------------------------------------------ C1

D1_, N1_ = 64, 8


def c1_inputs_np(seed):
    rng = np.random.default_rng(10_000 + seed)
    return {"x": f32(rng.normal(0, 1, (N1_, D1_))), "W": f32(rng.normal(0, 1, (D1_, D1_)) / math.sqrt(D1_)),
            "b": f32(rng.normal(0, 0.1, D1_))}


def c1_spec_np(d):
    import spec_base_ops as SB
    h = SB.linear(frs(d["x"]), frs(d["W"]), frs(d["b"]))
    y = [[SB.I(xv) + SB.gelu_tanh(hv) for xv, hv in zip(xr, hr)] for xr, hr in zip(frs(d["x"]), h)]
    return {"y": spec_arrays(y), "h": spec_arrays(h)}


def c1_case(seeds):
    st = {}
    fas = {}

    def prog(x, W, b):
        h = F.linear(x, W, b)
        return x + F.gelu(h, approximate="tanh"), h

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(prog, dynamic=False)
        launch(inputs(seeds[0]))
        torch.cuda.synchronize()

    def inputs(seed):
        d = c1_inputs_np(seed)
        if seed not in fas:
            fas[seed] = c1_spec_np(d)
        t = {k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in d.items()}
        t["_seed"] = seed
        return t

    def launch(t):
        y, h = st["fn"](t["x"], t["W"], t["b"])
        return {"y": y, "h": h}

    def spec(t):
        fa = fas[t["_seed"]]
        return {"y": tuple(a.reshape(N1_, D1_) for a in fa["y"]), "h": tuple(a.reshape(N1_, D1_) for a in fa["h"])}
    return setup, inputs, launch, spec, fas


def run_c1():
    from contract_v2 import classify_condition
    seeds = list(range(33))
    setup, inputs, launch, spec, fas = c1_case(seeds)
    t0 = time.time()
    rep, keep = fr("composition/C1_linear_gelu_residual", setup, inputs, launch, spec, seeds)
    main = report_summary(rep, keep, fas)
    main["contract"] = {"activations (gelu_tanh)": classify_condition("activations", {"id": "acti_gelu_tanh_gauss_contiguous",
                                                                                       "op": "gelu_tanh", "values": "gauss",
                                                                                       "layout": "contiguous"}),
                        "composition": classify_condition("composition_C1", {"id": "C1"})}
    main["seconds_total"] = round(time.time() - t0, 1)
    main["units"] = {"development": 1, "confirmation": len(seeds) - 1}
    save("C1_main", main)

    # variant: keep-alive off (arrow 0)
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    orig = TritonLaunchRecorder.__init__

    def init(self, *a, **k):
        k.setdefault("keep_storages", False)
        orig(self, *a, **k)
    TritonLaunchRecorder.__init__ = init
    try:
        s2 = list(range(3))
        setup, inputs, launch, spec, fas2 = c1_case(s2)
        rep2, keep2 = fr("composition/C1_keepalive_off", setup, inputs, launch, spec, s2)
    finally:
        TritonLaunchRecorder.__init__ = orig
    save("C1_variant_keepalive_off", report_summary(rep2, keep2, fas2))

    # variant: 8 confirmation units (arrow 3 sample size)
    s3 = list(range(9))
    setup, inputs, launch, spec, fas3 = c1_case(s3)
    rep3, keep3 = fr("composition/C1_n8", setup, inputs, launch, spec, s3)
    save("C1_variant_n8", report_summary(rep3, keep3, fas3))

    # variant: unregistered direction on the C1 residuals (arrow 2)
    from kernel_analyzer.reference_eval import analysis as A
    rows = keep["y"]
    lo = np.stack([r["k"] - r["r_hi"] for r in rows])
    hi = np.stack([r["k"] - r["r_lo"] for r in rows])
    ref = np.stack([0.5 * (r["r_lo"] + r["r_hi"]) for r in rows])
    try:
        res = A.apply_direction_rules("C1 y: e_num", lo, hi, ref, {"direction_rules": ["unregistered_direction"]}, 1,
                                      len(rows) - 1, 0.05)
        outcome = {"result": res}
    except Exception as exc:  # noqa: BLE001
        outcome = {"raised": f"{type(exc).__name__}: {exc}"}
    save("C1_variant_unregistered_direction", {"output": "y", "units": len(rows), **outcome})


# ------------------------------------------------------------------------------------------------ C2

B2, L2, D2, H2 = 2, 12, 32, 2
DH2 = D2 // H2


def c2_block(norm="rms", dm=D2):
    from torch.utils.checkpoint import checkpoint

    def rms(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, 1e-5)

    def ln(x, w):
        return F.layer_norm(x, (x.shape[-1],), w, None, 1e-5)

    nf = rms if norm == "rms" else ln

    def rot(t):
        return torch.cat([-t[..., DH2 // 2:], t[..., : DH2 // 2]], -1)

    def block(x, w, Wq, Wk, Wv, pos, cos_t, sin_t):
        h = checkpoint(nf, x, w, use_reentrant=False)
        q = (h @ Wq).view(B2, L2, H2, DH2).transpose(1, 2)    # Wq: (dm, D2)
        k = (h @ Wk).view(B2, L2, H2, DH2).transpose(1, 2)
        v = (h @ Wv).view(B2, L2, H2, DH2).transpose(1, 2)
        cos, sin = cos_t[pos], sin_t[pos]                     # rows gathered by position id
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        s = (q @ k.transpose(-1, -2)) / math.sqrt(DH2)
        mask = torch.ones(L2, L2, dtype=torch.bool, device=x.device).tril()
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        o = p @ v
        return o, h, cos, p
    return block


def c2_inputs(seed, dtype, dm=D2):
    rng = np.random.default_rng(20_000 + seed)
    scale = math.exp(rng.normal(0, 1.5))                       # declared: per-unit log-normal input scale
    d = {"x": rng.normal(0, 1, (B2, L2, dm)) * scale, "w": 1 + 0.1 * rng.normal(0, 1, dm),
         "Wq": rng.normal(0, 1, (dm, D2)) / math.sqrt(dm), "Wk": rng.normal(0, 1, (dm, D2)) / math.sqrt(dm),
         "Wv": rng.normal(0, 1, (dm, D2)) / math.sqrt(dm), "g": rng.normal(0, 1, (B2, H2, L2, DH2))}
    inv = 1.0 / (10000 ** (np.arange(0, DH2, 2) / DH2))
    ang = np.arange(64)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    t = {k: torch.tensor(v, dtype=dtype, device="cuda") for k, v in d.items()}
    t["cos_t"], t["sin_t"] = (torch.tensor(f(emb), dtype=dtype, device="cuda") for f in (np.cos, np.sin))
    t["pos"] = torch.tensor(rng.permutation(64)[:L2], dtype=torch.long, device="cuda")
    for k in ("x", "w"):
        t[k].requires_grad_(True)
    t["_seed"] = seed
    return t


def c2_case(dtype, norm, seeds, dm=D2):
    st = {}

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(c2_block(norm, dm), dynamic=False)
        launch(inputs(seeds[0]))
        torch.cuda.synchronize()

    def inputs(seed):
        return c2_inputs(seed, dtype, dm)

    def launch(t):
        t["x"].grad = None
        t["w"].grad = None
        o, h, cos, p = st["fn"](t["x"], t["w"], t["Wq"], t["Wk"], t["Wv"], t["pos"], t["cos_t"], t["sin_t"])
        o.backward(t["g"])
        return {"o": o, "h": h, "cos_gathered": cos, "p": p, "dx": t["x"].grad, "dw": t["w"].grad}
    return setup, inputs, launch


def run_c2():
    from contract_v2 import classify_condition
    seeds = list(range(33))
    setup, inputs, launch = c2_case(torch.float32, "rms", seeds)
    t0 = time.time()
    rep, keep = fr("composition/C2_rmsnorm_ckpt_rope_attention", setup, inputs, launch, lambda _t: None, seeds)
    main = report_summary(rep, keep)
    main["contract"] = classify_condition("composition_C2", {"id": "C2"})
    main["units"] = {"development": 1, "confirmation": len(seeds) - 1, "declared": "per-unit log-normal input scale (sigma 1.5)"}
    main["seconds_total"] = round(time.time() - t0, 1)
    save("C2_main", main)
    s2 = list(range(3))
    try:
        setup, inputs, launch = c2_case(torch.bfloat16, "layer", s2, dm=1027)   # pre-norm over a 1027-wide hidden state
        rep2, keep2 = fr("composition/C2_layernorm_bf16", setup, inputs, launch, lambda _t: None, s2)
        save("C2_variant_layernorm_bf16", report_summary(rep2, keep2))
    except Exception as exc:  # noqa: BLE001
        save("C2_variant_layernorm_bf16", {"raised": f"{type(exc).__name__}: {exc}"[:600], "trace": traceback.format_exc()[-2000:]})


# ------------------------------------------------------------------------------------------------ C3

SHAPES3 = ((5, 9), (7,), (3, 4))
HP3 = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
STEPS3 = ("init", "normal", "zero_grad", "grad_none", "reload")


def c3_grads(seed, t):
    rng = np.random.default_rng(30_000 + 97 * seed + t)
    gs = [f32(rng.normal(0, 1, s)) for s in SHAPES3]
    if STEPS3[t] == "zero_grad":
        gs = [np.zeros(s) for s in SHAPES3]
    if STEPS3[t] == "grad_none":
        gs[1] = None
    return gs


def c3_build(seed):
    rng = np.random.default_rng(31_000 + seed)
    params = [torch.tensor(f32(rng.normal(0, 1, s)), dtype=torch.float32, device="cuda", requires_grad=True) for s in SHAPES3]
    opt = torch.optim.AdamW(params, **HP3)
    return params, opt


def c3_set_grads(params, gs):
    for p, g in zip(params, gs):
        p.grad = None if g is None else torch.tensor(g, dtype=torch.float32, device="cuda")


def c3_snapshot(params, opt):
    snap = {"params": [p.detach().double().cpu().numpy().copy() for p in params], "state": []}
    for p in params:
        s = opt.state.get(p, {})
        snap["state"].append({k: (v.detach().double().cpu().numpy().copy() if torch.is_tensor(v) else v) for k, v in s.items()})
    return snap


def c3_spec(snap, gs):
    import spec_optimizers as SO
    out = {}
    for i, (theta, s) in enumerate(zip(snap["params"], snap["state"])):
        th = [Fr(float(v)) for v in theta.ravel()]
        st = SO.AdamState(len(th))
        if s:
            st.m = [Fr(float(v)) for v in s["exp_avg"].ravel()]
            st.v = [Fr(float(v)) for v in s["exp_avg_sq"].ravel()]
            st.t = int(np.asarray(s["step"]).ravel()[0])
        g = None if gs[i] is None else [Fr(float(v)) for v in gs[i].ravel()]
        new = SO.adam_step(th, g, st, lr=Fr(HP3["lr"]), beta1=Fr(HP3["betas"][0]), beta2=Fr(HP3["betas"][1]),
                           eps=Fr(HP3["eps"]), weight_decay=Fr(HP3["weight_decay"]), decoupled=True)
        out[f"param{i}"] = spec_arrays(new)
        if g is not None:
            out[f"state{i}.exp_avg"] = spec_arrays(st.m)
            out[f"state{i}.exp_avg_sq"] = spec_arrays(st.v)
    return out


def c3_case(t_step, seeds, reload_before):
    st = {"fas": {}, "k": {}}

    def advance(seed):
        torch._dynamo.reset()                                # a fresh optimizer per unit: avoid the recompile limit
        params, opt = c3_build(seed)
        step = torch.compile(opt.step)
        for t in range(t_step):
            c3_set_grads(params, c3_grads(seed, t))
            step()
        if reload_before:                                    # state_dict save / reload before the measured step
            sd = opt.state_dict()
            opt2 = torch.optim.AdamW(params, **HP3)
            opt2.load_state_dict(sd)
            opt, step = opt2, torch.compile(opt2.step)
        return params, opt, step

    def setup():
        torch._dynamo.reset()
        params, opt, step = advance(seeds[0])
        c3_set_grads(params, c3_grads(seeds[0], t_step))
        step()
        torch.cuda.synchronize()

    def inputs(seed):
        params, opt, step = advance(seed)
        gs = c3_grads(seed, t_step)
        c3_set_grads(params, gs)
        st["fas"][seed] = c3_spec(c3_snapshot(params, opt), gs)
        return {"params": params, "opt": opt, "step": step, "_seed": seed}

    def launch(t):
        t["step"]()
        outs = {f"param{i}": p for i, p in enumerate(t["params"])}
        for i, p in enumerate(t["params"]):
            for k, v in t["opt"].state.get(p, {}).items():
                if torch.is_tensor(v) and v.is_cuda and v.dim() > 0:
                    outs[f"state{i}.{k}"] = v
        st["k"][t["_seed"]] = {k: v.detach().double().cpu().numpy().copy() for k, v in outs.items()}
        return {k: v for k, v in outs.items() if k in st["fas"][t["_seed"]]}

    def spec(t):
        fa = st["fas"][t["_seed"]]
        return {k: (lo.reshape(v.shape), hi.reshape(v.shape)) for k, (lo, hi) in fa.items()
                for v in [t["params"][int(k[5])] if k.startswith("param") else t["params"][int(k[5])]]}
    return setup, inputs, launch, spec, st


def run_c3():
    seeds = [0, 1, 2]
    res, ks = {}, {}
    for t_step, name in enumerate(STEPS3):
        reload_before = name == "reload"
        try:
            setup, inputs, launch, spec, st = c3_case(t_step, seeds, reload_before)
            rep, keep = fr(f"composition/C3_adamw_{name}", setup, inputs, launch, spec, seeds)
            res[name] = report_summary(rep, keep, st["fas"])
            ks[name] = st["k"]
        except Exception as exc:  # noqa: BLE001
            res[name] = {"raised": f"{type(exc).__name__}: {exc}"[:600], "trace": traceback.format_exc()[-2000:]}
    # A -> B -> A: the reloaded run against the uninterrupted run at the same step (bitwise, every output)
    try:
        setup, inputs, launch, spec, st = c3_case(len(STEPS3) - 1, seeds, False)
        for s in seeds:
            t = inputs(s)
            launch(t)
        unint = st["k"]
        aba = {}
        for s in seeds:
            for k, v in ks.get("reload", {}).get(s, {}).items():
                u = unint.get(s, {}).get(k)
                aba.setdefault(k, {"compared_elements": 0, "bitwise_different": 0})
                if u is not None and u.shape == v.shape:
                    aba[k]["compared_elements"] += int(v.size)
                    aba[k]["bitwise_different"] += int((v != u).sum())
        res["A_B_A"] = aba
    except Exception as exc:  # noqa: BLE001
        res["A_B_A"] = {"raised": f"{type(exc).__name__}: {exc}"[:600]}
    # arrow 4 on the grad=None step with the classes this run actually has: the compiled step's parameter 1 has no element
    # written by Triton (FR not established), so it has no class; the eager step (torch for_loop) leaves it unchanged
    gn = res.get("grad_none", {}).get("outputs", {})
    p1 = gn.get("param1", {})
    res["arrow4_param1_grad_none"] = {"fr_conclusion": p1.get("conclusion"),
                                      "shared_relation": "not formed: classify.combine needs a class on both sides; the "
                                                         "candidate side is 'not established' (no comparable pair is not "
                                                         "reached by this run)"}
    res["units"] = {"development": 1, "confirmation": 2}
    save("C3_main", res)
    run_c3_persistent()


def run_c3_persistent():
    import triton
    import triton.language as tl

    @triton.jit
    def adamw_persistent(p_ptr, g_ptr, m_ptr, v_ptr, ticket_ptr, n, lr, b1, b2, eps, wd, bc1, bc2,
                         BLOCK: tl.constexpr, TILES: tl.constexpr):
        for _ in range(TILES):
            tile = tl.atomic_add(ticket_ptr, 1)              # work queue: the returned old value picks the tile
            offs = tile * BLOCK + tl.arange(0, BLOCK)
            msk = offs < n
            p = tl.load(p_ptr + offs, mask=msk)
            g = tl.load(g_ptr + offs, mask=msk)
            m = tl.load(m_ptr + offs, mask=msk)
            v = tl.load(v_ptr + offs, mask=msk)
            p = p * (1 - lr * wd)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            p = p - lr * (m / bc1) / (tl.sqrt(v / bc2) + eps)
            tl.store(p_ptr + offs, p, mask=msk)
            tl.store(m_ptr + offs, m, mask=msk)
            tl.store(v_ptr + offs, v, mask=msk)

    N, BLOCK, PROGS = 1000, 128, 4
    tiles = triton.cdiv(N, BLOCK)
    per_prog = triton.cdiv(tiles, PROGS)

    def inputs(seed):
        rng = np.random.default_rng(32_000 + seed)
        t = {k: torch.tensor(f32(rng.normal(0, 1, N)) if k != "v" else f32(rng.random(N)), dtype=torch.float32, device="cuda")
             for k in ("p", "g", "m", "v")}
        t["ticket"] = torch.zeros(1, dtype=torch.int32, device="cuda")
        t["_seed"] = seed
        return t

    def launch(t):
        adamw_persistent[(PROGS,)](t["p"], t["g"], t["m"], t["v"], t["ticket"], N, 1e-3, 0.9, 0.999, 1e-8, 0.01,
                                   1 - 0.9 ** 3, 1 - 0.999 ** 3, BLOCK=BLOCK, TILES=per_prog)
        return {"p": t["p"], "m": t["m"], "v": t["v"]}

    try:
        rep, keep = fr("composition/C3_persistent_atomic_ticket", lambda: launch(inputs(0)), inputs, launch, lambda _t: None,
                       [0, 1, 2])
        save("C3_variant_persistent_atomic_ticket", report_summary(rep, keep))
    except Exception as exc:  # noqa: BLE001
        save("C3_variant_persistent_atomic_ticket", {"raised": f"{type(exc).__name__}: {exc}"[:600],
                                                     "trace": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    for name in sys.argv[1:]:
        try:
            {"C1": run_c1, "C2": run_c2, "C3": run_c3}[name]()
        except Exception:  # noqa: BLE001
            print(name, "FAILED", traceback.format_exc()[-2500:], flush=True)
