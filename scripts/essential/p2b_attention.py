#!/usr/bin/env python3
"""2b family "attention" (protocol v2 section 3; registry tier1.attention): E, P, forward and backward.  F and mode-B FR
stay closed until the reviewer's attention spec is delivered; mode-A FR for the Triton candidates is in p2b_fr_modeA.py.

Conditions: the pre-registered coverage plan (24 distinct combinations of q/k lengths, mask, GQA, scale, head_dim) x 3
seeds; B = 2, 4 query heads (2 KV heads under GQA); lengths: equal 37/37, q<k 19/37, q>k 37/19, q=1 1/37.  Masks are
"allowed" matrices A[b, i, j], top-left aligned as in the SDPA documentation (``tril(diagonal=0)``): causal j <= i; padding:
batch 1 masks its last 3 keys; fully_masked_row: batch 1 rows 0 and Lq // 2 have no allowed key; sliding_window: i - 5 < j <= i
(rows of q>k beyond the last key + window are then fully masked too).  SDPA's ``is_causal=True`` is used for "causal", explicit
masks otherwise.  Rows without an allowed key are undefined: excluded from E and P and recorded per candidate as a convention;
the upstream gradient is zero on them.

Variants (P, auxiliary runs of the same candidate): ``repeat`` (determinism, for the bitwise properties), ``perturb`` (keys and
values at masked positions replaced by 10 x larger random values), ``v_ones`` (V = 1: every defined row must give 1), and
``gqa_repeated`` (K, V repeated to the query heads, enable_gqa off).

    python scripts/essential/p2b_attention.py run --env ka_main
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_attention.py run --env liger
    python scripts/essential/p2b_attention.py analyse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/attention"
OUT = ROOT / "results/essential/phase2b/attention"
SEEDS = (0, 1, 2)
B, HQ, W = 2, 4, 5
LENS = {"equal": (37, 37), "q<k": (19, 37), "q>k": (37, 19), "q=1": (1, 37)}
TAU = {"float32": 2.0 ** -12, "float64": 1e-9}
PERTURB_KEYS = {"causal": "last", "padding": "padded", "sliding_window": "first"}


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["attention"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = tuple(c[k] for k in ("q_len_k_len", "mask", "gqa", "scale", "head_dim"))
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = dict(c, high_risk=bool(c.get("high_risk")))
        d["id"] = (f"att_{c['q_len_k_len'].replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}_{c['mask']}"
                   f"_{'gqa' if c['gqa'] else 'mha'}_s{c['scale']}_d{c['head_dim']}")
        seen[key] = d
        out.append(d)
    return out


def allowed(cond):
    lq, lk = LENS[cond["q_len_k_len"]]
    i, j = np.arange(lq)[:, None], np.arange(lk)[None, :]
    a = np.ones((B, lq, lk), dtype=bool)
    m = cond["mask"]
    if m == "causal":
        a &= (j <= i)[None]
    elif m == "padding":
        a[1, :, lk - 3:] = False
    elif m == "fully_masked_row":
        a[1, [0, lq // 2], :] = False
    elif m == "sliding_window":
        a &= ((j <= i) & (j > i - W))[None]
    return a


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def make_inputs(cond, seed, variant):
    rng = _rng(f"att/{cond['id']}/{seed}")
    lq, lk = LENS[cond["q_len_k_len"]]
    hk = 2 if cond["gqa"] else HQ
    d = cond["head_dim"]
    q, k, v = rng.normal(0, 1, (B, HQ, lq, d)), rng.normal(0, 1, (B, hk, lk, d)), rng.normal(0, 1, (B, hk, lk, d))
    g = rng.normal(0, 1, (B, HQ, lq, d))
    a = allowed(cond)
    defined = a.any(-1)                                   # (B, lq)
    g = g * defined[:, None, :, None]
    if variant == "perturb":
        jset = perturb_keys(cond, a)
        prng = _rng(f"att-perturb/{cond['id']}/{seed}")
        for b, js in enumerate(jset):
            for j in js:
                k[b, :, j] = 10 * prng.normal(0, 1, (hk, d))
                v[b, :, j] = 10 * prng.normal(0, 1, (hk, d))
    if variant == "v_ones":
        v = np.ones_like(v)
    return {"q": q, "k": k, "v": v, "g": g, "allowed": a}


def perturb_keys(cond, a):
    kind = PERTURB_KEYS.get(cond["mask"])
    lk = a.shape[-1]
    if kind == "last":
        return [[lk - 1]] * B
    if kind == "first":
        return [[0]] * B
    if kind == "padded":
        return [[], list(range(lk - 3, lk))]
    return None


def unaffected_rows(cond, a):
    """(b, i) rows that have an allowed key and none of the perturbed keys allowed."""
    js = perturb_keys(cond, a)
    rows = np.zeros(a.shape[:2], dtype=bool)
    for b in range(B):
        rows[b] = a[b].any(-1) & ~a[b][:, js[b]].any(-1) if js[b] else a[b].any(-1)
    return rows


def variants_for(cond):
    v = ["base", "repeat", "v_ones"]
    if PERTURB_KEYS.get(cond["mask"]):
        v.append("perturb")
    if cond["gqa"]:
        v.append("gqa_repeated")
    return v


# ------------------------------------------------------------------------------------------------ candidates

def candidates_for(env):
    if env == "ka_main":
        return [{"id": "sdpa_math_float32", "kind": "sdpa", "backend": "MATH", "dtype": "float32", "device": "cuda"},
                {"id": "sdpa_math_bfloat16", "kind": "sdpa", "backend": "MATH", "dtype": "bfloat16", "device": "cuda"},
                {"id": "sdpa_efficient_float32", "kind": "sdpa", "backend": "EFFICIENT_ATTENTION", "dtype": "float32", "device": "cuda"},
                {"id": "sdpa_efficient_bfloat16", "kind": "sdpa", "backend": "EFFICIENT_ATTENTION", "dtype": "bfloat16", "device": "cuda"},
                {"id": "sdpa_flash_bfloat16", "kind": "sdpa", "backend": "FLASH_ATTENTION", "dtype": "bfloat16", "device": "cuda"},
                {"id": "sdpa_cudnn_bfloat16", "kind": "sdpa", "backend": "CUDNN_ATTENTION", "dtype": "bfloat16", "device": "cuda"},
                {"id": "sdpa_math_cpu_float64", "kind": "sdpa", "backend": "MATH", "dtype": "float64", "device": "cpu"},
                {"id": "manual_eager_float32", "kind": "manual", "dtype": "float32", "device": "cuda", "reference_only": True},
                {"id": "inductor_attention_float32", "kind": "manual", "compiled": True, "dtype": "float32", "device": "cuda"},
                {"id": "flex_eager_float32", "kind": "flex", "dtype": "float32", "device": "cuda", "reference_only": True},
                {"id": "flex_attention_float32", "kind": "flex", "compiled": True, "dtype": "float32", "device": "cuda"}]
    if env == "ka_main_f64":                 # closure protocol v3 2.2: same-device float32 / float64 pairs (precision invariance)
        return [{"id": "sdpa_math_cuda_float64", "kind": "sdpa", "backend": "MATH", "dtype": "float64", "device": "cuda"},
                {"id": "manual_eager_float64", "kind": "manual", "dtype": "float64", "device": "cuda", "reference_only": True},
                {"id": "sdpa_math_cpu_float32", "kind": "sdpa", "backend": "MATH", "dtype": "float32", "device": "cpu"}]
    if env == "liger":
        return [{"id": "hf_eager_attention", "kind": "hf", "dtype": "float32", "device": "cuda"},
                {"id": "xformers_memory_efficient", "kind": "xformers", "dtype": "bfloat16", "device": "cuda"}]
    raise KeyError(env)


REFERENCE = {"sdpa_efficient_float32": "sdpa_math_float32", "sdpa_efficient_bfloat16": "sdpa_math_bfloat16",
             "sdpa_flash_bfloat16": "sdpa_math_bfloat16", "sdpa_cudnn_bfloat16": "sdpa_math_bfloat16",
             "inductor_attention_float32": "manual_eager_float32", "flex_attention_float32": "flex_eager_float32",
             "hf_eager_attention": "sdpa_math_float32", "xformers_memory_efficient": "sdpa_math_bfloat16",
             "sdpa_math_bfloat16": None, "sdpa_math_float32": None, "sdpa_math_cpu_float64": None,
             "manual_eager_float32": None, "flex_eager_float32": None}


def manual_attention(q, k, v, mask, scale, rep):
    if rep > 1:
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        s = s.masked_fill(~mask[:, None], float("-inf"))
    return torch.softmax(s, -1) @ v


class Impl:
    """per (candidate, condition): builds the callable once (compiled candidates compile once per shape)."""

    def __init__(self, cand, cond):
        self.cand, self.cond = cand, cond
        self.compiled = {}

    def __call__(self, q, k, v, a, gqa):
        cand, cond = self.cand, self.cond
        d = q.shape[-1]
        scale = cond["scale"] if cond["scale"] is not None else 1.0 / d ** 0.5
        rep = q.shape[1] // k.shape[1]
        mask_t = None if cond["mask"] == "none" else torch.as_tensor(a, device=q.device)
        kind = cand["kind"]
        if kind == "sdpa":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            kw = {"scale": cond["scale"]}
            if gqa:
                kw["enable_gqa"] = True
            if cond["mask"] == "causal":
                kw["is_causal"] = True
            elif mask_t is not None:
                kw["attn_mask"] = mask_t[:, None]
            with sdpa_kernel([getattr(SDPBackend, cand["backend"])]):
                return F.scaled_dot_product_attention(q, k, v, **kw)
        if kind == "manual":
            fn = manual_attention
            if cand.get("compiled"):
                key = (tuple(q.shape), tuple(k.shape))
                if key not in self.compiled:
                    self.compiled[key] = torch.compile(manual_attention, dynamic=False)
                fn = self.compiled[key]
            return fn(q, k, v, mask_t, scale, rep)
        if kind == "flex":
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            bm = None
            if mask_t is not None:
                def mask_mod(b, h, qi, ki):
                    return mask_t[b, qi, ki]
                bm = create_block_mask(mask_mod, B, None, q.shape[2], k.shape[2], device=q.device)
            fn = flex_attention
            if cand.get("compiled"):
                key = (tuple(q.shape), tuple(k.shape), cond["mask"])
                if key not in self.compiled:
                    self.compiled[key] = torch.compile(flex_attention, dynamic=False)
                fn = self.compiled[key]
            return fn(q, k, v, block_mask=bm, scale=cond["scale"], enable_gqa=gqa)
        if kind == "hf":
            from transformers.models.llama.modeling_llama import eager_attention_forward
            mod = torch.nn.Module()
            mod.num_key_value_groups, mod.training = rep, False
            add = None
            if mask_t is not None:
                add = torch.zeros(mask_t.shape, dtype=q.dtype, device=q.device).masked_fill(~mask_t, torch.finfo(q.dtype).min)[:, None]
            out, _ = eager_attention_forward(mod, q, k, v, add, scaling=scale)
            return out.transpose(1, 2)
        if kind == "xformers":
            import xformers.ops as xo
            from xformers.ops.fmha import attn_bias as ab
            qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)        # BMHK
            if rep > 1:                                                              # BMGHK with expanded KV heads
                bq, mq, _, dd = qq.shape
                qq = qq.reshape(bq, mq, k.shape[1], rep, dd)
                kk = kk[:, :, :, None].expand(-1, -1, -1, rep, -1)
                vv = vv[:, :, :, None].expand(-1, -1, -1, rep, -1)
            bias = None
            if cond["mask"] == "causal":
                bias = ab.LowerTriangularMask()
            elif mask_t is not None:
                lq, lk = mask_t.shape[1:]
                pad = (lk + 7) // 8 * 8
                buf = torch.zeros((B, q.shape[1], lq, pad), dtype=q.dtype, device=q.device)
                buf[..., :lk] = torch.zeros((B, 1, lq, lk), dtype=q.dtype, device=q.device).masked_fill(~mask_t[:, None], float("-inf"))
                bias = buf[..., :lk]
            o = xo.memory_efficient_attention(qq, kk, vv, attn_bias=bias, scale=scale)
            if rep > 1:
                o = o.reshape(o.shape[0], o.shape[1], -1, o.shape[-1])
            return o.transpose(1, 2)
        raise KeyError(kind)


def run_one(impl, cand, cond, seed, variant):
    dt = getattr(torch, cand["dtype"])
    dev = cand["device"]
    inp = make_inputs(cond, seed, variant)
    gqa = cond["gqa"] and variant != "gqa_repeated"
    k_np, v_np = inp["k"], inp["v"]
    rep = HQ // k_np.shape[1]
    if variant == "gqa_repeated":
        k_np, v_np = np.repeat(k_np, rep, axis=1), np.repeat(v_np, rep, axis=1)
    q = torch.tensor(inp["q"], dtype=dt, device=dev, requires_grad=True)
    k = torch.tensor(k_np, dtype=dt, device=dev, requires_grad=True)
    v = torch.tensor(v_np, dtype=dt, device=dev, requires_grad=True)
    o = impl(q, k, v, inp["allowed"], gqa)
    o.backward(torch.tensor(inp["g"], dtype=dt, device=dev))
    f = lambda t: t.detach().double().cpu().numpy()         # noqa: E731
    dk, dv = f(k.grad), f(v.grad)
    if variant == "gqa_repeated":                            # gradients of the shared KV heads: sum over each group
        dk = dk.reshape(B, -1, rep, *dk.shape[2:]).sum(2)
        dv = dv.reshape(B, -1, rep, *dv.shape[2:]).sum(2)
    return {"out": f(o), "dq": f(q.grad), "dk": dk, "dv": dv}


def run(env, only=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    for cand in candidates_for(env):
        if only and cand["id"] not in only:
            continue
        path = CACHE / f"{cand['id']}.pkl"
        if path.exists():
            continue
        res, t0, slowest = {}, time.time(), (0.0, None)
        for cond in conditions():
            if cand.get("compiled"):
                torch._dynamo.reset()
            impl = Impl(cand, cond)
            for seed in SEEDS:
                rec = {}
                for var in variants_for(cond):
                    t1 = time.time()
                    try:
                        rec[var] = {"status": "ok", "outputs": run_one(impl, cand, cond, seed, var)}
                    except Exception as exc:  # noqa: BLE001
                        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                        st = ("unsupported" if any(x in msg for x in ("No available kernel", "No operator found"))
                              or "not supported" in msg.lower() else "error")
                        rec[var] = {"status": st, "reason": msg}
                        if "device-side assert" in msg or "CUDA error" in msg:
                            raise
                    dt = time.time() - t1
                    if dt > slowest[0]:
                        slowest = (dt, f"{cond['id']}/{seed}/{var}")
                res[(cond["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
                "slowest": [round(slowest[0], 2), slowest[1]], "env": env}
        path.write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] += 1
        print(cand["id"], dict(st), meta["seconds"], "s, slowest", meta["slowest"], flush=True)


# ------------------------------------------------------------------------------------------------ analysis

def _classes(x):
    return np.where(np.isnan(x), 1, np.where(np.isposinf(x), 2, np.where(np.isneginf(x), 3, 0)))


def compare(k, r, dtype, mask):
    """E on the elements in mask: (violations, compared, max ratio, class mismatches).  bf16: recorded, ratio vs 2^-8."""
    k, r = k[mask], r[mask]
    cls = _classes(k) != _classes(r)
    fin = (_classes(k) == 0) & (_classes(r) == 0)
    tau = TAU.get(dtype, 2.0 ** -8)
    lim = tau * (1 + np.abs(r))
    d = np.abs(k - r)
    bad = fin & (d > lim)
    ratio = float(np.max(np.where(fin, d / lim, 0), initial=0))
    return int((bad | cls).sum()), int(mask.sum()), ratio, int(cls.sum())


def row_mask(defined, shape, name):
    """element mask for an output / gradient of shape (B, H, L, D) from the defined (B, Lq) rows."""
    if name in ("out", "dq"):
        return np.broadcast_to(defined[:, None, :, None], shape)
    return np.ones(shape, dtype=bool)


def convention(o, a):
    """what a candidate returns on rows without an allowed key."""
    und = ~a.any(-1)
    if not und.any():
        return None
    vals = o["out"].transpose(0, 2, 1, 3)[und]                    # (n_rows, H, D)
    if np.isnan(vals).all():
        c = "nan"
    elif np.all(vals == 0):
        c = "zero"
    elif np.isfinite(vals).all():
        c = "finite_nonzero"
    else:
        c = "mixed"
    pollution = not all(np.isfinite(o[g]).all() for g in ("dk", "dv"))
    return {"forward": c, "nonfinite_kv_gradients": bool(pollution)}


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"conditions": len(conds), "seeds": list(SEEDS), "candidates": {}, "examples": defaultdict(list)}
    for name, d in data.items():
        cand = d["meta"]["candidate"]
        dtype = cand["dtype"]
        judged = dtype in TAU
        st = defaultdict(int)
        unsup = defaultdict(int)
        for (cid, seed), rec in d["res"].items():
            for var, r in rec.items():
                st[r["status"]] += 1
                if r["status"] != "ok" and var == "base":
                    unsup[f"{conds[cid]['q_len_k_len']}|{conds[cid]['mask']}|gqa={conds[cid]['gqa']}|d{conds[cid]['head_dim']}: {r['reason'][:80]}"] += 1
        entry = {"meta": {k: v for k, v in d["meta"].items() if k != "candidate"}, "statuses": dict(st),
                 "base_not_run": dict(unsup), "judged": judged, "E": None, "P": defaultdict(lambda: [0, 0]),
                 "conventions": defaultdict(int), "determinism": [0, 0]}
        ref = REFERENCE.get(name)
        e = {"conditions_compared": 0, "violating_conditions": 0, "class_mismatch_conditions": 0, "max_ratio": 0.0,
             "by_output": defaultdict(lambda: [0, 0])} if ref and ref in data else None
        viol_cond = set()
        for (cid, seed), rec in d["res"].items():
            cond = conds[cid]
            a = allowed(cond)
            defined = a.any(-1)
            b = rec.get("base")
            if not b or b["status"] != "ok":
                continue
            o = b["outputs"]
            c = convention(o, a)
            if c:
                entry["conventions"][f"{c['forward']}{' + nonfinite dK/dV' if c['nonfinite_kv_gradients'] else ''}"] += 1
            rp = rec.get("repeat")
            det = rp and rp["status"] == "ok" and all(np.array_equal(o[x], rp["outputs"][x], equal_nan=True) for x in o)
            entry["determinism"][0] += 1
            entry["determinism"][1] += int(bool(det))
            # E
            if e is not None:
                rb = data[ref]["res"].get((cid, seed), {}).get("base")
                if rb and rb["status"] == "ok":
                    any_bad = False
                    for x in ("out", "dq", "dk", "dv"):
                        m = row_mask(defined, o[x].shape, x)
                        if x in ("dk", "dv") and (c and c["nonfinite_kv_gradients"] or convention(rb["outputs"], a) and convention(rb["outputs"], a)["nonfinite_kv_gradients"]):
                            continue                                  # undefined rows pollute dK/dV: convention, recorded above
                        nb, nc, ratio, ncls = compare(o[x], rb["outputs"][x], dtype, m)
                        e["by_output"][x][0] += int(nb > 0)
                        e["by_output"][x][1] += 1
                        e["max_ratio"] = max(e["max_ratio"], ratio)
                        if nb:
                            any_bad = True
                            if len(out["examples"]["E"]) < 40:
                                out["examples"]["E"].append({"candidate": name, "condition": cid, "seed": seed, "output": x,
                                                             "violating": nb, "of": nc, "class_mismatch": ncls, "max_ratio": ratio})
                    e["conditions_compared"] += 1
                    if any_bad:
                        viol_cond.add(cid)
            # P
            tau = TAU.get(dtype)
            pt = rec.get("perturb")
            if pt and pt["status"] == "ok":
                rows = unaffected_rows(cond, a)
                m = rows[:, None, :, None]
                same = all(np.array_equal(np.broadcast_to(m, o[x].shape) * np.nan_to_num(o[x]),
                                          np.broadcast_to(m, o[x].shape) * np.nan_to_num(pt["outputs"][x])) for x in ("out", "dq"))
                key = "P_causal_independence" if cond["mask"] == "causal" else "P_masked_keys_no_influence"
                entry["P"][key][0] += 1
                entry["P"][key][1] += int(not same)
                if not same:
                    diff = max(float(np.max(np.abs(np.where(np.broadcast_to(m, o[x].shape), o[x] - pt["outputs"][x], 0)))) for x in ("out", "dq"))
                    if len(out["examples"]["P"]) < 40:
                        out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": key,
                                                     "deterministic": bool(det), "max_abs_change": diff})
                # keys masked for every row get zero gradient
                dead = ~a.any(1)                                  # (B, lk)
                if dead.any():
                    rep_ = o["dk"].shape[1]
                    z = all(np.all(o[x].transpose(0, 2, 1, 3)[dead] == 0) for x in ("dk", "dv"))
                    entry["P"]["P_dead_keys_zero_grad"][0] += 1
                    entry["P"]["P_dead_keys_zero_grad"][1] += int(not z)
            vo = rec.get("v_ones")
            if vo and vo["status"] == "ok":
                x = vo["outputs"]["out"].transpose(0, 2, 1, 3)[defined]
                lim = (tau if tau else 2.0 ** -8) * 2
                ok = bool(np.all(np.abs(x - 1) <= lim))
                entry["P"]["P_rows_sum_to_one"][0] += 1
                entry["P"]["P_rows_sum_to_one"][1] += int(not ok)
                if not ok and len(out["examples"]["P"]) < 40:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_rows_sum_to_one",
                                                 "max_abs_dev": float(np.max(np.abs(x - 1)))})
            gr = rec.get("gqa_repeated")
            if gr and gr["status"] == "ok":
                ok = True
                for x in ("out", "dq", "dk", "dv"):
                    nb, _, _, _ = compare(o[x], gr["outputs"][x], dtype if judged else "bfloat16", row_mask(defined, o[x].shape, x))
                    ok &= nb == 0
                entry["P"]["P_gqa_equals_repeated_kv"][0] += 1
                entry["P"]["P_gqa_equals_repeated_kv"][1] += int(not ok)
                if not ok and len(out["examples"]["P"]) < 40:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_gqa_equals_repeated_kv"})
        if e is not None:
            e["violating_conditions"] = len(viol_cond)
            e["violating_condition_ids"] = sorted(viol_cond)
            e["by_output"] = {k: f"{v[0]}/{v[1]}" for k, v in e["by_output"].items()}
            e["reference"] = ref
            e["status"] = "judged" if judged else "recorded (bf16)"
        entry["E"] = e
        entry["P"] = {k: f"{v[1]}/{v[0]}" for k, v in entry["P"].items()}
        entry["conventions"] = dict(entry["conventions"])
        entry["determinism"] = f"{entry['determinism'][1]}/{entry['determinism'][0]}"
        out["candidates"][name] = entry
        print(f"{name:28s} {dict(st)}  E: " + (f"{e['status']} viol-cond {e['violating_conditions']}/{len({k[0] for k in d['res']})} "
              f"{e['by_output']} max {e['max_ratio']:.3g}" if e else "-"))
        print(f"{'':28s} P {entry['P']}  conv {entry['conventions']}  det {entry['determinism']}")
        for k, v in unsup.items():
            print(f"{'':28s} not run: {k} x{v}")
    out["examples"] = dict(out["examples"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.env, a.only) if a.stage == "run" else analyse()
