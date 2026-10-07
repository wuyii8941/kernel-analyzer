#!/usr/bin/env python3
"""2b basic-operator families (protocol v2 section 3; registry tier1.matmul_linear / reductions / activations /
gather_layout): E, E64 and P, forward and backward.  F and mode-B FR stay closed until the specs are delivered.

One harness for the four families: each family gives its conditions (the coverage plan, de-duplicated), the inputs
(float32-representable values for every candidate, so the float64 candidates run on the same values), the operation, and
its P variants.  Candidates: eager CPU / CUDA float64 and float32, CUDA bfloat16 (recorded), Inductor float32 / bfloat16,
nightly eager / Inductor float32; in the liger environment the registered library candidates (Liger softmax, Liger and
Unsloth SwiGLU) for the conditions they implement.

Shapes.  matmul_linear: (1,k)x(k,1) = (1, 37) @ (37, 1); (m,0)x(0,n) = (5, 0) @ (0, 6); (37,129)x(129,61); batch-broadcast
(2, 3, 5, 7) @ (7, 4); bias "vector" (n,) via addmm / matmul + bias, "broadcast" (1, n).  reductions: size s -> x of shape
(3, s) reduced over the last axis (single), (3, 4, s) over the last two (multi), all axes (all), (3, 0) over the empty axis
(empty-extent); var / std with the default correction (1).  activations: (6, 33) (swiglu / geglu: two such inputs a, b).
gather_layout: source (6, 9), indices along dim 1 (distinct / repeated / all_same).

    python scripts/essential/p2b_basic.py run --family reductions --env ka_main
    python scripts/essential/p2b_basic.py analyse --family reductions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/basic"
OUT = ROOT / "results/essential/phase2b"
SEEDS = (0, 1, 2)
TAU = {"float32": 2.0 ** -12, "float64": 1e-9}


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def f32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def plan(family, keys):
    p = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"][family]["conditions"]
    out, seen = [], {}
    for c in p:
        key = tuple(str(c[k]) for k in keys)
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = dict(c, high_risk=bool(c.get("high_risk")))
        d["id"] = family[:4] + "_" + "_".join(str(c[k]).replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
                                               for k in keys)
        seen[key] = d
        out.append(d)
    return out


def layout(a, kind, dtype, device):
    t = torch.as_tensor(a).to(dtype)
    if kind == "transposed" and t.dim() >= 2:
        return t.transpose(-1, -2).contiguous().to(device).transpose(-1, -2)
    if kind == "offset":
        flat = torch.zeros(t.numel() + 3, dtype=dtype)
        flat[3:] = t.reshape(-1)
        return flat.to(device)[3:].view(t.shape)
    return t.to(device)


# ------------------------------------------------------------------------------------------------ matmul_linear

class Matmul:
    keys = ("shape", "layout", "bias", "values")
    SHAPES = {"(1,k)x(k,1)": ((1, 37), (37, 1)), "(m,0)x(0,n)": ((5, 0), (0, 6)), "(37,129)x(129,61)": ((37, 129), (129, 61)),
              "batch-broadcast": ((2, 3, 5, 7), (7, 4))}

    def conditions(self):
        return plan("matmul_linear", self.keys)

    def values(self, rng, kind, shape):
        if kind == "small_ints":
            return rng.integers(-3, 4, shape).astype(float)
        if kind == "cancel":
            x = rng.normal(0, 1, shape)
            return np.concatenate([x, -x], axis=-1)[..., :shape[-1]] if shape[-1] else x
        if kind == "mixed_scale":
            return rng.normal(0, 1, shape) * 10.0 ** rng.integers(-6, 7, shape)
        return rng.normal(0, 1, shape)

    def inputs(self, c, seed, variant):
        rng = _rng(f"mm/{c['id']}/{seed}")
        sa, sb = self.SHAPES[c["shape"]]
        a, b = f32(self.values(rng, c["values"], sa)), f32(self.values(rng, c["values"], sb))
        n = sb[-1]
        bias = None if c["bias"] is None else f32(self.values(rng, c["values"], (n,) if c["bias"] == "vector" else (1, n)))
        a2 = f32(self.values(rng, c["values"], sa))
        out_shape = np.broadcast_shapes(sa[:-1] + (1,), (1,) * (len(sa) - 1) + (n,)) if len(sa) > 2 else (sa[0], n)
        g = f32(rng.integers(-2, 3, out_shape).astype(float) if c["values"] == "small_ints" else rng.normal(0, 1, out_shape))
        if variant == "lin_a2":
            a = a2
        elif variant == "lin_sum":
            a = a + a2
        return {"a": a, "b": b, "bias": bias, "g": g}

    def variants(self, c):
        v = ["base", "lin_a2", "lin_sum", "transpose_identity", "accumulate_twice"]
        return v

    def run(self, c, t, variant):
        a, b = t["a"], t["b"]
        if variant == "transpose_identity":
            return {"out": torch.matmul(b.transpose(-1, -2), a.transpose(-1, -2)).transpose(-1, -2)}
        if t.get("bias") is not None:
            out = torch.addmm(t["bias"], a, b) if a.dim() == 2 else torch.matmul(a, b) + t["bias"]
        else:
            out = torch.matmul(a, b)
        return {"out": out}

    def grad_inputs(self, c):
        return ("a", "b", "bias")


# ------------------------------------------------------------------------------------------------ reductions

class Reductions:
    keys = ("op", "dims", "values", "size")

    def conditions(self):
        return plan("reductions", self.keys)

    def shape(self, c):
        s = int(c["size"])
        return {"single": (3, s), "multi": (3, 4, s), "all": (3, s), "empty-extent": (3, 0)}[c["dims"]]

    def dims(self, c):
        return {"single": (-1,), "multi": (-2, -1), "all": None, "empty-extent": (-1,)}[c["dims"]]

    def inputs(self, c, seed, variant):
        rng = _rng(f"red/{c['id']}/{seed}")
        shp = self.shape(c)
        v = c["values"]
        if v == "small_ints":
            x = rng.integers(-3, 4, shp).astype(float)
        elif v == "all_equal":
            x = np.full(shp, 1.25)
        elif v == "zeros":
            x = np.zeros(shp)
        elif v == "cancel":
            x = rng.normal(0, 1, shp) * 1e3
            if shp[-1] >= 2:
                x[..., 1] = -x[..., 0]
        else:
            x = rng.normal(0, 1, shp)
        if v == "neg_inf_row" and x.size:
            x[0] = -np.inf                                   # one fully -inf row
        x = f32(x)
        if variant == "shifted":
            x = x + 3.0                                      # exact for |x| < 2^20 at float32 spacing? (shift by a power-of-two-free constant)
        if variant == "permuted" and x.size:
            perm = _rng(f"red-perm/{c['id']}/{seed}").permutation(x.shape[-1])
            x = x[..., perm]
        out_shape = self.out_shape(c, shp)
        g = f32(rng.normal(0, 1, out_shape)) if out_shape is not None else None
        return {"x": x, "g": g}

    def out_shape(self, c, shp):
        op = c["op"]
        if op in ("softmax", "log_softmax", "cumsum"):
            return shp
        d = self.dims(c)
        if d is None:
            return ()
        keep = [s for i, s in enumerate(shp) if i - len(shp) not in d]
        return tuple(keep)

    def variants(self, c):
        v = ["base"]
        if c["op"] in ("softmax", "log_softmax", "logsumexp"):
            v.append("shifted")
        if c["op"] in ("sum", "prod", "amax", "amin") and c["dims"] in ("single", "all") and c["values"] == "small_ints":
            v.append("permuted")
        return v

    def run(self, c, t, variant):
        x, op, d = t["x"], c["op"], self.dims(c)
        if op in ("softmax", "log_softmax", "cumsum"):
            dim = -1 if d is None or len(d) == 1 else d[-1]
            if d is None or (d is not None and len(d) > 1):  # multi / all: normalise over the flattened reduced axes
                flat = x.reshape(x.shape[0], -1) if d is not None else x.reshape(1, -1)
                y = getattr(torch, op)(flat, -1) if op != "cumsum" else torch.cumsum(flat, -1)
                return {"out": y.reshape(x.shape)}
            return {"out": getattr(torch, op)(x, dim) if op != "cumsum" else torch.cumsum(x, dim)}
        kw = {} if d is None else {"dim": d}
        if op in ("sum", "mean", "amax", "amin", "logsumexp", "var", "std"):
            if op in ("amax", "amin"):
                return {"out": getattr(torch, op)(x, **({"dim": d} if d is not None else {"dim": tuple(range(x.dim()))}))}
            if op == "logsumexp":
                return {"out": torch.logsumexp(x, dim=d if d is not None else tuple(range(x.dim())))}
            return {"out": getattr(torch, op)(x, **kw)}
        if op == "prod":
            if d is None:
                return {"out": torch.prod(x)}
            y = x
            for dd in sorted(d):                             # prod has no multi-dim form: reduce one axis at a time
                y = torch.prod(y, dim=dd if dd < 0 else dd, keepdim=True)
            return {"out": y.reshape(self.out_shape(c, tuple(x.shape)))}
        raise KeyError(op)

    def grad_inputs(self, c):
        return ("x",)


# ------------------------------------------------------------------------------------------------ activations

class Activations:
    keys = ("op", "values", "layout")

    def conditions(self):
        return plan("activations", self.keys)

    def inputs(self, c, seed, variant):
        rng = _rng(f"act/{c['id']}/{seed}")
        shp = (6, 33)
        def vals():
            v = c["values"]
            if v == "zeros":
                return np.zeros(shp)
            if v == "huge":
                return rng.normal(0, 1, shp) * 1e30
            if v == "tiny":
                return rng.normal(0, 1, shp) * 1e-30
            return rng.normal(0, 2, shp)
        a, b = f32(vals()), f32(vals())
        if variant == "negated":
            a = -a
        g = f32(rng.normal(0, 1, shp))
        return {"a": a, "b": b, "g": g}

    def variants(self, c):
        v = ["base"]
        if c["op"] in ("silu", "gelu_erf", "gelu_tanh"):
            v.append("negated")
        return v

    def run(self, c, t, variant):
        a, b, op = t["a"], t["b"], c["op"]
        if op == "relu":
            return {"out": F.relu(a)}
        if op == "silu":
            return {"out": F.silu(a)}
        if op == "gelu_erf":
            return {"out": F.gelu(a)}
        if op == "gelu_tanh":
            return {"out": F.gelu(a, approximate="tanh")}
        if op == "swiglu":
            return {"out": F.silu(a) * b}
        return {"out": F.gelu(a) * b}

    def grad_inputs(self, c):
        return ("a", "b") if c["op"] in ("swiglu", "geglu") else ("a",)


# ------------------------------------------------------------------------------------------------ gather_layout

class Gather:
    keys = ("op", "index", "layout")

    def conditions(self):
        return plan("gather_layout", self.keys)

    def inputs(self, c, seed, variant):
        rng = _rng(f"gat/{c['id']}/{seed}")
        x = f32(rng.normal(0, 1, (6, 9)))
        k = c["index"]
        if k == "distinct":
            idx = np.stack([rng.permutation(9)[:5] for _ in range(6)])
        elif k == "repeated":
            idx = rng.integers(0, 4, (6, 5))
        else:
            idx = np.full((6, 5), 2)
        if variant == "ones_upstream":
            g = np.ones((6, 5) if c["op"] in ("gather", "take_along_dim") else (6, 5))
        else:
            g = f32(rng.normal(0, 1, (6, 5)))
        return {"x": x, "idx": idx, "g": g, "v": f32(rng.normal(0, 1, (6, 9)))}

    def variants(self, c):
        return ["base", "ones_upstream"] if c["op"] in ("gather", "index_select", "take_along_dim") else ["base"]

    def run(self, c, t, variant):
        x, idx, op = t["x"], t["idx"], c["op"]
        if op == "gather":
            return {"out": torch.gather(x, 1, idx)}
        if op == "take_along_dim":
            return {"out": torch.take_along_dim(x, idx, 1)}
        if op == "index_select":
            return {"out": torch.index_select(x, 1, idx[0])[:, :5].reshape(6, 5)}
        if op == "cat_split":
            parts = torch.split(x, [2, 3, 4], dim=1)
            y = torch.cat(parts, dim=1)
            return {"out": y[:, :5] * 1.0}
        # view_alias: write through a view of a fresh copy (inputs are not mutated); the result must show the write
        base = x.clone()
        view = base.transpose(0, 1)[1:4]                     # (3, 6) view of columns 1..3
        view.mul_(2.0)
        return {"out": base[:, :5] * 1.0}

    def grad_inputs(self, c):
        return ("x",)


FAMILIES = {"matmul_linear": Matmul(), "reductions": Reductions(), "activations": Activations(), "gather_layout": Gather()}


# ------------------------------------------------------------------------------------------------ candidates / driver

def candidates_for(env):
    if env == "ka_main":
        out = [{"id": f"eager_{d}_{t}", "device": d, "dtype": t} for d, t in
               (("cpu", "float64"), ("cuda", "float64"), ("cpu", "float32"), ("cuda", "float32"), ("cuda", "bfloat16"))]
        return out + [{"id": f"inductor_cuda_{t}", "device": "cuda", "dtype": t, "compiled": True} for t in ("float32", "bfloat16")]
    if env == "nightly":
        return [{"id": "nightly_eager_cuda_float32", "device": "cuda", "dtype": "float32"},
                {"id": "nightly_inductor_cuda_float32", "device": "cuda", "dtype": "float32", "compiled": True}]
    if env == "liger":
        return [{"id": "liger_softmax", "device": "cuda", "dtype": "float32", "lib": "liger_softmax", "family": "reductions"},
                {"id": "liger_swiglu", "device": "cuda", "dtype": "float32", "lib": "liger_swiglu", "family": "activations"},
                {"id": "unsloth_swiglu", "device": "cuda", "dtype": "bfloat16", "lib": "unsloth_swiglu", "family": "activations"}]
    raise KeyError(env)


REFERENCE = {"inductor_cuda_float32": "eager_cuda_float32", "inductor_cuda_bfloat16": "eager_cuda_bfloat16",
             "nightly_inductor_cuda_float32": "nightly_eager_cuda_float32", "nightly_eager_cuda_float32": "eager_cuda_float32",
             "eager_cuda_float64": "eager_cpu_float64", "liger_softmax": "eager_cuda_float32", "liger_swiglu": "eager_cuda_float32",
             "unsloth_swiglu": "eager_cuda_bfloat16"}


def lib_call(cand, fam, c, t):
    lib = cand["lib"]
    if lib == "liger_softmax":
        if c["op"] != "softmax" or c["dims"] != "single":
            raise NotImplementedError("Liger softmax: last-axis softmax only")
        from liger_kernel.transformers.functional import liger_softmax
        return {"out": liger_softmax(t["x"])}
    if lib == "liger_swiglu":
        if c["op"] != "swiglu":
            raise NotImplementedError("Liger SwiGLU only")
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        return {"out": LigerSiLUMulFunction.apply(t["a"], t["b"])}
    if lib == "unsloth_swiglu":
        if c["op"] != "swiglu":
            raise NotImplementedError("Unsloth SwiGLU only")
        from unsloth.kernels.swiglu import swiglu_fg_kernel                     # expects (batch, seq, hidden)
        return {"out": swiglu_fg_kernel(t["a"].contiguous()[None], t["b"].contiguous()[None])[0]}
    raise KeyError(lib)


def run_one(fam_name, cand, c, seed, variant, fn):
    fam = FAMILIES[fam_name]
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    inp = fam.inputs(c, seed, variant)
    lay = c.get("layout", "contiguous")
    gi = fam.grad_inputs(c)
    t = {}
    for k, v in inp.items():
        if v is None:
            t[k] = None
        elif k in ("idx",):
            t[k] = torch.as_tensor(v, dtype=torch.long, device=dev)
        elif k == "g":
            t[k] = torch.as_tensor(v).to(dt).to(dev)
        else:
            t[k] = layout(v, lay if k in gi else "contiguous", dt, dev)
            if k in gi:
                t[k].requires_grad_(True)
    outs = fn(t, variant)
    f = lambda a: a.detach().double().cpu().numpy().copy()       # noqa: E731
    res = {k: f(v) for k, v in outs.items()}
    if t.get("g") is not None and outs["out"].requires_grad and outs["out"].numel():
        reps = 2 if variant == "accumulate_twice" else 1
        for _ in range(reps):
            outs2 = fn(t, variant) if reps > 1 and _ > 0 else outs
            outs2["out"].backward(t["g"].reshape(outs2["out"].shape))
        for k in gi:
            if t.get(k) is not None and t[k].grad is not None:
                res["d" + k] = f(t[k].grad)
    return res


def run(fam_name, env):
    fam = FAMILIES[fam_name]
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for cand in candidates_for(env):
        if cand.get("family") not in (None, fam_name):
            continue
        path = CACHE / f"{fam_name}__{cand['id']}.pkl"
        if path.exists():
            continue
        res, t0 = {}, time.time()
        for c in fam.conditions():
            base_fn = (lambda t, v, _c=c: lib_call(cand, fam, _c, t)) if cand.get("lib") else (lambda t, v, _c=c: fam.run(_c, t, v))
            fn = base_fn
            if cand.get("compiled"):
                torch._dynamo.reset()
                comp = {}

                def fn(t, v, _c=c, _comp=comp):
                    if v not in _comp:
                        _comp[v] = torch.compile(lambda tt: fam.run(_c, tt, v), dynamic=False)
                    return _comp[v](t)
            for seed in SEEDS:
                rec = {}
                for var in fam.variants(c):
                    try:
                        rec[var] = {"status": "ok", "outputs": run_one(fam_name, cand, c, seed, var, fn)}
                    except NotImplementedError as exc:
                        rec[var] = {"status": "unsupported", "reason": str(exc)}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"}
                res[(c["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1), "env": env, "family": fam_name}
        path.write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] += 1
        print(fam_name, cand["id"], dict(st), meta["seconds"], "s", flush=True)


# ------------------------------------------------------------------------------------------------ analysis

def deviation(k, r, dtype, exact=False):
    k, r = np.asarray(k, float), np.asarray(r, float)
    if k.shape != r.shape:
        return k.size or 1, np.inf
    fin = np.isfinite(k) & np.isfinite(r)
    cls = ~fin & ~((np.isnan(k) & np.isnan(r)) | (k == r))
    if exact:
        return int(((fin & (k != r)) | cls).sum()), 0.0
    tau = TAU.get(dtype, 2.0 ** -8)
    d = np.abs(k - r)
    lim = tau * (1 + np.abs(r))
    return int(((fin & (d > lim)) | cls).sum()), float(np.max(np.where(fin, d / lim, 0), initial=0))


def exact_inputs(fam_name, c):
    """documented exactness: small-integer sums / products / extrema and matmuls are exactly representable."""
    if fam_name == "matmul_linear":
        return c["values"] == "small_ints"
    if fam_name == "reductions":
        return c["values"] in ("small_ints", "zeros") and c["op"] in ("sum", "amax", "amin", "cumsum")
    if fam_name == "gather_layout":
        return True                                          # pure data movement (backward: sums of upstream values)
    return False


def p_checks(fam_name, c, rec, inp_fn, dtype, judged):
    """family P properties -> {name: True/False (violated?) or None}."""
    res = {}
    b = rec.get("base")
    if not b or b["status"] != "ok":
        return res
    o = b["outputs"]
    tau = TAU.get(dtype, 2.0 ** -8)
    if fam_name == "matmul_linear":
        l2, ls = rec.get("lin_a2"), rec.get("lin_sum")
        if l2 and ls and l2["status"] == "ok" and ls["status"] == "ok" and c["bias"] is None:
            lhs, rhs = ls["outputs"]["out"], o["out"] + l2["outputs"]["out"]
            n, _ = deviation(lhs, rhs, dtype, exact=c["values"] == "small_ints")
            res["P_linearity"] = n > 0 if (judged or c["values"] == "small_ints") else None
        tr = rec.get("transpose_identity")
        if tr and tr["status"] == "ok" and c["bias"] is None:
            n, _ = deviation(tr["outputs"]["out"], o["out"], dtype, exact=c["values"] == "small_ints")
            res["P_transpose_identity"] = n > 0 if (judged or c["values"] == "small_ints") else None
        acc = rec.get("accumulate_twice")
        if acc and acc["status"] == "ok":
            bad = False
            for k in ("da", "db", "dbias"):
                if k in o and k in acc["outputs"]:
                    bad |= not np.array_equal(acc["outputs"][k], 2 * o[k], equal_nan=True)
            res["P_grad_accumulates"] = bad
    elif fam_name == "reductions":
        sh = rec.get("shifted")
        if sh and sh["status"] == "ok" and judged and c["values"] != "neg_inf_row":
            if c["op"] == "logsumexp":
                n, _ = deviation(sh["outputs"]["out"], o["out"] + 3.0, dtype)
                res["P_logsumexp_shift"] = n > 0
            else:
                n, _ = deviation(sh["outputs"]["out"], o["out"], dtype)
                res["P_softmax_shift_invariance"] = n > 0
        pm = rec.get("permuted")
        if pm and pm["status"] == "ok":
            res["P_permutation_invariance"] = not np.array_equal(pm["outputs"]["out"], o["out"], equal_nan=True)
        if c["op"] == "cumsum" and judged and c["dims"] == "single" and o["out"].size:
            inp = inp_fn("base")
            total = np.asarray(inp["x"]).sum(-1)
            n, _ = deviation(o["out"][..., -1], total, dtype)
            res["P_cumsum_last_equals_sum"] = n > 0
        if c["op"] in ("amax", "amin") and "dx" in o and o["out"].size:
            inp = inp_fn("base")
            x, g = np.asarray(inp["x"]), np.asarray(inp["g"])
            red = (-1,) if c["dims"] == "single" else ((-2, -1) if c["dims"] == "multi" else tuple(range(x.ndim)))
            ext = x.max(axis=red, keepdims=True) if c["op"] == "amax" else x.min(axis=red, keepdims=True)
            hit = (x == ext)
            cnt = hit.sum(axis=red, keepdims=True)
            expect = np.where(hit, np.reshape(g, ext.shape) / cnt, 0.0)       # documented: evenly distributed among ties
            n, _ = deviation(o["dx"], expect, dtype, exact=c["values"] == "small_ints" and dtype == "float64")
            res["P_D1_amax_ties_even_split"] = n > 0
    elif fam_name == "activations":
        ng = rec.get("negated")
        if ng and ng["status"] == "ok" and judged and c["values"] != "huge":
            inp = inp_fn("base")
            a = np.asarray(inp["a"])
            n, _ = deviation(o["out"] - ng["outputs"]["out"], a, dtype)
            res["P_odd_identity_f(x)-f(-x)=x"] = n > 0
        if c["op"] == "relu" and "da" in o:
            inp = inp_fn("base")
            a, g = np.asarray(inp["a"]), np.asarray(inp["g"])
            zero = a == 0
            if zero.any():
                res["P_relu_grad_at_zero_is_zero"] = bool(np.any(o["da"][zero] != 0))
    elif fam_name == "gather_layout":
        ou = rec.get("ones_upstream")
        if ou and ou["status"] == "ok" and "dx" in ou["outputs"]:
            inp = inp_fn("ones_upstream")
            idx = np.asarray(inp["idx"])
            cnt = np.zeros((6, 9))
            if c["op"] == "index_select":
                sel = idx[0]
                for j in range(5):
                    cnt[:, sel[j]] += 1
            else:
                for r in range(6):
                    for j in range(5):
                        cnt[r, idx[r, j]] += 1
            res["P_gather_backward_counts"] = not np.array_equal(ou["outputs"]["dx"], cnt)
        rnd = lambda a: torch.as_tensor(a).to(getattr(torch, dtype)).double().numpy()    # noqa: E731  (candidate dtype)
        if c["op"] == "cat_split":
            inp = inp_fn("base")
            res["P_split_cat_identity"] = not np.array_equal(o["out"], rnd(np.asarray(inp["x"]))[:, :5])
        if c["op"] == "view_alias":
            inp = inp_fn("base")
            x = rnd(np.asarray(inp["x"])).copy()
            x[:, 1:4] *= 2.0
            res["P_view_write_through"] = not np.array_equal(o["out"], x[:, :5])
    return res


def analyse(fam_name):
    fam = FAMILIES[fam_name]
    files = sorted(CACHE.glob(f"{fam_name}__*.pkl"))
    data = {p.stem.split("__", 1)[1]: pickle.loads(p.read_bytes()) for p in files}
    conds = {c["id"]: c for c in fam.conditions()}
    out = {"family": fam_name, "conditions": len(conds), "candidates": {}, "examples": defaultdict(list)}
    for name, d in data.items():
        dtype = d["meta"]["candidate"]["dtype"]
        judged = dtype in TAU
        st = defaultdict(int)
        errs = defaultdict(int)
        for (cid, seed), rec in d["res"].items():
            for var, r in rec.items():
                st[r["status"]] += 1
                if r["status"] == "error":
                    errs[f"{cid}/{var}: {r['reason'][:120]}"] += 1
        ref = REFERENCE.get(name)
        E, E64 = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        e_conds, e64_conds, emax = set(), set(), 0.0
        P = defaultdict(lambda: [0, 0])
        conv = defaultdict(int)
        for (cid, seed), rec in d["res"].items():
            c = conds[cid]
            b = rec.get("base")
            if not b or b["status"] != "ok":
                if b and b["status"] == "error":
                    conv[f"{cid}: raises {b['reason'][:90]}"] += 1
                continue
            o = b["outputs"]
            ex = exact_inputs(fam_name, c)
            for tgt, acc, cset in ((ref, E, e_conds), ("eager_cpu_float64", E64, e64_conds)):
                if tgt not in data or tgt == name:
                    continue
                rb = data[tgt]["res"].get((cid, seed), {}).get("base")
                if not rb or rb["status"] != "ok":
                    continue
                for k in o:
                    if k not in rb["outputs"]:
                        continue
                    n, ratio = deviation(o[k], rb["outputs"][k], dtype, exact=ex and judged and k == "out")
                    acc[k][0] += 1
                    acc[k][1] += int(n > 0)
                    if tgt == ref:
                        emax = max(emax, ratio)
                    if n:
                        cset.add(cid)
                        if len(out["examples"]["E"]) < 60:
                            out["examples"]["E"].append({"candidate": name, "vs": tgt, "condition": cid, "seed": seed, "output": k,
                                                         "violating": n, "max_ratio": ratio})
            for k, v in p_checks(fam_name, c, rec, lambda var, _c=c, _s=seed: fam.inputs(_c, _s, var), dtype, judged).items():
                if v is None:
                    continue
                P[k][0] += 1
                P[k][1] += int(v)
                if v and len(out["examples"]["P"]) < 60:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": k})
            if not np.all(np.isfinite(o["out"])) and seed == 0:
                conv[f"{cid}: non-finite output ({int((~np.isfinite(o['out'])).sum())} elements)"] += 1
        entry = {"meta": {k: v for k, v in d["meta"].items() if k != "candidate"}, "statuses": dict(st), "judged": judged,
                 "errors": dict(errs),
                 "E": {"reference": ref, "by_output": {k: f"{v[1]}/{v[0]}" for k, v in E.items()}, "violating_conditions": sorted(e_conds),
                       "max_ratio": emax} if ref in data else None,
                 "E64": {"by_output": {k: f"{v[1]}/{v[0]}" for k, v in E64.items()}, "violating_conditions": sorted(e64_conds)}
                 if name != "eager_cpu_float64" and "eager_cpu_float64" in data else None,
                 "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()}, "conventions_seed0": dict(conv)}
        out["candidates"][name] = entry
        print(f"{name:30s} {dict(st)}")
        if entry["E"]:
            print(f"{'':30s} E vs {ref}: {entry['E']['by_output']} {entry['E']['violating_conditions'][:5]} max {emax:.3g}")
        if entry["E64"]:
            print(f"{'':30s} E64: {entry['E64']['by_output']} {entry['E64']['violating_conditions'][:8]} ({len(e64_conds)})")
        print(f"{'':30s} P {entry['P']}")
        for k in list(errs)[:6]:
            print(f"{'':30s} error: {k} x{errs[k]}")
    out["examples"] = dict(out["examples"])
    (OUT / fam_name).mkdir(parents=True, exist_ok=True)
    (OUT / fam_name / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.family, a.env) if a.stage == "run" else analyse(a.family)
