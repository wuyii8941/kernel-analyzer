"""FR group (protocol section 5): Inductor candidates through the tool's mode B, f from the specification cache.

Per condition: ``check.run`` with the three seeds (compile and warm-up in ``setup``, outside the recorder, so
autotuning launches are not captured); keeps K, K_R (lo, hi) and the tool's ok mask per output and seed, plus the
tool's notes (outputs not written by Triton, non-Triton intermediates, TTIR coverage).  Spec elements without a
value (undefined, ties, out of scope) are given as (-inf, +inf), which the tool excludes from its residuals.
"""
from __future__ import annotations

import pickle
import time
import traceback

import numpy as np
import torch

import candidates as K
import common
import run_phase1 as R

FR_CANDIDATES = {"ce": ["inductor_cuda32", "inductor_cuda_bf16"], "pool": ["inductor_cuda32"], "index": ["inductor_cuda32"]}
OUT_MAP = {"ce": {"loss": "loss", "grad": "grad"}, "pool": {"out": "out", "grad": "grad"},
           "index": {"out": "out", "grad_self": ("grad", "self"), "grad_source": ("grad", "source")}}
MAIN = {"ce": "R_A", "avg_pool": "R2", "max_pool": "R_prop"}


def main_reading(family, cond):
    if family == "index":
        return "R_prop" if cond.get("reduce") in ("amax", "amin") else "main"
    return MAIN["ce"] if family == "ce" else MAIN[cond["op"]]


def spec_for(family, cond, seed, dtype, shapes):
    kind = "f32" if (family == "ce" and cond["eps"] != 0 and dtype != "float64") else "f64"
    s = R.load_spec(family, cond["id"], seed, "base", dtype, kind)
    out = {}
    rd = (s or {}).get("readings", {}).get(main_reading(family, cond), {})
    for name, shape in shapes.items():
        unknown = (np.full(shape, -np.inf), np.full(shape, np.inf))
        key = OUT_MAP[family][name]
        val = None
        if s is not None and s.get("status") == "ok":
            if isinstance(key, tuple):
                g = s.get("grad")
                val = None if g is None else g[key[1]]
            elif rd.get("status") == "ok":
                val = rd.get(key)
        if val is None:
            out[name] = unknown
        else:
            lo, hi = np.asarray(val[0], dtype=np.float64).reshape(shape), np.asarray(val[1], dtype=np.float64).reshape(shape)
            if isinstance(key, tuple) or name == "grad":      # NaN in a spec gradient = no strict value (ties)
                none = np.isnan(lo) | np.isnan(hi)
                lo, hi = np.where(none, -np.inf, lo), np.where(none, np.inf, hi)
            out[name] = (lo, hi)
    return out


def _case(family, cond, cand):
    """setup / inputs / launch closures for one condition (base variant), mirroring candidates.run_*."""
    import torch.nn.functional as F
    device, dtype, _ = K.CANDIDATES[cand]
    make = R.FAMILIES[family][1]
    state = {}

    def tensors(seed):
        inp = make(cond, seed)
        if family == "ce":
            x = inp["logits"]
            if cond["layout"] == "transposed":
                base = K._t(x.T, device, dtype).contiguous().requires_grad_(True)
            else:
                base = K._t(x, device, dtype).requires_grad_(True)
            tgt = (K._t(inp["target"], device, dtype) if cond["target"] == "prob"
                   else torch.as_tensor(inp["target"], dtype=torch.long, device=device))
            w = None if inp["weights"] is None else K._t(inp["weights"], device, dtype)
            v = torch.as_tensor(K.ce_upstream(cond, seed), dtype=dtype, device=device)
            return {"base": base, "target": tgt, "weight": w, "v": v}
        if family == "pool":
            x = K._t(inp["x"][None], device, dtype, cond.get("layout", "contiguous")).requires_grad_(True)
            return {"x": x, "inp": inp}
        lay = cond.get("layout", "contiguous")
        s = K._t(inp["self"], device, dtype, lay).requires_grad_(True)
        src = K._t(inp["source"], device, dtype, lay).requires_grad_(True)
        idx = torch.as_tensor(inp["index"], dtype=torch.long, device=device)
        v = torch.as_tensor(K.index_upstream(inp, tuple(s.shape)), dtype=dtype, device=device)
        return {"s": s, "src": src, "idx": idx, "v": v}

    if family == "ce":
        kw = dict(ignore_index=cond["ignore_index"], reduction=cond["reduction"], label_smoothing=float(cond["eps"]))

        def fwd(logits, target, weight):
            return F.cross_entropy(logits, target, weight=weight, **kw)

        def launch(t):
            for k in ("base",):
                t[k].grad = None
            logits = t["base"].t() if cond["layout"] == "transposed" else t["base"]
            loss = state["f"](logits, t["target"], t["weight"])
            (loss * t["v"]).sum().backward()
            g = t["base"].grad.t() if cond["layout"] == "transposed" else t["base"].grad
            return {"loss": loss.reshape(-1), "grad": g}
    elif family == "pool":
        nd = cond["nd"]
        if cond["op"] == "avg_pool":
            pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[nd]
            pkw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                       ceil_mode=cond["ceil_mode"], count_include_pad=cond["count_include_pad"])
            if nd > 1:
                pkw["divisor_override"] = cond["divisor_override"]
        else:
            pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[nd]
            pkw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                       dilation=cond["dilation"], ceil_mode=cond["ceil_mode"])

        def fwd(x):
            return pool(x, **pkw)

        def launch(t):
            t["x"].grad = None
            y = state["f"](t["x"])
            v = K.pool_upstream(cond, t["inp"], tuple(y.shape[1:]))
            (y * torch.as_tensor(v[None], dtype=y.dtype, device=device)).sum().backward()
            return {"out": y[0], "grad": t["x"].grad[0]}
    else:
        op, dim = cond["op"], make(cond, 0)["dim"]
        if op == "index_add":
            def fwd(s, idx, src):
                return torch.index_add(s, dim, idx, src, alpha=cond["alpha"])
        elif op == "index_reduce":
            def fwd(s, idx, src):
                return torch.index_reduce(s, dim, idx, src, cond["reduce"], include_self=cond["include_self"])
        else:
            def fwd(s, idx, src):
                return torch.scatter_reduce(s, dim, idx, src, cond["reduce"], include_self=cond["include_self"])

        def launch(t):
            t["s"].grad, t["src"].grad = None, None
            y = state["f"](t["s"], t["idx"], t["src"])
            (y * t["v"]).sum().backward()
            return {"out": y, "grad_self": t["s"].grad, "grad_source": t["src"].grad}

    def setup():
        torch._dynamo.reset()
        state["f"] = torch.compile(fwd, fullgraph=True, dynamic=False)
        launch(tensors(0))                                   # compile + autotune outside the recorder
        torch.cuda.synchronize()

    return setup, tensors, launch


def run(family, limit=None, only=None):
    conds = [c for c in R.FAMILIES[family][0]() if c["op"] != "flce"][:limit]
    for cand in [c for c in FR_CANDIDATES[family] if only is None or c == only]:
        dtype = K.DTYPE_NAME[K.CANDIDATES[cand][1]]
        path = R.CACHE / "fr" / family / f"{cand}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = pickle.loads(path.read_bytes()) if path.exists() else {}
        t0 = time.time()
        for i, cond in enumerate(conds):
            if cond["id"] in done:
                continue
            raw = None
            try:
                setup, tensors, launch = _case(family, cond, cand)

                def spec(t, _cond=cond):
                    outs = launch_shapes[0]
                    seed = t["_seed"]
                    return spec_for(family, _cond, seed, dtype, outs)

                # the output shapes (from one eager-free dry run is not possible before compile: use the spec cache)
                launch_shapes = [None]

                def make_inputs(seed, _t=tensors):
                    d = _t(seed)
                    d["_seed"] = seed
                    return d

                def launch_rec(t, _l=launch):
                    outs = _l(t)
                    launch_shapes[0] = {k: tuple(v.shape) for k, v in outs.items()}
                    return outs

                rep, keep = common.fr_run(f"{family}/{cond['id']}/{cand}", setup, make_inputs, launch_rec, spec)
                raw = {"status": "ok",
                       "keep": {k: [{kk: p[kk] for kk in ("r_lo", "r_hi", "k", "ok", "shape")} for p in v] for k, v in keep.items()},
                       "notes": {k: rep.get(k) for k in ("outputs_not_written_by_triton", "outputs_modified_after_last_triton_write",
                                                         "outputs_whose_writing_programs_aborted", "ttir_coverage_complete",
                                                         "external_reentries_seed0", "launches")},
                       "mixed": {k: v.get("depends_on_non_triton_intermediates") for k, v in rep.get("outputs", {}).items()},
                       "special": {k: v.get("special_values") for k, v in rep.get("outputs", {}).items()},
                       "seconds": rep.get("seconds")}
            except Exception as exc:  # noqa: BLE001
                raw = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-1200:]}
            done[cond["id"]] = raw
            if (i + 1) % 10 == 0 or i == len(conds) - 1:
                path.write_bytes(pickle.dumps(done))
                print(f"fr {family}/{cand}: {i + 1}/{len(conds)} {time.time() - t0:.0f} s", flush=True)
        path.write_bytes(pickle.dumps(done))
