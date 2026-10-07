#!/usr/bin/env python3
"""2b call / state scenario "A -> B -> A" for compiled candidates (protocol v2 section 3, search unit "call / state
scenario"): one ``torch.compile``d callable is called with input A, then with B (a different shape, dtype, layout, flag or
mode that forces a guard failure / recompile or a dynamic-shape path), then with A again.  Property: every call equals a
fresh compile of the same call and equals eager (τ32 for float32, bitwise where the op is exact); the third call equals the
first bitwise (the cached graph for A is reused, not B's).

Programs (from the 2b families): layer_norm, rms_norm, softmax, logsumexp, cumsum, gather, index_add, embedding_bag(mean),
gelu(tanh), swiglu, batch_norm (train -> eval -> train), max_pool2d(return_indices), scaled_dot_product_attention (math),
cross_entropy(label_smoothing), adamw step (compiled optimizer step on two parameter sets).  B variants: shape (another
length), dtype (bfloat16), layout (transposed storage), flag (a Python scalar argument: eps / dim / reduction), requires_grad
toggled, and module mode where it applies.  Each sequence is run with dynamic=False and dynamic=None (default automatic
dynamic shapes).

    python scripts/essential/p2b_compile_sequences.py run
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/essential/phase2b/compile_sequences"
TAU32 = 2.0 ** -12


def T(shape, seed, dtype=torch.float32, layout="contiguous", device="cuda"):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
    if layout == "transposed":
        return t.transpose(-1, -2).contiguous().to(device).transpose(-1, -2)
    return t.to(device)


def IDX(n, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (n,), generator=g).to("cuda")


# each program: fn(args...) and a list of (label, args-builder) for A and B variants
def programs():
    P = {}
    P["layer_norm"] = (lambda x, eps: F.layer_norm(x, (x.shape[-1],), eps=eps),
                       {"A": lambda: (T((4, 33), 0), 1e-5), "B_shape": lambda: (T((4, 65), 1), 1e-5),
                        "B_dtype": lambda: (T((4, 33), 2, torch.bfloat16), 1e-5), "B_layout": lambda: (T((33, 4), 3, layout="transposed").T.contiguous().T, 1e-5),
                        "B_flag": lambda: (T((4, 33), 4), 1e-2)})
    P["rms_norm"] = (lambda x, eps: F.rms_norm(x, (x.shape[-1],), eps=eps),
                     {"A": lambda: (T((4, 33), 0), 1e-5), "B_shape": lambda: (T((7, 33), 1), 1e-5), "B_flag": lambda: (T((4, 33), 2), 1e-1),
                      "B_dtype": lambda: (T((4, 33), 3, torch.bfloat16), 1e-5)})
    P["softmax"] = (lambda x, d: torch.softmax(x, d),
                    {"A": lambda: (T((5, 17), 0), -1), "B_flag": lambda: (T((5, 17), 1), 0), "B_shape": lambda: (T((5, 129), 2), -1),
                     "B_layout": lambda: (T((5, 17), 3, layout="transposed"), -1)})
    P["logsumexp"] = (lambda x, d: torch.logsumexp(x, d),
                      {"A": lambda: (T((5, 17), 0), -1), "B_flag": lambda: (T((5, 17), 1), 0), "B_shape": lambda: (T((3, 1031), 2), -1)})
    P["cumsum"] = (lambda x, d: torch.cumsum(x, d),
                   {"A": lambda: (T((5, 17), 0), -1), "B_flag": lambda: (T((5, 17), 1), 0), "B_shape": lambda: (T((5, 4099), 2), -1)})
    P["gather"] = (lambda x, i: torch.gather(x, 1, i),
                   {"A": lambda: (T((6, 9), 0), IDX(30, 9, 0).reshape(6, 5)), "B_shape": lambda: (T((6, 40), 1), IDX(60, 40, 1).reshape(6, 10)),
                    "B_layout": lambda: (T((6, 9), 2, layout="transposed"), IDX(30, 9, 2).reshape(6, 5))})
    P["index_add"] = (lambda s, i, src, a: torch.index_add(s, 0, i, src, alpha=a),
                      {"A": lambda: (T((7, 3), 0), IDX(11, 7, 0), T((11, 3), 1), 2.5), "B_flag": lambda: (T((7, 3), 2), IDX(11, 7, 3), T((11, 3), 4), -1.0),
                       "B_shape": lambda: (T((1, 3), 5), IDX(11, 1, 6), T((11, 3), 7), 2.5)})
    P["embedding_bag_mean"] = (lambda w, i, o: F.embedding_bag(i, w, o, mode="mean"),
                               {"A": lambda: (T((20, 8), 0), IDX(15, 20, 0), torch.tensor([0, 4, 9, 12], device="cuda")),
                                "B_shape": lambda: (T((20, 8), 1), IDX(15, 20, 1), torch.tensor([0, 4, 4, 9], device="cuda"))})
    P["gelu_tanh"] = (lambda x: F.gelu(x, approximate="tanh"),
                      {"A": lambda: (T((6, 33), 0),), "B_shape": lambda: (T((6, 34), 1),), "B_dtype": lambda: (T((6, 33), 2, torch.bfloat16),)})
    P["swiglu"] = (lambda a, b: F.silu(a) * b,
                   {"A": lambda: (T((6, 33), 0), T((6, 33), 1)), "B_bcast": lambda: (T((6, 33), 2), T((1, 33), 3)),
                    "B_layout": lambda: (T((6, 33), 4, layout="transposed"), T((6, 33), 5))})
    P["max_pool2d_idx"] = (lambda x, k: F.max_pool2d(x, k, return_indices=True),
                           {"A": lambda: (T((1, 2, 8, 9), 0), 2), "B_flag": lambda: (T((1, 2, 8, 9), 1), 3), "B_shape": lambda: (T((1, 2, 5, 5), 2), 2)})
    P["sdpa_math"] = (lambda q, k, v, c: F.scaled_dot_product_attention(q, k, v, is_causal=c),
                      {"A": lambda: (T((2, 4, 9, 16), 0), T((2, 4, 9, 16), 1), T((2, 4, 9, 16), 2), False),
                       "B_flag": lambda: (T((2, 4, 9, 16), 3), T((2, 4, 9, 16), 4), T((2, 4, 9, 16), 5), True),
                       "B_shape": lambda: (T((2, 4, 9, 16), 6), T((2, 4, 13, 16), 7), T((2, 4, 13, 16), 8), False)})
    P["cross_entropy"] = (lambda x, y, eps, red: F.cross_entropy(x, y, label_smoothing=eps, reduction=red),
                          {"A": lambda: (T((6, 11), 0), IDX(6, 11, 0), 0.1, "mean"), "B_flag": lambda: (T((6, 11), 1), IDX(6, 11, 1), 0.0, "mean"),
                           "B_flag2": lambda: (T((6, 11), 2), IDX(6, 11, 2), 0.1, "sum"), "B_shape": lambda: (T((9, 11), 3), IDX(9, 11, 3), 0.1, "mean")})
    return P


def to_np(x):
    if isinstance(x, (tuple, list)):
        return [to_np(v) for v in x]
    return x.detach().double().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()


def same(a, b, tol):
    if isinstance(a, list):
        return all(same(x, y, tol) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if tol == 0 or not np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.all((np.abs(a - b) <= tol * (1 + np.abs(b))) | (np.isnan(a) & np.isnan(b))))


def run_sequence(name, fn, builders, label_b, dynamic):
    torch._dynamo.reset()
    cfn = torch.compile(fn, dynamic=dynamic)
    argsA, argsB = builders["A"](), builders[label_b]()
    outA1 = to_np(cfn(*argsA))
    outB = to_np(cfn(*argsB))
    outA2 = to_np(cfn(*argsA))
    tol = TAU32 if all(not (isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16) for a in argsB) else 2.0 ** -8
    eagerA, eagerB = to_np(fn(*argsA)), to_np(fn(*argsB))
    torch._dynamo.reset()
    freshB = to_np(torch.compile(fn, dynamic=dynamic)(*argsB))
    return {"A1_vs_eager": same(outA1, eagerA, TAU32), "B_vs_eager": same(outB, eagerB, tol), "A2_vs_eager": same(outA2, eagerA, TAU32),
            "info_A1_eq_A2_bitwise": same(outA1, outA2, 0), "info_B_eq_fresh_compile_bitwise": same(outB, freshB, 0)}


def run_bn():
    """module mode: BatchNorm1d compiled, train -> eval -> train (running stats evolve; eval must use them)."""
    res = {}
    for dynamic in (False, None):
        torch._dynamo.reset()
        m = torch.nn.BatchNorm1d(6).cuda()
        m_e = torch.nn.BatchNorm1d(6).cuda()
        cm = torch.compile(m, dynamic=dynamic)
        xs = [T((4, 6, 10), s) for s in range(3)]
        outs, outs_e = [], []
        for mode, x in zip(("train", "eval", "train"), xs):
            m.train(mode == "train")
            m_e.train(mode == "train")
            outs.append(to_np(cm(x)))
            outs_e.append(to_np(m_e(x)))
        res[f"dynamic={dynamic}"] = {"all_calls_vs_eager": all(same(a, b, TAU32) for a, b in zip(outs, outs_e)),
                                     "running_stats_vs_eager": same(to_np(m.running_mean), to_np(m_e.running_mean), TAU32)
                                     and same(to_np(m.running_var), to_np(m_e.running_var), TAU32)}
    return res


def run_opt():
    """compiled optimizer step on parameter set A, then set B (other shapes), then A again; vs eager AdamW on copies."""
    res = {}
    for dynamic in (False, None):
        torch._dynamo.reset()
        pa = [torch.nn.Parameter(T((5, 3), 0)), torch.nn.Parameter(T((7,), 1))]
        pb = [torch.nn.Parameter(T((4, 4), 2))]
        ea = [torch.nn.Parameter(p.detach().clone()) for p in pa]
        eb = [torch.nn.Parameter(p.detach().clone()) for p in pb]
        oa, ob = torch.optim.AdamW(pa, lr=1e-2), torch.optim.AdamW(pb, lr=1e-2)
        oea, oeb = torch.optim.AdamW(ea, lr=1e-2, foreach=False), torch.optim.AdamW(eb, lr=1e-2, foreach=False)
        step = torch.compile(lambda o: o.step(), dynamic=dynamic)
        ok = True
        for k, (o, oe, ps, es) in enumerate(((oa, oea, pa, ea), (ob, oeb, pb, eb), (oa, oea, pa, ea))):
            for p, e in zip(ps, es):
                g = T(tuple(p.shape), 10 + k)
                p.grad, e.grad = g.clone(), g.clone()
            step(o)
            oe.step()
            ok &= all(same(to_np(p), to_np(e), TAU32) for p, e in zip(ps, es))
        res[f"dynamic={dynamic}"] = {"all_steps_vs_eager": bool(ok)}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    out, t0 = {"sequences": {}, "module_mode": None, "optimizer": None}, time.time()
    for name, (fn, builders) in programs().items():
        for label in [k for k in builders if k != "A"]:
            for dynamic in (False, None):
                key = f"{name}/{label}/dynamic={dynamic}"
                try:
                    out["sequences"][key] = run_sequence(name, fn, builders, label, dynamic)
                except Exception as exc:  # noqa: BLE001
                    out["sequences"][key] = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
                                             "trace": traceback.format_exc()[-800:]}
                r = out["sequences"][key]
                req = {k: v for k, v in r.items() if not k.startswith("info_")}
                flag = "error" if "error" in r else ("ok" if all(req.values()) else "VIOLATION " + str({k: v for k, v in req.items() if not v}))
                if "error" not in r and not all(r.values()):
                    flag += " (bitwise differences: " + ", ".join(k for k, v in r.items() if k.startswith("info_") and not v) + ")"
                print(key, flag, flush=True)
    out["module_mode"] = run_bn()
    out["optimizer"] = run_opt()
    print("batchnorm train/eval/train", out["module_mode"])
    print("optimizer A/B/A", out["optimizer"])
    seq = out["sequences"].values()
    out["summary"] = {"sequences": len(out["sequences"]), "errors": sum("error" in r for r in seq),
                      "violations": sum(1 for r in seq if "error" not in r and not all(v for k, v in r.items() if not k.startswith("info_"))),
                      "bitwise_differences_only": sum(1 for r in seq if "error" not in r and all(v for k, v in r.items() if not k.startswith("info_"))
                                                      and not all(r.values())),
                      "seconds": round(time.time() - t0, 1)}
    print(out["summary"])
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
