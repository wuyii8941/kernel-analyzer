#!/usr/bin/env python3
"""2b G6 (composition reuse) and G7 (numerical-effect stream) through the frozen tool (detector 2.2), mode A.

G6 -- unfamiliar compositions: 10 programs composed from the 2b families (pre-norm transformer block, RoPE attention, GELU MLP,
GroupNorm-SiLU block, cross-entropy head, MoE block, reduction chain, gather/scatter round trip, conv-free BatchNorm-pool chain,
loss + compiled AdamW step), compiled with Inductor, float32, forward and backward.  Nothing is written per program except the
generic case wrapper (inputs, one call, the outputs to check); the question is how often the tool establishes the automatic
reference K_R without further work, and at what cost.  Per program: Triton launches, outputs written by Triton, TTIR coverage,
complete K_R fraction, binding / modification notes, time by phase, and E_R -- the float64 eager run of the same composition on
the same inputs inside the K_R enclosure (widened by τ32) -- plus E (compiled vs eager float32).

G7 -- numerical-effect stream (interim report E6: the numerical effects are a separate branch): 20 low-precision programs
(bfloat16 / float16 Inductor compilations of 2b operators) through mode A with the tool's default development / confirmation
units (32 + 64 seeds); per output the tool's direction rules and default detector decide whether e_num = K - K_R has a
confirmed systematic (average) effect.  These are numerical effects of the implementation, never essential errors; they are
reported apart.

    python scripts/essential/p2b_g6_g7.py g6
    python scripts/essential/p2b_g6_g7.py g7
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/essential/phase2b"
TAU32 = 2.0 ** -12


def gen(seed, *shape, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float64) * scale).float().double()


# ------------------------------------------------------------------------------------------------ G6 programs
# each: (params(seed0) -> dict of float64 CPU tensors, inputs(seed) -> dict, fn(p, x) -> dict of outputs incl. grads keys)

def rms(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def attn(q, k, v, causal=True):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        L = q.shape[-2]
        s = s.masked_fill(torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v


def rope(x, pos):
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d))
    ang = pos[:, None].to(x.dtype) * inv[None]
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


G6 = {}


def g6(name):
    def deco(f):
        G6[name] = f
        return f
    return deco


@g6("prenorm_block")
def _p1():
    P = {"w1": gen(1, 32), "wq": gen(2, 32, 32, scale=0.2), "wk": gen(3, 32, 32, scale=0.2), "wv": gen(4, 32, 32, scale=0.2),
         "w2": gen(5, 32), "wg": gen(6, 32, 64, scale=0.2), "wu": gen(7, 32, 64, scale=0.2), "wd": gen(8, 64, 32, scale=0.2)}

    def fn(p, x):
        x = x["x"]
        h = rms(x, p["w1"])
        B, L, D = h.shape
        q, k, v = ((h @ p[w]).view(B, L, 4, 8).transpose(1, 2) for w in ("wq", "wk", "wv"))
        y = x + attn(q, k, v).transpose(1, 2).reshape(B, L, D)
        z = rms(y, p["w2"])
        return y + (F.silu(z @ p["wg"]) * (z @ p["wu"])) @ p["wd"]
    return P, (lambda s: {"x": gen(100 + s, 2, 9, 32)}), fn


@g6("rope_attention")
def _p2():
    P = {"emb": gen(11, 50, 32, scale=0.5), "wo": gen(12, 32, 32, scale=0.2)}

    def fn(p, x):
        h = p["emb"][x["ids"]]
        B, L, D = h.shape
        q = rope(h.view(B, L, 4, 8).transpose(1, 2), torch.arange(L, device=h.device) + 3)
        k = rope(h.view(B, L, 4, 8).transpose(1, 2), torch.arange(L, device=h.device) + 3)
        return attn(q, k, h.view(B, L, 4, 8).transpose(1, 2)).transpose(1, 2).reshape(B, L, D) @ p["wo"]
    return P, (lambda s: {"ids": torch.randint(0, 50, (2, 11), generator=torch.Generator().manual_seed(200 + s))}), fn


@g6("gelu_mlp_layernorm")
def _p3():
    P = {"ln_w": gen(21, 32), "ln_b": gen(22, 32), "w1": gen(23, 32, 96, scale=0.2), "w2": gen(24, 96, 32, scale=0.2)}

    def fn(p, x):
        h = F.layer_norm(x["x"], (32,), p["ln_w"], p["ln_b"])
        return x["x"] + F.gelu(h @ p["w1"], approximate="tanh") @ p["w2"]
    return P, (lambda s: {"x": gen(300 + s, 3, 7, 32)}), fn


@g6("groupnorm_silu")
def _p4():
    P = {"gw": gen(31, 12), "gb": gen(32, 12), "lin": gen(33, 12, 12, scale=0.3)}

    def fn(p, x):
        h = F.silu(F.group_norm(x["x"], 4, p["gw"], p["gb"]))
        return torch.einsum("bcl,cd->bdl", h, p["lin"]) + x["x"]
    return P, (lambda s: {"x": gen(400 + s, 2, 12, 15)}), fn


@g6("ce_head")
def _p5():
    P = {"w": gen(41, 32, 101, scale=0.2)}

    def fn(p, x):
        logits = x["h"] @ p["w"]
        return F.cross_entropy(logits, x["y"], label_smoothing=0.1, ignore_index=-100).reshape(1)
    return P, (lambda s: {"h": gen(500 + s, 13, 32),
                          "y": torch.where(torch.arange(13) % 5 == 0, -100,
                                           torch.randint(0, 101, (13,), generator=torch.Generator().manual_seed(500 + s)))}), fn


@g6("moe_block")
def _p6():
    P = {"router": gen(51, 16, 4, scale=0.5), "W": gen(52, 4, 16, 16, scale=0.3)}

    def fn(p, x):
        h = x["x"]
        probs = torch.softmax(h @ p["router"], -1)
        w, e = torch.topk(probs, 2, -1)
        w = w / w.sum(-1, keepdim=True)
        T = h.shape[0]
        flat_e, flat_t = e.reshape(-1), torch.arange(T, device=h.device).repeat_interleave(2)
        ys = torch.bmm(h[flat_t][:, None], p["W"][flat_e]).squeeze(1) * w.reshape(-1)[:, None]
        return torch.zeros_like(h).index_add(0, flat_t, ys)
    return P, (lambda s: {"x": gen(600 + s, 13, 16)}), fn


@g6("reduction_chain")
def _p7():
    P = {"a": gen(61, 7)}

    def fn(p, x):
        s = torch.softmax(x["x"] * p["a"][:, None], -1)
        return torch.cat([torch.cumsum(s, -1), torch.logsumexp(x["x"], -1, keepdim=True), x["x"].var(-1, keepdim=True)], -1)
    return P, (lambda s: {"x": gen(700 + s, 7, 33)}), fn


@g6("gather_scatter_roundtrip")
def _p8():
    P = {"scale": gen(71, 9)}

    def fn(p, x):
        g = torch.gather(x["x"] * p["scale"], 1, x["idx"])
        return torch.zeros_like(x["x"]).scatter_add(1, x["idx"], torch.tanh(g))
    return P, (lambda s: {"x": gen(800 + s, 6, 9), "idx": torch.randint(0, 9, (6, 5), generator=torch.Generator().manual_seed(800 + s))}), fn


@g6("batchnorm_pool_chain")
def _p9():
    P = {"bw": gen(81, 6), "bb": gen(82, 6)}

    def fn(p, x):
        h = F.relu(F.batch_norm(x["x"], None, None, p["bw"], p["bb"], training=True))
        return F.avg_pool1d(F.max_pool1d(h, 3, 2, 1), 2, 2, ceil_mode=False)
    return P, (lambda s: {"x": gen(900 + s, 4, 6, 17)}), fn


@g6("loss_then_adamw_step")
def _p10():
    P = {"w": gen(91, 16, 8, scale=0.3)}

    def fn(p, x):
        loss = (torch.tanh(x["x"] @ p["w"]) - x["t"]).pow(2).mean()
        g = torch.autograd.grad(loss, p["w"], create_graph=False)[0]
        m = 0.1 * g
        v = 0.001 * g * g
        step = 1e-2 * (m / 0.1) / (torch.sqrt(v / 0.001) + 1e-8)
        return torch.cat([(p["w"] * (1 - 1e-2 * 0.1) - step).reshape(-1), loss.reshape(1)])
    return P, (lambda s: {"x": gen(1000 + s, 10, 16), "t": gen(1001 + s, 10, 8)}), fn


def run_g6():
    torch.backends.cuda.matmul.allow_tf32 = False
    res, t0 = {}, time.time()
    for name, make in G6.items():
        P64, inputs, fn = make()
        try:
            st = {}

            def to_dev(d, dt=torch.float32, dev="cuda"):
                return {k: (v.to(dev, dt).requires_grad_(True) if v.is_floating_point() else v.to(dev)) for k, v in d.items()}

            def call(p, x):
                out = fn(p, x)
                outs = {"out": out}
                grads_of = [k for k, v in list(p.items()) + list(x.items()) if isinstance(v, torch.Tensor) and v.requires_grad]
                if out.requires_grad and name != "loss_then_adamw_step":
                    gs = torch.autograd.grad(out, [({**p, **x})[k] for k in grads_of], torch.ones_like(out) * 0.5, allow_unused=True)
                    outs.update({f"d_{k}": g for k, g in zip(grads_of, gs) if g is not None})
                return outs

            def setup():
                torch._dynamo.reset()
                st["p"] = to_dev(P64)
                st["fn"] = torch.compile(call, dynamic=False)
                launch(inputs_(0))
                torch.cuda.synchronize()

            def inputs_(seed):
                d = to_dev(inputs(seed))
                d["_seed"] = seed
                return d

            def launch(t):
                x = {k: v for k, v in t.items() if k != "_seed"}
                return st["fn"](st["p"], x)

            rep, keep = common.fr_run(f"g6/{name}", setup, inputs_, launch, lambda _t: None)
            # E_R and E: float64 eager (CPU) and float32 eager (CUDA) of the same composition on the same inputs
            e_r, e = {}, {}
            for sd in (0, 1, 2):
                x64 = {k: (v.double().requires_grad_(True) if v.is_floating_point() else v) for k, v in inputs(sd).items()}
                p64 = {k: v.double().requires_grad_(True) for k, v in P64.items()}
                ref64 = {k: v.detach().numpy() for k, v in call(p64, x64).items()}
                ref32 = {k: v.detach().double().cpu().numpy() for k, v in call(to_dev(P64), to_dev(inputs(sd))).items()}
                for oname, rows in keep.items():
                    for r in rows:
                        if r["seed"] != sd or oname not in ref64:
                            continue
                        ok = np.asarray(r["ok"], bool)
                        lo, hi = np.asarray(r["r_lo"], float), np.asarray(r["r_hi"], float)
                        rv = np.asarray(ref64[oname], float).reshape(-1)
                        if rv.size != lo.size:
                            continue
                        d = np.nan_to_num(np.maximum(np.maximum(lo - rv, rv - hi), 0), nan=np.inf)
                        bad = ok & (d > TAU32 * (1 + np.abs(rv)))
                        a = e_r.setdefault(oname, [0, 0])
                        a[0] += int(ok.sum())
                        a[1] += int(bad.sum())
                kc = {}
                outs32 = {k: v.detach().double().cpu().numpy() for k, v in st["fn"](st["p"], {k: v for k, v in to_dev(inputs(sd)).items()}).items()}
                for oname in outs32:
                    k_, r_ = outs32[oname], ref32[oname]
                    viol = int(np.sum(np.abs(k_ - r_) > TAU32 * (1 + np.abs(r_))))
                    a = e.setdefault(oname, [0, 0])
                    a[0] += k_.size
                    a[1] += viol
            outs = rep.get("outputs", {})
            res[name] = {"status": "ok", "launches": len(rep.get("launches") or []),
                         "outputs": sorted(keep.keys()) + sorted(rep.get("outputs_not_written_by_triton") or []),
                         "triton_outputs": sorted(keep.keys()),
                         "not_written_by_triton": rep.get("outputs_not_written_by_triton"),
                         "binding_not_established": rep.get("outputs_binding_not_established"),
                         "modified_after_last_triton_write": rep.get("outputs_modified_after_last_triton_write"),
                         "ttir_coverage_complete": rep.get("ttir_coverage_complete"),
                         "complete_fraction": {k: v["reference_classes"]["finite_complete_fraction"] for k, v in outs.items()},
                         "E_R_kr_contains_float64": {k: f"{v[1]} outside / {v[0]}" for k, v in e_r.items()},
                         "E_compiled_vs_eager": {k: f"{v[1]} beyond / {v[0]}" for k, v in e.items()},
                         "timing_seconds": rep.get("timing_seconds"), "wrapper_lines": 0}
        except Exception as exc:  # noqa: BLE001
            res[name] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-1500:]}
        r = res[name]
        print(name, r["status"], {k: r.get(k) for k in ("launches", "not_written_by_triton", "complete_fraction", "E_R_kr_contains_float64",
                                                       "E_compiled_vs_eager")} if r["status"] == "ok" else r["reason"][:300], flush=True)
    (OUT / "g6_compositions").mkdir(parents=True, exist_ok=True)
    (OUT / "g6_compositions" / "results.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "programs": res},
                                                                     indent=1, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ G7 programs

def g7_programs():
    bf, hf = torch.bfloat16, torch.float16
    P = []
    P.append(("layer_norm_bf16", bf, lambda s: [gen(s, 8, 64)], lambda x: F.layer_norm(x, (64,))))
    P.append(("rms_norm_bf16", bf, lambda s: [gen(s, 8, 64)], lambda x: F.rms_norm(x, (64,), eps=1e-5)))
    P.append(("group_norm_bf16", bf, lambda s: [gen(s, 4, 8, 16)], lambda x: F.group_norm(x, 4)))
    P.append(("batch_norm_train_bf16", bf, lambda s: [gen(s, 8, 6, 12)], lambda x: F.batch_norm(x, None, None, training=True)))
    P.append(("softmax_bf16", bf, lambda s: [gen(s, 8, 257)], lambda x: torch.softmax(x, -1)))
    P.append(("log_softmax_fp16", hf, lambda s: [gen(s, 8, 257)], lambda x: torch.log_softmax(x, -1)))
    P.append(("logsumexp_bf16", bf, lambda s: [gen(s, 8, 1027)], lambda x: torch.logsumexp(x, -1)))
    P.append(("sum_long_bf16", bf, lambda s: [gen(s, 4, 4099)], lambda x: x.sum(-1)))
    P.append(("mean_long_fp16", hf, lambda s: [gen(s, 4, 4099)], lambda x: x.mean(-1)))
    P.append(("var_bf16", bf, lambda s: [gen(s, 4, 1027)], lambda x: x.var(-1)))
    P.append(("cumsum_bf16", bf, lambda s: [gen(s, 2, 1027)], lambda x: torch.cumsum(x, -1)))
    P.append(("gelu_tanh_bf16", bf, lambda s: [gen(s, 8, 257)], lambda x: F.gelu(x, approximate="tanh")))
    P.append(("silu_mul_bf16", bf, lambda s: [gen(s, 8, 257), gen(s + 1, 8, 257)], lambda a, b: F.silu(a) * b))
    P.append(("geglu_fp16", hf, lambda s: [gen(s, 8, 257), gen(s + 1, 8, 257)], lambda a, b: F.gelu(a) * b))
    P.append(("ce_label_smoothing_bf16", bf, lambda s: [gen(s, 16, 101)],
              lambda x: F.cross_entropy(x, torch.arange(16, device=x.device) % 101, label_smoothing=0.1).reshape(1)))
    P.append(("attention_softmax_bf16", bf, lambda s: [gen(s, 2, 4, 33, 33)],
              lambda sc: torch.softmax(sc / 4.0 + torch.ones(33, 33, device=sc.device, dtype=sc.dtype).triu(1) * -1e4, -1)))
    P.append(("scatter_add_bf16", bf, lambda s: [gen(s, 64, 8)],
              lambda x: torch.zeros(7, 8, device=x.device, dtype=x.dtype).index_add(0, torch.arange(64, device=x.device) % 7, x)))
    P.append(("embedding_bag_mean_bf16", bf, lambda s: [gen(s, 50, 16)],
              lambda w: F.embedding_bag(torch.arange(40, device=w.device) % 50, w, torch.tensor([0, 7, 20, 33], device=w.device), mode="mean")))
    P.append(("adamw_update_bf16", bf, lambda s: [gen(s, 64, 16), gen(s + 1, 64, 16), gen(s + 2, 64, 16).abs() * 1e-3],
              lambda p, m, v: p * (1 - 1e-3) - 1e-3 * (0.9 * m) / (torch.sqrt(0.999 * v) + 1e-8)))
    P.append(("residual_rms_fp16", hf, lambda s: [gen(s, 8, 64), gen(s + 1, 8, 64)],
              lambda x, r: F.rms_norm(x + r, (64,), eps=1e-5) * 1.5))
    return P


def run_g7():
    res, t0 = {}, time.time()
    for name, dt, make, fn in g7_programs():
        try:
            st = {}

            def setup():
                torch._dynamo.reset()
                st["fn"] = torch.compile(fn, dynamic=False)
                launch(inputs_(0))
                torch.cuda.synchronize()

            def inputs_(seed):
                return {"args": [a.to("cuda", dt) for a in make(seed)], "_seed": seed}

            def launch(t):
                return {"out": st["fn"](*t["args"])}

            rep = common.fr_run(f"g7/{name}", setup, inputs_, launch, lambda _t: None, seeds=tuple(range(0, 96)))[0]
            o = rep.get("outputs", {}).get("out", {})
            num = o.get("numerical", {})
            rules = [{"rule": r.get("rule"), "verdict": r.get("verdict"), "interpretation": r.get("interpretation")} for r in num.get("rules", [])]
            det = num.get("default_detector", {})
            res[name] = {"status": "ok", "dtype": str(dt), "launches": len(rep.get("launches") or []),
                         "complete_fraction": o.get("reference_classes", {}).get("finite_complete_fraction"),
                         "not_written_by_triton": rep.get("outputs_not_written_by_triton"),
                         "rules": rules, "default_detector": {k: det.get(k) for k in ("verdict", "decision", "p_value", "statistic") if k in det},
                         "scale": num.get("scale"), "seconds": rep.get("seconds")}
        except Exception as exc:  # noqa: BLE001
            res[name] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-1200:]}
        r = res[name]
        if r["status"] == "ok":
            print(name, "complete", r["complete_fraction"], "rules", [(x["rule"], x["verdict"]) for x in r["rules"]],
                  "detector", r["default_detector"], "rel_rms", (r["scale"] or {}).get("relative_rms"), flush=True)
        else:
            print(name, r["status"], r["reason"][:300], flush=True)
    (OUT / "g7_numerical_stream").mkdir(parents=True, exist_ok=True)
    (OUT / "g7_numerical_stream" / "results.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "programs": res},
                                                                         indent=1, default=str) + "\n")


if __name__ == "__main__":
    {"g6": run_g6, "g7": run_g7}[sys.argv[1]]()
