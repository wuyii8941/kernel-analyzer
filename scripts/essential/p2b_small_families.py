#!/usr/bin/env python3
"""2b families "checkpoint", "rope", "moe" (protocol v2 section 3; registry tier1): E and P.  F stays closed until the specs
are delivered.

checkpoint (12 planned combinations of impl x graph x compile): non-reentrant / reentrant torch.utils.checkpoint around a
block, and transformers' ``gradient_checkpointing_enable`` on a 2-layer Llama (liger env); graphs: mlp (Linear-GELU-Linear),
shared_param (the same Linear applied twice inside and outside the checkpointed block), reuse_twice (an intermediate used
twice); compile on / off.  Candidates: CPU float64, CUDA float32, compiled CUDA float32, HF.  P (pre-registered): checkpoint on
== off, bitwise for float64 on CPU, τ32 otherwise (outputs and every parameter gradient).

rope (7 planned combinations of convention x offset x scaling x head_dim): an in-repo reference for each convention (rotate-half
= GPT-NeoX / HF Llama; interleaved = GPT-J / original), HF ``apply_rotary_pos_emb`` (rotate-half), Liger ``liger_rotary_pos_emb``
(rotate-half) and Unsloth ``fast_rope_embedding`` (rotate-half); conventions are kept apart: a candidate is compared only on
its own convention, the other convention is recorded as "not applicable" (not as a deviation).  P (pre-registered): norm per
rotated pair preserved; <R(m) q, R(n) k> depends on m - n only (positions shifted by 5); R(0) = identity.

moe (10 planned combinations of topk x experts x capacity x routing): the registry's in-repo references -- a per-token loop on
CPU float64, a sort-based permute + index_add implementation on CUDA float32 and its compiled form.  P (pre-registered):
permute -> unpermute is the identity; combine weights per token sum to 1; dropped tokens contribute zero; the auxiliary
load-balancing loss equals the Switch formula E * sum_e f_e P_e on the counts.

    python scripts/essential/p2b_small_families.py run --family checkpoint --env ka_main   (and --env liger)
    python scripts/essential/p2b_small_families.py analyse --family checkpoint
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
CACHE = ROOT / ".cache/essential/p2b/small"
OUT = ROOT / "results/essential/phase2b"
SEEDS = (0, 1, 2)
TAU32 = 2.0 ** -12


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def plan(family, keys):
    p = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"][family]["conditions"]
    out, seen = [], {}
    for c in p:
        key = tuple(str(c[k]) for k in keys)
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = dict(c, high_risk=bool(c.get("high_risk")), id=family[:4] + "_" + "_".join(str(c[k]) for k in keys))
        seen[key] = d
        out.append(d)
    return out


def npy(t):
    return t.detach().double().cpu().numpy().copy()


# ------------------------------------------------------------------------------------------------ checkpoint

CK_KEYS = ("impl", "graph", "compile")


class Block(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.l1, self.l2 = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)

    def inner(self, h):
        if self.graph == "mlp":
            return self.l2(F.gelu(self.l1(h)))
        if self.graph == "shared_param":
            return self.l2(F.gelu(self.l1(h)))
        z = F.gelu(self.l1(h))
        return self.l2(z * torch.tanh(z))                       # reuse_twice: z used twice

    def forward(self, x, ckpt, reentrant):
        h = self.l2(F.gelu(self.l1(x))) if self.graph == "shared_param" else x     # the same parameters outside the block
        if ckpt:
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(self.inner, h, use_reentrant=reentrant)
        else:
            y = self.inner(h)
        return y


def ck_run(cand, c, seed, variant):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    rng = _rng(f"ck/{c['id']}/{seed}")
    ckpt = variant == "on"
    if c["impl"] == "hf_gradient_checkpointing":
        if not cand.get("hf"):
            raise NotImplementedError("HF gradient checkpointing runs in the liger environment")
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(1000 + seed)
        m = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")).to(dev, dt)
        m.train()
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids = torch.as_tensor(rng.integers(0, 64, (2, 9)), device=dev)
        fn = (lambda: m(input_ids=ids, labels=ids).loss)
        if c["compile"]:
            fn = torch.compile(fn)
        loss = fn()
        loss.backward()
        return {"loss": np.array([float(loss)]), **{f"g_{n}": npy(p.grad) for n, p in m.named_parameters() if p.grad is not None}}
    if cand.get("hf"):
        raise NotImplementedError("torch checkpoint candidates run in ka_main")
    torch.manual_seed(2000 + seed)
    m = Block(c["graph"]).to(dev, dt)
    x = torch.tensor(rng.normal(0, 1, (4, 8)), dtype=dt, device=dev, requires_grad=True)
    g = torch.tensor(rng.normal(0, 1, (4, 8)), dtype=dt, device=dev)
    reentrant = c["impl"] == "reentrant"
    fwd = (lambda xx: m(xx, ckpt, reentrant))
    if c["compile"] or cand.get("compiled"):
        torch._dynamo.reset()
        fwd = torch.compile(fwd)
    y = fwd(x)
    y.backward(g)
    return {"y": npy(y), "dx": npy(x.grad), **{f"g_{n}": npy(p.grad) for n, p in m.named_parameters()}}


# ------------------------------------------------------------------------------------------------ rope

RO_KEYS = ("convention", "offset", "scaling", "head_dim")


def rope_tables(c, positions, dev, dt):
    d = c["head_dim"]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    pos = torch.as_tensor(positions, dtype=torch.float64)
    if c["scaling"] == "linear2":
        pos = pos / 2.0
    ang = pos[:, None] * inv[None, :]                       # (L, d/2)
    return ang


def rope_ref(c, q, positions):
    """in-repo reference in float64: rotate-half pairs (i, i + d/2); interleaved pairs (2i, 2i + 1)."""
    ang = rope_tables(c, positions, q.device, q.dtype).to(q.device)
    cos, sin = torch.cos(ang).to(q.dtype), torch.sin(ang).to(q.dtype)
    d = q.shape[-1]
    if c["convention"] == "rotate_half":
        x1, x2 = q[..., : d // 2], q[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    x1, x2 = q[..., 0::2], q[..., 1::2]
    out = torch.empty_like(q)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def rope_call(cand, c, q, k, positions):
    lib = cand["lib"]
    if lib == "ref":
        return rope_ref(c, q, positions), rope_ref(c, k, positions)
    if c["convention"] != "rotate_half":
        raise NotImplementedError(f"{lib} implements rotate-half only (convention, not a deviation)")
    ang = rope_tables(c, positions, q.device, q.dtype).to(q.device)
    emb = torch.cat([ang, ang], -1)
    cos, sin = torch.cos(emb).to(q.dtype)[None], torch.sin(emb).to(q.dtype)[None]           # (1, L, d)
    if lib == "hf":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        return apply_rotary_pos_emb(q, k, cos, sin)
    if lib == "liger":
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        return liger_rotary_pos_emb(q, k, cos, sin)
    if lib == "unsloth":
        from unsloth.kernels.rope_embedding import fast_rope_embedding
        qq, kk = fast_rope_embedding(q.transpose(1, 2).contiguous().transpose(1, 2), k.transpose(1, 2).contiguous().transpose(1, 2),
                                     cos[0, :, : c["head_dim"] // 2], sin[0, :, : c["head_dim"] // 2])
        return qq, kk
    raise KeyError(lib)


def ro_run(cand, c, seed, variant):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    rng = _rng(f"ro/{c['id']}/{seed}")
    L, H, d = 11, 4, c["head_dim"]
    q = torch.tensor(rng.normal(0, 1, (2, H, L, d)), dtype=dt, device=dev, requires_grad=True)
    k = torch.tensor(rng.normal(0, 1, (2, H, L, d)), dtype=dt, device=dev, requires_grad=True)
    off = c["offset"] + (5 if variant == "shifted" else 0)
    positions = np.arange(L) + off if variant != "zero_pos" else np.zeros(L)
    qo, ko = rope_call(cand, c, q, k, positions)
    gq = torch.tensor(rng.normal(0, 1, qo.shape), dtype=dt, device=dev)
    (qo * gq).sum().add((ko * gq).sum()).backward()
    return {"q": npy(qo), "k": npy(ko), "dq": npy(q.grad), "dk": npy(k.grad), "q_in": npy(q), "k_in": npy(k)}


# ------------------------------------------------------------------------------------------------ moe

MO_KEYS = ("topk", "experts", "capacity", "routing")


def moe_inputs(c, seed):
    rng = _rng(f"moe/{c['id']}/{seed}")
    T, D, E = 13, 6, c["experts"]
    x = rng.normal(0, 1, (T, D)).astype(np.float32).astype(float)
    logits = rng.normal(0, 1, (T, E))
    if c["routing"] == "ties":
        logits = np.round(logits * 2) / 2                     # quantised: ties among the top experts
        logits[:, :2] = logits[:, :1]                         # experts 0 and 1 always tie
    elif c["routing"] == "one_expert":
        logits[:, 0] += 10.0                                  # (nearly) every token prefers expert 0
    logits = logits.astype(np.float32).astype(float)
    W = rng.normal(0, 0.5, (E, D, D)).astype(np.float32).astype(float)
    cap = {None: None, "exact": math.ceil(T * c["topk"] / E), "overflow": max(1, math.ceil(T * c["topk"] / E) // 2)}[c["capacity"]]
    return x, logits, W, cap


def route(logits_t, k):
    """top-k with ties broken by the lower expert index (torch.topk on CPU and CUDA is not specified for ties: the
    implementations sort the logits stably with the expert index as the tie key)."""
    E = logits_t.shape[-1]
    key = logits_t - torch.arange(E, dtype=logits_t.dtype, device=logits_t.device) * 0.0
    probs = torch.softmax(logits_t, -1)
    order = torch.argsort(-key, dim=-1, stable=True)[:, :k]
    w = torch.gather(probs, 1, order)
    w = w / w.sum(-1, keepdim=True)
    return order, w, probs


def moe_loop(c, x, logits, W, cap):
    T = x.shape[0]
    order, w, probs = route(logits, c["topk"])
    out = torch.zeros_like(x)
    load = [0] * c["experts"]
    dropped = torch.zeros(T, c["topk"], dtype=torch.bool)
    for t in range(T):                                        # tokens in order: capacity is first come, first served
        for j in range(c["topk"]):
            e = int(order[t, j])
            if cap is not None and load[e] >= cap:
                dropped[t, j] = True
                continue
            load[e] += 1
            out[t] += w[t, j] * (x[t] @ W[e])
    return out, order, w, probs, dropped


def moe_vectorised(c, x, logits, W, cap):
    T, K = x.shape[0], c["topk"]
    order, w, probs = route(logits, K)
    flat_e = order.reshape(-1)                                # token-major (t, j)
    flat_t = torch.arange(T, device=x.device).repeat_interleave(K)
    perm = torch.argsort(flat_e * (T * K) + torch.arange(T * K, device=x.device), stable=True)   # group by expert, keep token order
    e_sorted = flat_e[perm]
    rank = torch.arange(T * K, device=x.device) - torch.searchsorted(e_sorted, e_sorted, right=False)
    keep_sorted = torch.ones_like(rank, dtype=torch.bool) if cap is None else rank < cap
    xs = x[flat_t[perm]]                                      # permute
    ys = torch.bmm(xs[:, None, :], W[e_sorted]).squeeze(1)
    ys = ys * w.reshape(-1)[perm][:, None] * keep_sorted[:, None]
    out = torch.zeros_like(x).index_add(0, flat_t[perm], ys)   # unpermute + combine
    dropped = torch.zeros(T * K, dtype=torch.bool, device=x.device)
    dropped[perm] = ~keep_sorted
    unperm = torch.empty_like(xs)
    unperm[perm] = xs
    return out, order, w, probs, dropped.reshape(T, K), unperm.reshape(T, K, -1)[:, 0]


def mo_run(cand, c, seed, variant):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    xn, ln, Wn, cap = moe_inputs(c, seed)
    x, logits, W = (torch.tensor(a, dtype=dt, device=dev) for a in (xn, ln, Wn))
    if cand["impl"] == "loop":
        out, order, w, probs, dropped = moe_loop(c, x, logits, W, cap)
        unperm = x
    else:
        fn = moe_vectorised
        if cand["impl"] == "compiled":
            torch._dynamo.reset()
            fn = torch.compile(moe_vectorised)
        out, order, w, probs, dropped, unperm = fn(c, x, logits, W, cap)
    E = c["experts"]
    counts = torch.zeros(E, dtype=torch.float64)
    for e in order[:, 0].cpu().tolist():                      # Switch aux loss: fraction of tokens whose top-1 is e
        counts[e] += 1
    f = counts / x.shape[0]
    P = probs.double().cpu().mean(0)
    aux = float(E * (f * P).sum())
    return {"out": npy(out), "order": order.cpu().numpy(), "w": npy(w), "dropped": dropped.cpu().numpy(), "aux": np.array([aux]),
            "unperm": npy(unperm), "x": npy(x), "cap": cap}


# ------------------------------------------------------------------------------------------------ driver

FAMS = {
    "checkpoint": dict(keys=CK_KEYS, run=ck_run, variants=lambda c: ["off", "on"],
                       cands={"ka_main": [{"id": "torch_checkpoint_cpu", "device": "cpu", "dtype": "float64"},
                                          {"id": "torch_checkpoint_cuda", "device": "cuda", "dtype": "float32"},
                                          {"id": "torch_checkpoint_compiled", "device": "cuda", "dtype": "float32", "compiled": True}],
                              "liger": [{"id": "hf_gradient_checkpointing", "device": "cuda", "dtype": "float32", "hf": True}]}),
    "rope": dict(keys=RO_KEYS, run=ro_run, variants=lambda c: ["base", "shifted", "zero_pos"],
                 cands={"ka_main": [{"id": "ref_torch", "device": "cuda", "dtype": "float32", "lib": "ref"},
                                    {"id": "ref_torch_float64", "device": "cpu", "dtype": "float64", "lib": "ref"}],
                        "liger": [{"id": "hf_apply_rotary", "device": "cuda", "dtype": "float32", "lib": "hf"},
                                  {"id": "liger_rope", "device": "cuda", "dtype": "float32", "lib": "liger"},
                                  {"id": "unsloth_rope", "device": "cuda", "dtype": "float32", "lib": "unsloth"}]}),
    "moe": dict(keys=MO_KEYS, run=mo_run, variants=lambda c: ["base"],
                cands={"ka_main": [{"id": "ref_loop_cpu", "device": "cpu", "dtype": "float64", "impl": "loop"},
                                   {"id": "vectorised_scatter_cuda", "device": "cuda", "dtype": "float32", "impl": "vectorised"},
                                   {"id": "vectorised_compiled", "device": "cuda", "dtype": "float32", "impl": "compiled"}]}),
}


def run(fam, env):
    F_ = FAMS[fam]
    CACHE.mkdir(parents=True, exist_ok=True)
    for cand in F_["cands"].get(env, []):
        res, t0 = {}, time.time()
        for c in plan(fam, F_["keys"]):
            for seed in SEEDS:
                rec = {}
                for var in F_["variants"](c):
                    try:
                        rec[var] = {"status": "ok", "outputs": F_["run"](cand, c, seed, var)}
                    except NotImplementedError as exc:
                        rec[var] = {"status": "unsupported", "reason": str(exc)}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"}
                res[(c["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1), "env": env}
        (CACHE / f"{fam}__{cand['id']}.pkl").write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] += 1
        print(fam, cand["id"], dict(st), meta["seconds"], "s", flush=True)


def close(a, b, exact):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    if exact:
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.all((np.abs(a - b) <= TAU32 * (1 + np.abs(b))) | (np.isnan(a) & np.isnan(b))))


def analyse(fam):
    F_ = FAMS[fam]
    conds = {c["id"]: c for c in plan(fam, F_["keys"])}
    data = {p.stem.split("__", 1)[1]: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob(f"{fam}__*.pkl"))}
    out = {"family": fam, "conditions": len(conds), "candidates": {}, "examples": []}
    for name, d in data.items():
        cand = d["meta"]["candidate"]
        P, E = defaultdict(lambda: [0, 0]), [0, 0]
        st = defaultdict(int)
        for (cid, seed), rec in d["res"].items():
            c = conds[cid]
            for r in rec.values():
                st[r["status"]] += 1
                if r["status"] == "error" and len(out["examples"]) < 30:
                    out["examples"].append({"candidate": name, "condition": cid, "error": r["reason"]})
            if fam == "checkpoint":
                a, b = rec.get("off"), rec.get("on")
                if a and b and a["status"] == "ok" and b["status"] == "ok":
                    exact = cand["dtype"] == "float64" and cand["device"] == "cpu"
                    ok = all(close(b["outputs"][k], a["outputs"][k], exact) for k in a["outputs"])
                    P["P_checkpoint_on_equals_off"][0] += 1
                    P["P_checkpoint_on_equals_off"][1] += int(not ok)
                    if not ok:
                        bad = [k for k in a["outputs"] if not close(b["outputs"][k], a["outputs"][k], exact)]
                        out["examples"].append({"candidate": name, "condition": cid, "seed": seed, "outputs": bad[:5]})
            elif fam == "rope":
                b = rec.get("base")
                if not b or b["status"] != "ok":
                    continue
                o = b["outputs"]
                dd = c["head_dim"]
                pair = ((lambda v: v[..., : dd // 2] ** 2 + v[..., dd // 2:] ** 2) if c["convention"] == "rotate_half"
                        else (lambda v: v[..., 0::2] ** 2 + v[..., 1::2] ** 2))
                P["P_pair_norm_preserved"][0] += 1
                P["P_pair_norm_preserved"][1] += int(not close(pair(o["q"]), pair(o["q_in"]), False))
                sh = rec.get("shifted")
                if sh and sh["status"] == "ok":
                    dots = np.einsum("bhid,bhjd->bhij", o["q"], o["k"])
                    dots2 = np.einsum("bhid,bhjd->bhij", sh["outputs"]["q"], sh["outputs"]["k"])
                    P["P_relative_position"][0] += 1
                    P["P_relative_position"][1] += int(not close(dots2, dots, False))
                z = rec.get("zero_pos")
                if z and z["status"] == "ok":
                    P["P_identity_at_zero"][0] += 1
                    P["P_identity_at_zero"][1] += int(not close(z["outputs"]["q"], z["outputs"]["q_in"], True))
                ref = "ref_torch" if name != "ref_torch" else None
                if ref and ref in data:
                    rb = data[ref]["res"].get((cid, seed), {}).get("base")
                    if rb and rb["status"] == "ok":
                        ok = all(close(o[k], rb["outputs"][k], False) for k in ("q", "k", "dq", "dk"))
                        E[0] += 1
                        E[1] += int(not ok)
                        if not ok:
                            out["examples"].append({"candidate": name, "vs": ref, "condition": cid, "seed": seed})
            else:                                             # moe
                b = rec.get("base")
                if not b or b["status"] != "ok":
                    continue
                o = b["outputs"]
                P["P_permute_unpermute_identity"][0] += 1
                P["P_permute_unpermute_identity"][1] += int(not close(o["unperm"], o["x"], True))
                P["P_combine_weights_sum_to_1"][0] += 1
                P["P_combine_weights_sum_to_1"][1] += int(not close(o["w"].sum(-1), np.ones(o["w"].shape[0]), False))
                if o["dropped"].any():
                    xn, ln, Wn, cap = moe_inputs(c, seed)
                    # recompute each token's output from its kept assignments only (dropped ones must contribute zero)
                    exp = np.zeros_like(o["out"])
                    for t in range(o["out"].shape[0]):
                        for j in range(c["topk"]):
                            if not o["dropped"][t, j]:
                                exp[t] += o["w"][t, j] * (xn[t] @ Wn[int(o["order"][t, j])])
                    P["P_dropped_contribute_zero"][0] += 1
                    P["P_dropped_contribute_zero"][1] += int(not close(o["out"], exp, False))
                E_ = c["experts"]
                xn, ln, Wn, cap = moe_inputs(c, seed)
                pr = np.exp(ln - ln.max(-1, keepdims=True))
                pr /= pr.sum(-1, keepdims=True)
                fr = np.bincount(o["order"][:, 0], minlength=E_) / o["order"].shape[0]
                aux = E_ * float((fr * pr.mean(0)).sum())
                P["P_aux_loss_formula"][0] += 1
                P["P_aux_loss_formula"][1] += int(not close(o["aux"], np.array([aux]), False))
                ref = "ref_loop_cpu"
                if name != ref and ref in data:
                    rb = data[ref]["res"].get((cid, seed), {}).get("base")
                    if rb and rb["status"] == "ok":
                        ok = close(o["out"], rb["outputs"]["out"], False) and np.array_equal(o["order"], rb["outputs"]["order"]) \
                            and np.array_equal(o["dropped"], rb["outputs"]["dropped"])
                        E[0] += 1
                        E[1] += int(not ok)
                        if not ok:
                            out["examples"].append({"candidate": name, "vs": ref, "condition": cid, "seed": seed,
                                                    "order_equal": bool(np.array_equal(o["order"], rb["outputs"]["order"])),
                                                    "dropped_equal": bool(np.array_equal(o["dropped"], rb["outputs"]["dropped"]))})
        out["candidates"][name] = {"meta": {k: v for k, v in d["meta"].items() if k != "candidate"}, "statuses": dict(st),
                                   "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()}, "E": f"{E[1]}/{E[0]}" if E[0] else None}
        print(f"{name:28s} {dict(st)} P {out['candidates'][name]['P']} E {out['candidates'][name]['E']}")
    for e in out["examples"][:20]:
        print("  ", e)
    (OUT / fam).mkdir(parents=True, exist_ok=True)
    (OUT / fam / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--family", required=True, choices=list(FAMS))
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.family, a.env) if a.stage == "run" else analyse(a.family)
