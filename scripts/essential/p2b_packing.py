#!/usr/bin/env python3
"""2b family "packing" (protocol v2 section 3; registry tier1.packing): E and P.  F stays closed until the spec is delivered.

Conditions: the coverage plan's (docs, lengths) combinations with the factor ``impl`` projected out (every implementation is
a candidate) -- 9 combinations x 3 seeds.  Document lengths: equal = 12 each; unequal = [13] / [5, 19] / [3, 11, 6, 17, 9];
one_token_doc = [1] / [1, 15] / [6, 1, 9, 1, 12].

Candidates.  Attention level (B = 1, 4 heads, head_dim 16, float32, CUDA): SDPA (efficient backend) with a block-diagonal
causal boolean mask, flex_attention (compiled) with a document & causal mask_mod; their eager references are SDPA math with
the same mask and flex_attention uncompiled.  Model level (liger env): a 2-layer Llama (transformers 4.57.3, random weights,
hidden 64, 4 query / 2 KV heads, float32) fed the packed sequence with position_ids reset per document and no attention mask
(the documented packed-sequence path), attn_implementation "sdpa"; reference: the same weights with "eager".

P (pre-registered): the packed output of each document equals the document run alone (τ32 per element), forward and backward
(attention level: dq, dk, dv with the upstream sliced per document; model level: logits and the gradient of the summed
logits weighted by a fixed random upstream with respect to the input embeddings).

    python scripts/essential/p2b_packing.py run --env ka_main
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_packing.py run --env liger
    python scripts/essential/p2b_packing.py analyse
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
CACHE = ROOT / ".cache/essential/p2b/packing"
OUT = ROOT / "results/essential/phase2b/packing"
SEEDS = (0, 1, 2)
H, D, TAU32 = 4, 16, 2.0 ** -12
LENGTHS = {("equal", 1): [12], ("equal", 2): [12, 12], ("equal", 5): [12] * 5,
           ("unequal", 1): [13], ("unequal", 2): [5, 19], ("unequal", 5): [3, 11, 6, 17, 9],
           ("one_token_doc", 1): [1], ("one_token_doc", 2): [1, 15], ("one_token_doc", 5): [6, 1, 9, 1, 12]}


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["packing"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = (c["docs"], c["lengths"])
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            seen[key]["planned_impls"].append(c["impl"])
            continue
        d = {"id": f"pack_{c['docs']}docs_{c['lengths']}", "docs": c["docs"], "lengths": c["lengths"],
             "doc_lengths": LENGTHS[(c["lengths"], c["docs"])], "high_risk": bool(c.get("high_risk")), "planned_impls": [c["impl"]]}
        seen[key] = d
        out.append(d)
    return out


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def bounds(cond):
    e = np.cumsum([0] + cond["doc_lengths"])
    return list(zip(e[:-1], e[1:]))


def doc_mask(cond):
    t = sum(cond["doc_lengths"])
    doc = np.repeat(np.arange(len(cond["doc_lengths"])), cond["doc_lengths"])
    i, j = np.arange(t)[:, None], np.arange(t)[None, :]
    return (doc[:, None] == doc[None, :]) & (j <= i)


# ------------------------------------------------------------------------------------------------ attention level

def attn_candidates():
    return [{"id": "sdpa_block_mask", "kind": "sdpa", "backend": "EFFICIENT_ATTENTION"},
            {"id": "sdpa_math_block_mask", "kind": "sdpa", "backend": "MATH", "reference_only": True},
            {"id": "flex_block_mask", "kind": "flex", "compiled": True},
            {"id": "flex_eager_block_mask", "kind": "flex", "reference_only": True}]


def attn_call(cand, q, k, v, mask_np, compiled_cache):
    if cand["kind"] == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([getattr(SDPBackend, cand["backend"])]):
            if mask_np is None:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.as_tensor(mask_np, device=q.device)[None, None])
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    t = q.shape[2]
    m = torch.as_tensor(mask_np if mask_np is not None else np.tril(np.ones((t, t), dtype=bool)), device=q.device)

    def mask_mod(b, h, qi, ki):
        return m[qi, ki]
    bm = create_block_mask(mask_mod, 1, None, t, t, device=q.device)
    fn = flex_attention
    if cand.get("compiled"):
        if t not in compiled_cache:
            compiled_cache[t] = torch.compile(flex_attention, dynamic=False)
        fn = compiled_cache[t]
    return fn(q, k, v, block_mask=bm)


def attn_run(cand, cond, seed, cache):
    rng = _rng(f"pack/{cond['id']}/{seed}")
    t = sum(cond["doc_lengths"])
    q, k, v, g = (rng.normal(0, 1, (1, H, t, D)) for _ in range(4))
    f = lambda x: x.detach().double().cpu().numpy()      # noqa: E731

    def go(sl, mask):
        qq, kk, vv = (torch.tensor(a[:, :, sl], dtype=torch.float32, device="cuda", requires_grad=True) for a in (q, k, v))
        o = attn_call(cand, qq, kk, vv, mask, cache)
        o.backward(torch.tensor(g[:, :, sl], dtype=torch.float32, device="cuda"))
        return {"out": f(o), "dq": f(qq.grad), "dk": f(kk.grad), "dv": f(vv.grad)}

    packed = go(slice(0, t), doc_mask(cond))
    alone = [go(slice(a, b), None) for a, b in bounds(cond)]
    return {"packed": packed, "alone": alone}


# ------------------------------------------------------------------------------------------------ model level (liger env)

def hf_candidates():
    return [{"id": "hf_packed", "impl": "sdpa"}, {"id": "hf_packed_eager", "impl": "eager", "reference_only": True}]


def hf_model(impl):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1234)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=101, max_position_embeddings=128, attn_implementation=impl)
    return LlamaForCausalLM(cfg).to("cuda").float().eval()


def hf_run(model, cond, seed):
    rng = _rng(f"pack-hf/{cond['id']}/{seed}")
    t = sum(cond["doc_lengths"])
    ids = rng.integers(0, 101, (1, t))
    up = rng.normal(0, 1, (1, t, 101))
    pos = np.concatenate([np.arange(n) for n in cond["doc_lengths"]])[None]
    f = lambda x: x.detach().double().cpu().numpy()      # noqa: E731

    def go(sl, position_ids):
        emb = model.get_input_embeddings()(torch.as_tensor(ids[:, sl], device="cuda")).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, position_ids=torch.as_tensor(position_ids, device="cuda"), use_cache=False)
        (out.logits * torch.tensor(up[:, sl], dtype=torch.float32, device="cuda")).sum().backward()
        return {"logits": f(out.logits), "demb": f(emb.grad)}

    packed = go(slice(0, t), pos)
    alone = [go(slice(a, b), np.arange(b - a)[None]) for a, b in bounds(cond)]
    return {"packed": packed, "alone": alone}


def run(env):
    CACHE.mkdir(parents=True, exist_ok=True)
    cands = attn_candidates() if env == "ka_main" else hf_candidates()
    for cand in cands:
        path = CACHE / f"{cand['id']}.pkl"
        if path.exists():
            continue
        res, t0 = {}, time.time()
        model = hf_model(cand["impl"]) if env == "liger" else None
        for cond in conditions():
            cache = {}
            if cand.get("compiled"):
                torch._dynamo.reset()
            for seed in SEEDS:
                try:
                    out = hf_run(model, cond, seed) if env == "liger" else attn_run(cand, cond, seed, cache)
                    res[(cond["id"], seed)] = {"status": "ok", **out}
                except Exception as exc:  # noqa: BLE001
                    res[(cond["id"], seed)] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1), "env": env}
        path.write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for r in res.values():
            st[r["status"]] += 1
        print(cand["id"], dict(st), meta["seconds"], "s", flush=True)


# ------------------------------------------------------------------------------------------------ analysis

REFERENCE = {"sdpa_block_mask": "sdpa_math_block_mask", "flex_block_mask": "flex_eager_block_mask", "hf_packed": "hf_packed_eager"}


def viol(k, r):
    k, r = np.asarray(k), np.asarray(r)
    if k.shape != r.shape:
        return k.size or 1, 1, np.inf
    d = np.abs(k - r)
    lim = TAU32 * (1 + np.abs(r))
    bad = ~np.isfinite(k) | ~np.isfinite(r) | (d > lim)
    return int(bad.sum()), int(bad.size), float(np.max(d / lim, initial=0))


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"conditions": len(conds), "candidates": {}, "examples": []}
    for name, d in data.items():
        e_t, p_t = [0, 0, 0.0], [0, 0, 0.0]
        for (cid, seed), r in d["res"].items():
            if r["status"] != "ok":
                continue
            cond = conds[cid]
            # P: each document of the packed run == the document alone
            bad_any, worst = False, 0.0
            for (a, b), al in zip(bounds(cond), r["alone"]):
                for x, v in al.items():
                    pk = r["packed"][x]
                    sl = pk[:, :, a:b] if pk.ndim == 4 else pk[:, a:b]
                    nb, _, ratio = viol(sl, v)
                    worst = max(worst, ratio)
                    bad_any |= nb > 0
            p_t[0] += 1
            p_t[1] += int(bad_any)
            p_t[2] = max(p_t[2], worst)
            if bad_any and len(out["examples"]) < 30:
                out["examples"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_no_leakage", "max_ratio": worst})
            ref = REFERENCE.get(name)
            if ref in data:
                rr = data[ref]["res"].get((cid, seed))
                if rr and rr["status"] == "ok":
                    bad_e, w = False, 0.0
                    for x in r["packed"]:
                        nb, _, ratio = viol(r["packed"][x], rr["packed"][x])
                        bad_e |= nb > 0
                        w = max(w, ratio)
                    e_t[0] += 1
                    e_t[1] += int(bad_e)
                    e_t[2] = max(e_t[2], w)
                    if bad_e and len(out["examples"]) < 30:
                        out["examples"].append({"candidate": name, "condition": cid, "seed": seed, "property": "E", "max_ratio": w})
        st = defaultdict(int)
        for r in d["res"].values():
            st[r["status"]] += 1
        out["candidates"][name] = {"statuses": dict(st), "seconds": d["meta"]["seconds"],
                                   "E": {"reference": REFERENCE.get(name), "violating": e_t[1], "compared": e_t[0], "max_ratio": e_t[2]}
                                   if REFERENCE.get(name) in data else None,
                                   "P_no_leakage": {"violating": p_t[1], "checked": p_t[0], "max_ratio": p_t[2]}}
        c = out["candidates"][name]
        print(f"{name:24s} {dict(st)} E {c['E'] and (str(c['E']['violating']) + '/' + str(c['E']['compared']) + ' max %.3g' % c['E']['max_ratio'])}"
              f"  P_no_leakage {p_t[1]}/{p_t[0]} max {p_t[2]:.3g}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.env) if a.stage == "run" else analyse()
