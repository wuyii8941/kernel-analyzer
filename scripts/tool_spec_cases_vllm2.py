"""vLLM main (2026-10-05) Triton kernels, second batch: breadth over the kernel families vLLM serves with.

Imported from .cache/pylibs/vllm_shim (scripts/build_vllm_shim.py).  FnCase (make / run / ref): the kernel runs on the
case's dtype (fp16 unless noted), the specification is the documented semantics in float64 on the same inputs
(declared bound 2^-40 * max|f|).  Rounding steps inside a kernel (casts to the storage dtype between fused stages) are
numerical: K_R takes them as exact, so they land in e_num, not e_sem.

Environment: ka_main + PYTHONPATH=.cache/pylibs/vllm_shim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_cases_tridao import FnCase, rn

CASES = []
H16 = torch.float16


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _softmax_attn(q, k, v, keep, scale, cap=0.0, sink=None, alibi=None, rel=None):
    """q (H, Lq, D), k/v (H, Lk, D) float64; keep (Lq, Lk) or (H, Lq, Lk) bool."""
    s = torch.einsum("hqd,hkd->hqk", q, k) * scale
    if cap > 0:
        s = cap * torch.tanh(s / cap)
    if alibi is not None:
        s = s + alibi[:, None, None] * rel[None]
    s = s.masked_fill(~keep, float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink[:, None, None].expand(s.shape[0], s.shape[1], 1)], -1)
        p = torch.softmax(s, -1)[..., :-1]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("hqk,hkd->hqd", p, v), torch.logsumexp(s, -1)


# ---------------------------------------------------------------------------------------------------------------
# context_attention_fwd (v1/attention/ops/triton_prefill_attention.py): dense varlen prefill, encoder attention
# ---------------------------------------------------------------------------------------------------------------


def _prefill_case(name, lens, H, Hk, D, causal, wq=0, wk=0, sinks=False):
    def make(g):
        T = sum(lens)
        d = {"q": rn(g, T, H, D).half(), "k": rn(g, T, Hk, D).half(), "v": rn(g, T, Hk, D).half()}
        if sinks:
            d["sinks"] = 2 * rn(g, H)
        return d

    def run(inp):
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
        q = inp["q"]
        o = torch.empty_like(q)
        start = _cu(lens)[:-1].to(q.device)
        context_attention_fwd(q, inp["k"], inp["v"], o, start, torch.tensor(lens, dtype=torch.int32, device=q.device),
                              max(lens), is_causal=causal, softmax_scale=D ** -0.5,
                              sliding_window_q=wq or None, sliding_window_k=wk or None, sinks=inp.get("sinks"))
        return {"o": o}

    def ref(inp):
        outs, s0 = [], 0
        for L in lens:
            q = inp["q"][s0:s0 + L].transpose(0, 1)
            k = inp["k"][s0:s0 + L].repeat_interleave(H // Hk, 1).transpose(0, 1)
            v = inp["v"][s0:s0 + L].repeat_interleave(H // Hk, 1).transpose(0, 1)
            pq = torch.arange(L, device=q.device)[:, None]
            pk = torch.arange(L, device=q.device)[None, :]
            keep = torch.ones(L, L, dtype=torch.bool, device=q.device)
            if causal:
                keep &= pq >= pk
            if wq:
                keep &= pq - pk <= wq
            if wk:
                keep &= pk - pq <= wk
            o, _ = _softmax_attn(q, k, v, keep, D ** -0.5, sink=inp.get("sinks"))
            outs.append(o.transpose(0, 1))
            s0 += L
        return {"o": torch.cat(outs)}

    return FnCase(f"v2_prefill_{name}", "vllm context_attention_fwd (triton_prefill_attention)",
                  f"varlen attention lens={lens} H={H}/{Hk} D={D} causal={causal} window=({wq},{wk}) sinks={sinks}",
                  make, run, ref)


CASES += [
    _prefill_case("causal", [37, 150], 4, 2, 64, True),
    _prefill_case("bidir", [37, 150], 4, 4, 64, False),
    _prefill_case("bidir_sw", [37, 150], 4, 2, 64, False, wq=15, wk=15),
    _prefill_case("causal_sinks_d80", [37, 150], 4, 2, 80, True, sinks=True),
]

# ---------------------------------------------------------------------------------------------------------------
# decode_attention_fwd (triton_decode_attention.py): split-KV paged decode, MHA / GQA / MLA, logit cap
# ---------------------------------------------------------------------------------------------------------------


def _decode_case(name, seq_lens, H, Hk, Dk, Dv, page, splits, cap=0.0, mla=False):
    B = len(seq_lens)

    def make(g):
        npg = [(L + page - 1) // page for L in seq_lens]
        total = sum(npg) + 3
        perm = torch.randperm(total, generator=g)
        r2t = torch.zeros(B, max(npg), dtype=torch.int32)
        o = 0
        for b in range(B):
            r2t[b, :npg[b]] = perm[o:o + npg[b]]
            o += npg[b]
        d = {"q": rn(g, B, H, Dk).half(), "kbuf": rn(g, total, page, Hk, Dk).half(), "r2t": r2t,
             "seq": torch.tensor(seq_lens, dtype=torch.int32)}
        if not mla:
            d["vbuf"] = rn(g, total, page, Hk, Dv).half()
        return d

    def run(inp):
        from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
        q, kb = inp["q"], inp["kbuf"]
        vb = kb[..., :Dv] if mla else inp["vbuf"]
        o = torch.empty(B, H, Dv, device=q.device, dtype=q.dtype)
        lse = torch.empty(B, H, device=q.device, dtype=torch.float32)
        logits = torch.empty(B, H, splits, Dv + 1, device=q.device, dtype=torch.float32)
        decode_attention_fwd(q, kb, vb, o, lse, inp["r2t"], inp["seq"], logits, splits, Dk ** -0.5, page,
                             logit_cap=cap, is_mla=mla)
        return {"o": o, "lse": lse}

    def ref(inp):
        o, lse = [], []
        for b in range(B):
            L = int(inp["seq"][b])
            tok = torch.arange(L, device=inp["q"].device)
            pages = inp["r2t"][b].long()[tok // page]
            k = inp["kbuf"][pages, tok % page]            # (L, Hk, Dk)
            v = (inp["kbuf"][..., :Dv] if mla else inp["vbuf"])[pages, tok % page]
            k = k.repeat_interleave(H // Hk, 1).transpose(0, 1)
            v = v.repeat_interleave(H // Hk, 1).transpose(0, 1)
            keep = torch.ones(1, L, dtype=torch.bool, device=k.device)
            ob, lb = _softmax_attn(inp["q"][b][:, None, :], k, v, keep, Dk ** -0.5, cap=cap)
            o.append(ob[:, 0])
            lse.append(lb[:, 0])
        return {"o": torch.stack(o), "lse": torch.stack(lse)}

    return FnCase(f"v2_decode_{name}", "vllm decode_attention_fwd (triton_decode_attention)",
                  f"paged decode seq={seq_lens} H={H}/{Hk} Dk={Dk} Dv={Dv} page={page} splits={splits} cap={cap} "
                  f"mla={mla}; lse = logsumexp of the (capped) scores", make, run, ref)


CASES += [
    _decode_case("mha", [300, 17, 64], 4, 4, 64, 64, 16, 4),
    _decode_case("gqa4_cap", [300, 17, 64], 8, 2, 64, 64, 16, 4, cap=30.0),
    _decode_case("gqa_splits_gt_len", [3, 1, 40], 8, 2, 128, 128, 1, 8),
    _decode_case("mla", [300, 17, 64], 16, 1, 576, 512, 32, 4, mla=True),
]

# ---------------------------------------------------------------------------------------------------------------
# merge_attn_states (triton_merge_attn_states.py)
# ---------------------------------------------------------------------------------------------------------------


def _merge_case(name, T, H, D, prefill_with_ctx=None):
    def make(g):
        return {"pa": rn(g, T, H, D).half(), "sa": rn(g, T, H, D).half(), "pl": 3 * rn(g, H, T), "sl": 3 * rn(g, H, T)}

    def run(inp):
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states
        out = torch.empty_like(inp["pa"])
        lse = torch.empty_like(inp["pl"])
        merge_attn_states(out, inp["pa"], inp["pl"], inp["sa"], inp["sl"], lse,
                          prefill_tokens_with_context=prefill_with_ctx)
        return {"out": out, "lse": lse}

    def ref(inp):
        n = T if prefill_with_ctx is None else prefill_with_ctx
        pl, sl = inp["pl"].clone(), inp["sl"]
        pl[:, n:] = float("-inf")  # tokens without context take the suffix as is
        m = torch.maximum(pl, sl)
        wp, ws = torch.exp(pl - m), torch.exp(sl - m)
        out = inp["pa"] * (wp / (wp + ws)).t()[..., None] + inp["sa"] * (ws / (wp + ws)).t()[..., None]
        return {"out": out, "lse": m + torch.log(wp + ws)}

    return FnCase(f"v2_merge_{name}", "vllm merge_attn_states (Triton)",
                  f"LSE-weighted merge of prefix and suffix attention T={T} H={H} D={D} "
                  f"prefill_tokens_with_context={prefill_with_ctx}", make, run, ref)


CASES += [_merge_case("d128", 37, 8, 128), _merge_case("d80_partial_ctx", 37, 8, 80, prefill_with_ctx=20)]

# ---------------------------------------------------------------------------------------------------------------
# prefix_prefill.context_attention_fwd: paged context (k_cache [blocks, Hk, D/8, bs, 8]) + new tokens;
# chunked_prefill_paged_decode: the same plus a decode kernel for 1-token queries
# ---------------------------------------------------------------------------------------------------------------


def _paged_inputs(g, qlens, ctx_lens, Hk, D, bs, H):
    B = len(qlens)
    seq = [a + b for a, b in zip(qlens, ctx_lens)]
    nb = [(L + bs - 1) // bs for L in seq]
    total = sum(nb) + 2
    perm = torch.randperm(total, generator=g)
    bt = torch.zeros(B, max(nb), dtype=torch.int32)
    o = 0
    for b in range(B):
        bt[b, :nb[b]] = perm[o:o + nb[b]]
        o += nb[b]
    kd = [rn(g, L, Hk, D).half() for L in seq]
    vd = [rn(g, L, Hk, D).half() for L in seq]
    kc = torch.zeros(total, bs, Hk, D, dtype=H16)
    vc = torch.zeros(total, bs, Hk, D, dtype=H16)
    for b in range(B):
        for t in range(seq[b]):  # the whole sequence is in the cache (the kernels read context from it)
            kc[bt[b, t // bs], t % bs] = kd[b][t]
            vc[bt[b, t // bs], t % bs] = vd[b][t]
    q = torch.cat([rn(g, L, H, D).half() for L in qlens])
    k_new = torch.cat([kd[b][ctx_lens[b]:] for b in range(B)])
    v_new = torch.cat([vd[b][ctx_lens[b]:] for b in range(B)])
    k5, v4 = _cache_views(kc, vc)  # vLLM's paged layouts, built here so the kernel reads case inputs
    return {"q": q, "k": k_new, "v": v_new, "k5": k5, "v4": v4, "bt": bt, "kd_flat": torch.cat(kd),
            "vd_flat": torch.cat(vd)}


def _paged_ref(inp, qlens, ctx_lens, H, Hk, D, window=0, sink=None):
    outs, q0, s0 = [], 0, 0
    for ql, cl in zip(qlens, ctx_lens):
        L = ql + cl
        q = inp["q"][q0:q0 + ql].transpose(0, 1)
        k = inp["kd_flat"][s0:s0 + L].repeat_interleave(H // Hk, 1).transpose(0, 1)
        v = inp["vd_flat"][s0:s0 + L].repeat_interleave(H // Hk, 1).transpose(0, 1)
        pq = torch.arange(cl, L, device=q.device)[:, None]
        pk = torch.arange(L, device=q.device)[None, :]
        keep = pk <= pq
        if window:
            keep &= pq - pk < window
        o, _ = _softmax_attn(q, k, v, keep, D ** -0.5, sink=sink)
        outs.append(o.transpose(0, 1))
        q0, s0 = q0 + ql, s0 + L
    return torch.cat(outs)


def _cache_views(kc, vc, x=8):
    nbk, bs, Hk, D = kc.shape
    k5 = kc.view(nbk, bs, Hk, D // x, x).permute(0, 2, 3, 1, 4).contiguous()
    v4 = vc.permute(0, 2, 3, 1).contiguous()
    return k5, v4


def _prefix_case(name, qlens, ctx_lens, H, Hk, D, bs=16, window=0, sinks=False):
    def make(g):
        d = _paged_inputs(g, qlens, ctx_lens, Hk, D, bs, H)
        if sinks:
            d["sinks"] = 2 * rn(g, H)
        return d

    def run(inp):
        from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd
        q = inp["q"]
        o = torch.empty_like(q)
        k5, v4 = inp["k5"], inp["v4"]
        one = torch.tensor(1.0, device=q.device)
        context_attention_fwd(q, inp["k"], inp["v"], o, "auto", k5, v4, inp["bt"], _cu(qlens).to(q.device),
                              torch.tensor([a + b for a, b in zip(qlens, ctx_lens)], dtype=torch.int32,
                                           device=q.device),
                              max(a + b for a, b in zip(qlens, ctx_lens)), max(qlens), one, one,
                              sliding_window=window or None, sm_scale=D ** -0.5, sinks=inp.get("sinks"))
        return {"o": o}

    def ref(inp):
        return {"o": _paged_ref(inp, qlens, ctx_lens, H, Hk, D, window, inp.get("sinks"))}

    return FnCase(f"v2_prefix_{name}", "vllm prefix_prefill.context_attention_fwd",
                  f"paged-context prefill q={qlens} ctx={ctx_lens} H={H}/{Hk} D={D} block={bs} window={window} "
                  f"sinks={sinks}", make, run, ref)


def _chunked_case(name, qlens, ctx_lens, H, Hk, D, bs=16, window=0):
    def make(g):
        return _paged_inputs(g, qlens, ctx_lens, Hk, D, bs, H)

    def run(inp):
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
        q = inp["q"]
        o = torch.empty_like(q)
        k5, v4 = inp["k5"], inp["v4"]
        one = torch.tensor(1.0, device=q.device)
        seq = torch.tensor([a + b for a, b in zip(qlens, ctx_lens)], dtype=torch.int32, device=q.device)
        chunked_prefill_paged_decode(q, inp["k"], inp["v"], o, "auto", k5, v4, inp["bt"], _cu(qlens).to(q.device),
                                     seq, int(seq.max()), max(qlens), one, one, sliding_window=window or None,
                                     sm_scale=D ** -0.5)
        return {"o": o}

    def ref(inp):
        return {"o": _paged_ref(inp, qlens, ctx_lens, H, Hk, D, window)}

    return FnCase(f"v2_chunked_{name}", "vllm chunked_prefill_paged_decode",
                  f"mixed prefill / decode q={qlens} ctx={ctx_lens} H={H}/{Hk} D={D} block={bs} window={window}",
                  make, run, ref)


CASES += [
    _prefix_case("gqa", [20, 33], [40, 7], 8, 2, 64),
    _prefix_case("sw_sinks", [20, 33], [40, 7], 8, 2, 64, window=24, sinks=True),
    _chunked_case("mixed", [1, 20, 1, 9], [70, 40, 3, 0], 8, 2, 64),
    _chunked_case("mixed_sw", [1, 20, 1], [70, 40, 33], 8, 2, 64, window=32),
]

# ---------------------------------------------------------------------------------------------------------------
# MRoPE (rotary_embedding/mrope.py triton_mrope) and the fused QK-RMSNorm + RoPE + gate kernel (Qwen3.5)
# ---------------------------------------------------------------------------------------------------------------


def _rot(x1, x2, c, s):
    return x1 * c - x2 * s, x2 * c + x1 * s


def _mrope_case(name, T, H, Hk, hd, rd, section, interleaved, neox):
    def make(g):
        pos = torch.randint(0, 64, (3, T), generator=g)
        inv = 1.0 / (10000 ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
        f = pos[..., None].double() * inv  # (3, T, rd/2)
        return {"q": rn(g, T, H * hd).half(), "k": rn(g, T, Hk * hd).half(), "cos": f.cos().float(),
                "sin": f.sin().float()}

    def run(inp):
        from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
        q, k = triton_mrope(inp["q"], inp["k"], inp["cos"], inp["sin"], section, hd, rd, interleaved, neox)
        return {"q": q, "k": k}

    def sel(x):  # (3, T, rd/2) -> (T, rd/2)
        c = torch.arange(rd // 2, device=x.device)
        if interleaved:
            h = (c % 3 == 1) & (c < 3 * section[1])
            w = (c % 3 == 2) & (c < 3 * section[2])
            return torch.where(h, x[1], torch.where(w, x[2], x[0]))
        t_end, h_end = section[0], section[0] + section[1]
        return torch.where(c < t_end, x[0], torch.where(c < h_end, x[1], x[2]))

    def apply(x, n, c, s):
        x = x.view(T, n, hd).clone()
        r = x[..., :rd]
        if neox:
            a, b = _rot(r[..., :rd // 2], r[..., rd // 2:], c[:, None], s[:, None])
            r2 = torch.cat([a, b], -1)
        else:
            a, b = _rot(r[..., 0::2], r[..., 1::2], c[:, None], s[:, None])
            r2 = torch.stack([a, b], -1).flatten(-2)
        x[..., :rd] = r2
        return x.view(T, n * hd)

    def ref(inp):
        c, s = sel(inp["cos"]), sel(inp["sin"])
        return {"q": apply(inp["q"], H, c, s), "k": apply(inp["k"], Hk, c, s)}

    return FnCase(f"v2_mrope_{name}", "vllm triton_mrope", f"MRoPE section={section} interleaved={interleaved} "
                  f"neox={neox} head={hd} rotary={rd} (Qwen2-VL / Qwen3-VL); cos/sin given in fp32", make, run, ref)


CASES += [
    _mrope_case("qwen2vl", 37, 8, 2, 128, 128, [16, 24, 24], False, True),
    _mrope_case("qwen3vl_interleaved", 37, 8, 2, 128, 128, [24, 20, 20], True, True),
    _mrope_case("partial_gptj", 37, 6, 2, 128, 64, [8, 12, 12], False, False),
]


def _qknorm_case(name, T, H, Hk, hd, rd, mrope=None):
    def make(g):
        d = {"qg": rn(g, T, H * 2 * hd).bfloat16(), "k": rn(g, T, Hk * hd).bfloat16(),
             "qw": 0.2 * rn(g, hd), "kw": 0.2 * rn(g, hd)}
        inv = 1.0 / (10000 ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
        f = torch.arange(256, dtype=torch.float64)[:, None] * inv
        d["cache"] = torch.cat([f.cos(), f.sin()], -1).float()
        d["pos"] = torch.randint(0, 256, (3, T) if mrope else (T,), generator=g)
        return d

    def run(inp):
        from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
        q, k, gate = fused_qk_rmsnorm_rope_gate(inp["qg"], inp["k"], inp["qw"], inp["kw"], inp["cache"], inp["pos"],
                                                1e-6, H, Hk, hd, rd, mrope_section=mrope, norm_beta=1.0)
        return {"q": q, "k": k, "gate": gate}

    def ref(inp):
        qg = inp["qg"].view(T, H, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = inp["k"].view(T, Hk, hd)

        def norm(x, w):
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * (w + 1.0)

        pos = inp["pos"]
        half = rd // 2
        if mrope:
            c_ = torch.arange(half, device=pos.device)
            hsel = (c_ % 3 == 1) & (c_ < 3 * mrope[1])
            wsel = (c_ % 3 == 2) & (c_ < 3 * mrope[2])
            p = torch.where(hsel, pos[1][:, None], torch.where(wsel, pos[2][:, None], pos[0][:, None]))
            cos = torch.gather(inp["cache"][:, :half], 0, p)
            sin = torch.gather(inp["cache"][:, half:], 0, p)
        else:
            cos, sin = inp["cache"][pos, :half], inp["cache"][pos, half:]

        def rope(x):
            x = x.clone()
            a, b = _rot(x[..., :half], x[..., half:rd], cos[:, None], sin[:, None])
            x[..., :half], x[..., half:rd] = a, b
            return x

        return {"q": rope(norm(q, inp["qw"])).reshape(T, H * hd), "k": rope(norm(k, inp["kw"])).reshape(T, Hk * hd),
                "gate": gate.reshape(T, H * hd)}

    return FnCase(f"v2_qknorm_rope_{name}", "vllm fused_qk_rmsnorm_rope_gate (Qwen3-Next / Qwen3.5 attention)",
                  f"split -> Gemma RMSNorm (w + 1) -> partial NeoX RoPE rd={rd}/{hd} (mrope={mrope}) -> gate copy, "
                  f"bf16", make, run, ref)


CASES += [_qknorm_case("text", 19, 8, 2, 256, 64), _qknorm_case("mrope", 19, 8, 2, 256, 64, mrope=[11, 11, 10])]

# ---------------------------------------------------------------------------------------------------------------
# triton_reshape_and_cache_flash: KV cache write by slot mapping (padding slot -1 ignored)
# ---------------------------------------------------------------------------------------------------------------


def _cache_write_make(g):
    T, Hk, D, nb, bs = 23, 4, 80, 9, 16
    slots = torch.randperm(nb * bs, generator=g)[:T].long()
    slots[[3, 11]] = -1
    return {"key": rn(g, T, Hk, D).half(), "value": rn(g, T, Hk, D).half(), "kc": rn(g, nb, bs, Hk, D).half(),
            "vc": rn(g, nb, bs, Hk, D).half(), "slots": slots}


def _cache_write_run(inp):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
    kc, vc = inp["kc"], inp["vc"]
    one = torch.tensor(1.0, device=kc.device)
    triton_reshape_and_cache_flash(inp["key"], inp["value"], kc, vc, inp["slots"], "auto", one, one)
    return {"kc": kc, "vc": vc}


def _cache_write_ref(inp):
    kc, vc = inp["kc"].clone(), inp["vc"].clone()
    nb, bs = kc.shape[:2]
    for t, s in enumerate(inp["slots"].tolist()):
        if s >= 0:
            kc.view(nb * bs, *kc.shape[2:])[s] = inp["key"][t]
            vc.view(nb * bs, *vc.shape[2:])[s] = inp["value"][t]
    return {"kc": kc, "vc": vc}


CASES += [FnCase("v2_reshape_and_cache_flash", "vllm triton_reshape_and_cache_flash",
                 "KV cache write cache[slot] = key/value, slot -1 skipped", _cache_write_make, _cache_write_run,
                 _cache_write_ref)]

# ===============================================================================================================
# Sampling (v1/worker/gpu/sample: Model Runner V2) and speculative decoding (v1/sample/rejection_sampler.py)
# ===============================================================================================================


def _logits_rows(g, B, V, frac_masked=0.0):
    x = 3 * rn(g, B, V)
    if frac_masked:
        x[torch.rand(B, V, generator=g) < frac_masked] = float("-inf")
    return x


def _min_p_make(g):
    B, V = 6, 3000
    return {"logits": _logits_rows(g, B, V, 0.1), "idx": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
            "min_p": torch.tensor([0.0, 0.05, 0.3, 0.9, 1e-4, 0.5])}


def _min_p_run(inp):
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    x = inp["logits"]
    apply_min_p(x, inp["idx"], inp["min_p"])
    return {"logits": x}


def _min_p_ref(inp):
    x = inp["logits"].clone()
    for r in range(x.shape[0]):
        mp = float(inp["min_p"][inp["idx"][r]])
        if mp == 0.0:
            continue
        thr = x[r].max() + torch.log(torch.tensor(mp, dtype=torch.float64))
        x[r] = x[r].masked_fill(x[r] < thr, float("-inf"))
    return {"logits": x}


CASES += [FnCase("v2_min_p", "vllm V2 sampler apply_min_p", "logits below max + log(min_p) -> -inf (row's min_p 0: "
                 "unchanged); min_p as stored (fp32)", _min_p_make, _min_p_run, _min_p_ref)]


def _temp_make(g):
    return {"logits": _logits_rows(g, 5, 9000).bfloat16(), "idx": torch.tensor([4, 3, 2, 1, 0], dtype=torch.int32),
            "temp": torch.tensor([0.0, 1.0, 0.7, 1.3, 0.05])}


def _temp_run(inp):
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    x = inp["logits"]
    apply_temperature(x, inp["idx"], inp["temp"])
    return {"logits": x}


def _temp_ref(inp):
    x = inp["logits"].clone()
    for r in range(x.shape[0]):
        t = float(inp["temp"][inp["idx"][r]])
        if t not in (0.0, 1.0):
            x[r] = x[r] / t
    return {"logits": x}


CASES += [FnCase("v2_temperature", "vllm V2 sampler apply_temperature", "logits / temperature (0 and 1: unchanged), "
                 "bf16 logits", _temp_make, _temp_run, _temp_ref)]


# --- penalties: bincount (atomic_or into a packed prompt mask, atomic_add into output counts) and apply ----------

_PV = 5000


def _pen_state(g, R, prompt_lens, out_lens, maxlen=96):
    all_ids = torch.randint(0, 300, (R, maxlen), generator=g)  # small id range: repeats are common
    return all_ids, torch.tensor(prompt_lens, dtype=torch.int32), \
        torch.tensor([p + o for p, o in zip(prompt_lens, out_lens)], dtype=torch.int32)


def _bincount_make(g):
    all_ids, pl, fl = _pen_state(g, 4, [40, 0, 70, 5], [20, 9, 0, 33])
    return {"all_ids": all_ids, "pl": pl, "fl": fl, "idx": torch.tensor([2, 0, 3, 1], dtype=torch.int32)}


def _bincount_run(inp):
    from vllm.v1.worker.gpu.sample.penalties import bincount
    dev = inp["all_ids"].device
    mask = torch.full((4, (_PV + 31) // 32), 0x5A5A5A5A, dtype=torch.int32, device=dev)  # stale content
    counts = torch.full((4, _PV), 7, dtype=torch.int32, device=dev)
    bincount(inp["idx"], inp["all_ids"], inp["pl"], inp["fl"], mask, counts, int(inp["fl"].max()))
    return {"prompt_mask": mask, "output_counts": counts}


def _bincount_ref(inp):
    R = 4
    mask = torch.zeros(R, (_PV + 31) // 32, dtype=torch.int64)
    counts = torch.zeros(R, _PV, dtype=torch.int64)
    for r in range(R):
        p, f = int(inp["pl"][r]), int(inp["fl"][r])
        for t in inp["all_ids"][r, :p].tolist():
            mask[r, t // 32] |= 1 << (t % 32)
        for t in inp["all_ids"][r, p:f].tolist():
            counts[r, t] += 1
    mask = torch.where(mask >= 2 ** 31, mask - 2 ** 32, mask)  # int32 bit pattern
    return {"prompt_mask": mask.double(), "output_counts": counts.double()}


CASES += [FnCase("v2_penalty_bincount", "vllm V2 penalties.bincount (atomic_or / atomic_add)",
                 "packed prompt-token bit mask and output-token counts per request", _bincount_make, _bincount_run,
                 _bincount_ref)]


def _pen_make(g, spec):
    R = 3
    all_ids, pl, fl = _pen_state(g, R, [30, 12, 50], [10, 25, 0])
    idx = torch.tensor([0, 0, 0, 1, 2], dtype=torch.int32) if spec else torch.tensor([2, 0, 1], dtype=torch.int32)
    local = torch.tensor([0, 1, 2, 0, 0], dtype=torch.int32) if spec else torch.zeros(3, dtype=torch.int32)
    T = idx.numel()
    return {"logits": _logits_rows(g, T, _PV), "idx": idx, "local": local, "all_ids": all_ids, "pl": pl, "fl": fl,
            "input_ids": torch.randint(0, 300, (T,), generator=g),
            "rep": torch.tensor([1.3, 1.0, 0.8]), "freq": torch.tensor([0.4, -0.2, 0.0]),
            "pres": torch.tensor([0.5, 0.0, 1.1])}


def _pen_run(inp):
    from vllm.v1.worker.gpu.sample.penalties import apply_penalties, bincount
    dev = inp["logits"].device
    R = 3
    mask = torch.zeros((R, (_PV + 31) // 32), dtype=torch.int32, device=dev)
    counts = torch.zeros((R, _PV), dtype=torch.int32, device=dev)
    bincount(torch.arange(R, dtype=torch.int32, device=dev), inp["all_ids"], inp["pl"], inp["fl"], mask, counts,
             int(inp["fl"].max()))
    x = inp["logits"]
    apply_penalties(x, inp["idx"], inp["input_ids"], inp["local"], inp["rep"], inp["freq"], inp["pres"], mask,
                    counts)
    return {"logits": x}


def _pen_ref(inp):
    x = inp["logits"].clone()
    T = x.shape[0]
    for t in range(T):
        r = int(inp["idx"][t])
        p, f = int(inp["pl"][r]), int(inp["fl"][r])
        out = inp["all_ids"][r, p:f].tolist()
        pos = int(inp["local"][t])
        out += inp["input_ids"][t - pos + 1: t + 1].tolist()  # draft tokens before this position
        cnt = torch.zeros(_PV, dtype=torch.float64, device=x.device)
        for tok in out:
            cnt[tok] += 1
        prompt = torch.zeros(_PV, dtype=torch.bool, device=x.device)
        prompt[inp["all_ids"][r, :p].long()] = True
        seen = prompt | (cnt > 0)
        rp, fq, pr = float(inp["rep"][r]), float(inp["freq"][r]), float(inp["pres"][r])
        row = x[t]
        if rp != 1.0:
            row = torch.where(seen, torch.where(row > 0, row / rp, row * rp), row)
        row = row - fq * cnt - pr * (cnt > 0).double()
        x[t] = row
    return {"logits": x}


CASES += [
    FnCase("v2_penalties", "vllm V2 apply_penalties (+ bincount)", "repetition (prompt + output, HF rule), "
           "frequency and presence (output) penalties", lambda g: _pen_make(g, False), _pen_run, _pen_ref),
    FnCase("v2_penalties_spec", "vllm V2 apply_penalties with draft positions", "as v2_penalties; draft tokens "
           "before the position count as output", lambda g: _pen_make(g, True), _pen_run, _pen_ref),
]


# --- logprobs ------------------------------------------------------------------------------------------------------

def _logprob_make(g):
    B, V = 5, 7000
    x = _logits_rows(g, B, V, 0.05)
    # a sampled token has a finite logit (with a -inf one the kernel counts its padding lanes, loaded as -inf)
    sampled = torch.stack([torch.nonzero(torch.isfinite(x[b])).flatten()[
        torch.randint(0, int(torch.isfinite(x[b]).sum()), (1,), generator=g)][0] for b in range(B)])
    return {"logits": x, "ids": torch.randint(0, V, (B, 6), generator=g), "sampled": sampled}


def _logprob_run(inp):
    from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel, compute_token_logprobs
    x = inp["logits"]
    lp = compute_token_logprobs(x, inp["ids"])
    ranks = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    _ranks_kernel[(x.shape[0],)](ranks, x, x.stride(0), inp["sampled"], x.shape[1], BLOCK_SIZE=8192)
    return {"logprobs": lp, "ranks": ranks}


def _logprob_ref(inp):
    x = inp["logits"]
    ls = torch.log_softmax(x, -1)
    lp = torch.gather(ls, 1, inp["ids"])
    v = torch.gather(x, 1, inp["sampled"][:, None])
    return {"logprobs": lp, "ranks": (x >= v).sum(1).double()}


CASES += [FnCase("v2_logprobs_ranks", "vllm V2 compute_token_logprobs + _ranks_kernel",
                 "log_softmax at the requested ids; rank = #(logits >= logit of the sampled token)",
                 _logprob_make, _logprob_run, _logprob_ref)]


# --- logit bias / allowed ids / min-tokens stop masking --------------------------------------------------------------

def _bias_make(g):
    T, V, R = 4, 3000, 3
    d = {"logits": _logits_rows(g, T, V), "idx": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
         "pos": torch.tensor([0, 7, 2, 30], dtype=torch.int64),
         "n_allowed": torch.tensor([0, 5, 0], dtype=torch.int32),
         "allowed": torch.randint(0, V, (R, 8), generator=g),
         "n_bias": torch.tensor([3, 0, 4], dtype=torch.int32),
         "bias_ids": torch.stack([torch.randperm(V, generator=g)[:8] for _ in range(R)]),
         "bias": 2 * rn(g, R, 8),
         "min_lens": torch.tensor([5, 0, 10], dtype=torch.int32),
         "n_stop": torch.tensor([2, 0, 3], dtype=torch.int32),
         "restore": torch.zeros(R, dtype=torch.bool),
         "stop_ids": torch.randint(0, V, (R, 8), generator=g)}
    return d


def _bias_run(inp):
    from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias
    x = inp["logits"]
    apply_logit_bias(x, inp["idx"], inp["pos"], inp["n_allowed"], inp["allowed"].int(), inp["n_bias"],
                     inp["bias_ids"].int(), inp["bias"], inp["min_lens"], inp["n_stop"], inp["restore"],
                     inp["stop_ids"].int())
    return {"logits": x}


def _bias_ref(inp):
    x = inp["logits"].clone()
    for t in range(x.shape[0]):
        r = int(inp["idx"][t])
        na = int(inp["n_allowed"][r])
        if na > 0:
            keep = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
            keep[inp["allowed"][r, :na].long()] = True
            x[t] = x[t].masked_fill(~keep, float("-inf"))
        nb = int(inp["n_bias"][r])
        for j in range(nb):
            x[t, inp["bias_ids"][r, j]] += inp["bias"][r, j]
        if int(inp["n_stop"][r]) > 0 and int(inp["pos"][t]) + 1 < int(inp["min_lens"][r]):
            x[t, inp["stop_ids"][r, :int(inp["n_stop"][r])].long()] = float("-inf")
    return {"logits": x}


CASES += [FnCase("v2_logit_bias", "vllm V2 apply_logit_bias", "allowed-token restriction, additive logit bias, "
                 "stop tokens masked before min_tokens", _bias_make, _bias_run, _bias_ref)]


# --- gumbel-max sampling: per-block argmax of logits / T + Gumbel noise from murmur3(seed, pos, token) -------------

def _murmur3_uniform32(seed, pos, keys):
    import numpy as np
    M = np.uint64(0xFFFFFFFF)

    def u32(x):
        return np.asarray(x, dtype=np.uint64) & M

    def rotl(v, s):
        return ((v << np.uint64(s)) | (v >> np.uint64(32 - s))) & M

    def mix(h, k):
        k = (k * np.uint64(0xCC9E2D51)) & M
        k = rotl(k, 15)
        k = (k * np.uint64(0x1B873593)) & M
        h = h ^ k
        h = rotl(h, 13)
        return (h * np.uint64(5) + np.uint64(0xE6546B64)) & M

    def fmix(h):
        h = h ^ (h >> np.uint64(16))
        h = (h * np.uint64(0x85EBCA6B)) & M
        h = h ^ (h >> np.uint64(13))
        h = (h * np.uint64(0xC2B2AE35)) & M
        return h ^ (h >> np.uint64(16))

    seed = int(seed) & ((1 << 64) - 1)
    h = np.zeros(np.shape(keys), dtype=np.uint64)
    h = mix(h, u32(seed & 0xFFFFFFFF))
    h = mix(h, u32((seed >> 32) & 0xFFFFFFFF))
    h = mix(h, u32(int(pos) & 0xFFFFFFFF))
    h = mix(h, u32(keys))
    r = fmix(h ^ np.uint64(16))
    return (r.astype(np.float64) + 0.5) * 2.0 ** -32


def _gumbel_make(g):
    T, V = 4, 2500
    return {"logits": _logits_rows(g, T, V), "idx": torch.tensor([1, 0, 2, 1], dtype=torch.int32),
            "temp": torch.tensor([0.7, 1.0, 0.0]), "seed": torch.randint(0, 2 ** 62, (3,), generator=g),
            "pos": torch.randint(0, 5000, (T,), generator=g)}


def _gumbel_run(inp):
    from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel
    x = inp["logits"]
    T, V = x.shape
    nb = (V + 1023) // 1024
    la = torch.empty(T, nb, dtype=torch.int64, device=x.device)
    lm = torch.empty(T, nb, dtype=torch.float32, device=x.device)
    _gumbel_sample_kernel[(T, nb)](la, la.stride(0), lm, lm.stride(0), None, 0, 0, None, None, 0, x, x.stride(0),
                                   inp["idx"], inp["seed"], inp["pos"], inp["temp"], V, BLOCK_SIZE=1024,
                                   IS_DRAFTING=False, APPLY_TEMPERATURE=True, USE_FP64=False, PER_TOKEN_COL=False)
    return {"block_argmax": la, "block_max": lm}


def _gumbel_ref(inp):
    import numpy as np
    x = inp["logits"]
    T, V = x.shape
    nb = (V + 1023) // 1024
    am = torch.zeros(T, nb, dtype=torch.float64)
    mx = torch.zeros(T, nb, dtype=torch.float64)
    for t in range(T):
        r = int(inp["idx"][t])
        temp = float(inp["temp"][r])
        s = x[t].cpu().numpy().copy()
        if temp != 0.0:
            u = _murmur3_uniform32(int(inp["seed"][r]), int(inp["pos"][t]), np.arange(V))
            s = s / temp - np.log(-np.log1p(-u))
        for b in range(nb):
            blk = s[b * 1024:(b + 1) * 1024]
            am[t, b] = b * 1024 + int(np.argmax(blk))
            mx[t, b] = blk.max()
    return {"block_argmax": am.to(x.device), "block_max": mx.to(x.device)}


CASES += [FnCase("v2_gumbel_sample", "vllm V2 _gumbel_sample_kernel (per-block argmax; the wrapper's final "
                 "argmax over blocks is torch)",
                 "argmax(logits / T + G), G = -log(-log(1 - u)), u = (murmur3(seed, pos, token) + 0.5) 2^-32; "
                 "T = 0: argmax(logits)", _gumbel_make, _gumbel_run, _gumbel_ref)]


# --- speculative decoding: rejection sampler (v1/sample/rejection_sampler.py) ------------------------------------

def _rs_inputs(g, nd, V, with_draft_probs):
    T = sum(nd)
    tp = torch.softmax(1.5 * rn(g, T, V), -1)
    d = {"target_probs": tp, "draft_ids": torch.randint(0, V, (T,), generator=g).int(),
         "bonus": torch.randint(0, V, (len(nd), 1), generator=g), "u": torch.rand(T, generator=g).double(),
         "cu": torch.tensor(nd).cumsum(0).int(), "greedy": torch.tensor([i % 3 == 0 for i in range(len(nd))])}
    # make some drafts likely to be accepted: draft = argmax of the target at half of the positions
    am = tp.argmax(-1).int()
    pick = torch.rand(T, generator=g) < 0.5
    d["draft_ids"] = torch.where(pick, am, d["draft_ids"])
    if with_draft_probs:
        dp = torch.softmax(1.5 * rn(g, T, V), -1)
        d["draft_probs"] = dp
    d["inv_q"] = 1.0 / torch.empty(len(nd), V).exponential_(generator=g)
    return d


def _rs_case(name, nd, V, with_draft_probs):
    maxs = max(nd)

    def make(g):
        return _rs_inputs(g, nd, V, with_draft_probs)

    def run(inp):
        from vllm.v1.sample.rejection_sampler import (_rejection_greedy_sample, _rejection_random_sample,
                                                      _sample_recovered_tokens)
        dev = inp["target_probs"].device
        B = len(nd)
        dp = inp.get("draft_probs")
        rec = torch.empty(sum(nd), dtype=torch.int32, device=dev)
        _sample_recovered_tokens(rec, inp["cu"], inp["draft_ids"], dp, inp["target_probs"], inp["inv_q"].float(),
                                 max_spec_len=maxs)
        out = torch.full((B, maxs + 1), -1, dtype=torch.int32, device=dev)
        am = inp["target_probs"].argmax(-1)
        _rejection_greedy_sample(out, inp["cu"], inp["draft_ids"], am, inp["bonus"], inp["greedy"], maxs, None, None)
        _rejection_random_sample(out, inp["cu"], inp["draft_ids"], dp, inp["target_probs"], inp["bonus"], rec,
                                 inp["u"], inp["greedy"], maxs, None)
        return {"recovered": rec, "output": out}

    def ref(inp):
        tp, dp = inp["target_probs"], inp.get("draft_probs")
        T, V_ = tp.shape
        rec = torch.zeros(T, dtype=torch.float64)
        req = torch.repeat_interleave(torch.arange(len(nd)), torch.tensor(nd))
        inv_q = inp["inv_q"].float().double()  # the kernel reads the fp32 noise
        for t in range(T):
            if dp is not None:
                pr = torch.clamp(tp[t] - dp[t], min=0)
            else:
                pr = tp[t].clone()
                pr[int(inp["draft_ids"][t])] = 0
            rec[t] = float(torch.argmax(pr * inv_q[req[t]].to(pr.device)))
        out = torch.full((len(nd), maxs + 1), -1.0, dtype=torch.float64)
        s = 0
        for b, n in enumerate(nd):
            rejected = False
            for p in range(n):
                t = s + p
                did = int(inp["draft_ids"][t])
                if bool(inp["greedy"][b]):
                    tok = int(torch.argmax(tp[t]))
                    out[b, p] = tok
                    rejected = did != tok
                else:
                    q = 1.0 if dp is None else float(dp[t, did])
                    acc = q > 0 and float(tp[t, did]) / q >= float(inp["u"][t])
                    out[b, p] = did if acc else rec[t]
                    rejected = not acc
                if rejected:
                    break
            if not rejected:
                out[b, n] = int(inp["bonus"][b, 0])
            s += n
        dev = tp.device
        return {"recovered": rec.to(dev), "output": out.to(dev)}

    return FnCase(f"v2_rejection_{name}", "vllm rejection sampler Triton kernels (recovered, greedy, random)",
                  f"accept draft if p/q >= u, else recovered = argmax(max(p - q, 0) / E); greedy: target argmax; "
                  f"bonus when all accepted; drafts {nd}, draft probs {with_draft_probs}", make, run, ref)


CASES += [_rs_case("draft_probs", [3, 1, 4, 2, 5], 2000, True), _rs_case("ngram_no_draft_probs", [3, 1, 4, 2, 5],
                                                                            2000, False)]


# --- Triton top-k / top-p (v1/sample/ops/topk_topp_triton.py) -----------------------------------------------------

def _topkp_spec(x, k, p):
    """Keep: top-k by value (exactly k with continuous inputs), then the smallest prefix (by probability over the
    kept tokens) whose mass reaches p (mass of the strictly higher tokens < p)."""
    out = x.clone()
    for r in range(x.shape[0]):
        row = x[r]
        keep = torch.isfinite(row)
        if k is not None and int(k[r]) < row.numel() and int(k[r]) < int(keep.sum()):
            th = row[keep].topk(int(k[r])).values[-1]
            keep &= row >= th
        if p is not None and float(p[r]) < 1.0:
            pr = torch.zeros_like(row)
            pr[keep] = torch.softmax(row[keep], 0)
            order = torch.argsort(row, descending=True)
            ps = pr[order]
            above = torch.cumsum(ps, 0) - ps
            sel = torch.zeros_like(keep)
            sel[order[above < float(p[r])]] = True
            keep &= sel
        out[r] = row.masked_fill(~keep, float("-inf"))
    return out


def _topkp_case(name, B, V, use_k, use_p, few_finite=0):
    def make(g):
        x = 2 * rn(g, B, V)
        if few_finite:
            m = torch.full((B, V), float("-inf"))
            for r in range(B):
                ids = torch.randperm(V, generator=g)[:few_finite]
                m[r, ids] = x[r, ids]
            x = m
        d = {"logits": x}
        if use_k:
            d["k"] = torch.randint(1, 60, (B,), generator=g).int() if not few_finite else torch.full((B,), 20).int()
        if use_p:
            d["p"] = (0.5 + 0.45 * torch.rand(B, generator=g)) if not few_finite else torch.full((B,), 0.95)
        return d

    def run(inp):
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
        return {"logits": apply_top_k_top_p_triton(inp["logits"], inp.get("k"), inp.get("p"))}

    def ref(inp):
        return {"logits": _topkp_spec(inp["logits"], inp.get("k"), inp.get("p"))}

    return FnCase(f"v2_topkp_{name}", "vllm apply_top_k_top_p_triton (pivot search)",
                  f"exact top-k then top-p masking, B={B} V={V} k={use_k} p={use_p} few_finite={few_finite}",
                  make, run, ref)


CASES += [
    _topkp_case("k", 8, 4096, True, False),
    _topkp_case("p_monolithic", 80, 4096, False, True),
    _topkp_case("p_split", 8, 4096, False, True),
    _topkp_case("kp", 8, 4096, True, True),
    _topkp_case("grammar_few_finite_kp", 8, 4096, True, True, few_finite=2),  # pytorch-free recall of vllm#59785
]


# --- LoRA shrink / expand (lora/ops/triton_ops): pointer tables for several slices ---------------------------------

def _lora_meta(mapping):
    _, sorted_idx = torch.sort(mapping, stable=True)
    ids, counts = torch.unique(mapping, sorted=True, return_counts=True)
    start = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return sorted_idx.int(), counts.int(), start.int(), ids.int()


def _lora_case(name, op, slices, T=24, K=256, R=16, L=3):
    def make(g):
        mapping = torch.randint(-1, L, (T,), generator=g)
        d = {"mapping": mapping}
        if op == "shrink":
            d["x"] = rn(g, T, K).half()
            for s in range(slices):
                d[f"w{s}"] = (0.1 * rn(g, L, R, K)).half()
        else:
            d["y"] = rn(g, slices, T, R).half()
            d["out"] = rn(g, T, K * slices).half()
            for s in range(slices):
                d[f"w{s}"] = (0.1 * rn(g, L, K, R)).half()
        return d

    def run(inp):
        from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
        from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
        from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink
        ws = [inp[f"w{s}"] for s in range(slices)]
        TritonLaunchRecorder.register_implicit(*ws)  # reached through the kernel's pointer table when slices > 1
        sorted_idx, counts, start, ids = (t.to(ws[0].device) for t in _lora_meta(inp["mapping"].cpu()))
        no_lora = torch.tensor([bool((inp["mapping"] == -1).all())])
        nact = torch.tensor([ids.numel()])
        if op == "shrink":
            out = torch.empty(slices, T, R, dtype=torch.float32, device=ws[0].device)
            _lora_shrink(inp["x"], ws, out, inp["mapping"], sorted_idx, counts, start, ids, no_lora, nact, 0.5)
            return {"y": out}
        out = inp["out"]
        _lora_expand(inp["y"], ws, out, inp["mapping"], sorted_idx, counts, start, ids, no_lora, nact,
                     offset_start=0, add_inputs=True)
        return {"out": out}

    def ref(inp):
        m = inp["mapping"]
        if op == "shrink":
            y = torch.zeros(slices, T, R, dtype=torch.float64, device=m.device)
            for s in range(slices):
                for t in range(T):
                    if int(m[t]) >= 0:
                        y[s, t] = 0.5 * inp["w{}".format(s)][int(m[t])] @ inp["x"][t]
            return {"y": y}
        out = inp["out"].clone()
        for s in range(slices):
            for t in range(T):
                if int(m[t]) >= 0:
                    out[t, s * K:(s + 1) * K] += inp[f"w{s}"][int(m[t])] @ inp["y"][s, t]
        return {"out": out}

    return FnCase(f"v2_lora_{op}_{slices}slice", f"vllm _lora_{op} (Triton, multi-LoRA)",
                  f"{'y_s[t] = 0.5 * A_s[lora(t)] x[t]' if op == 'shrink' else 'out[t, slice s] += B_s[lora(t)] y_s[t]'}"
                  f" for tokens with a LoRA (mapping -1: untouched); {slices} slice(s), rank {R}", make, run, ref)


CASES += [_lora_case("s", "shrink", 1), _lora_case("s3", "shrink", 3), _lora_case("e", "expand", 1),
          _lora_case("e3", "expand", 3)]


# --- fused MoE (bf16 path of fused_moe_kernel) ------------------------------------------------------------------------

def _moe_align(topk_ids, block, E):
    flat = topk_ids.flatten()
    sorted_ids, expert_ids = [], []
    for e in range(E):
        toks = torch.nonzero(flat == e).flatten().tolist()
        if not toks:
            continue
        pad = (-len(toks)) % block
        toks += [flat.numel()] * pad
        sorted_ids += toks
        expert_ids += [e] * (len(toks) // block)
    return (torch.tensor(sorted_ids, dtype=torch.int32), torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int32))


def _moe_case(name, T, K, N, E, topk, routed):
    BM = 16

    def make(g):
        logits = rn(g, T, E)
        w, ids = torch.topk(torch.softmax(logits, -1), topk, -1)
        s, e, n = _moe_align(ids, BM, E)
        return {"a": rn(g, T, K).half(), "b": (0.1 * rn(g, E, N, K)).half(), "w": w.float(), "ids": ids,
                "sorted": s, "experts": e, "ntpp": n}

    def run(inp):
        import triton.language as tl
        from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
        c = torch.zeros(T, topk, N, dtype=torch.float16, device=inp["a"].device)
        cfg = {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "num_warps": 4,
               "num_stages": 2}
        invoke_fused_moe_triton_kernel(inp["a"], inp["b"], c, None, None, inp["w"], inp["sorted"], inp["experts"],
                                       inp["ntpp"], routed, topk, cfg, tl.float16, False, False, False, False, False)
        return {"c": c}

    def ref(inp):
        c = torch.zeros(T, topk, N, dtype=torch.float64, device=inp["a"].device)
        for t in range(T):
            for j in range(topk):
                e = int(inp["ids"][t, j])
                c[t, j] = inp["b"][e] @ inp["a"][t] * (inp["w"][t, j] if routed else 1.0)
        return {"c": c}

    return FnCase(f"v2_fused_moe_{name}", "vllm fused_moe_kernel (fp16, invoke_fused_moe_triton_kernel)",
                  f"C[t, j] = (w[t, j] if routed) * B[e(t, j)] A[t]; T={T} K={K} N={N} E={E} topk={topk}; token "
                  f"grouping computed as moe_align_block_size does (block {BM}, padding id T*topk)", make, run, ref)


CASES += [_moe_case("plain", 37, 128, 96, 8, 2, False), _moe_case("routed", 37, 128, 96, 8, 2, True)]


# --- batch-invariant kernels (model_executor/determinism/batch_invariant.py) --------------------------------------

def _bi_mm_make(g):
    return {"a": rn(g, 70, 96).half(), "b": rn(g, 96, 130).half(), "bias": rn(g, 130).half()}


def _bi_mm_run(inp):
    from vllm.model_executor.determinism.batch_invariant import matmul_persistent
    return {"c": matmul_persistent(inp["a"], inp["b"], inp["bias"])}


def _bi_lsm_run(inp):
    from vllm.model_executor.determinism.batch_invariant import log_softmax, mean_dim, rms_norm_batch_invariant
    return {"log_softmax": log_softmax(inp["x"], -1), "mean": mean_dim(inp["x"], 1),
            "rms": rms_norm_batch_invariant(inp["x"], inp["w"], 1e-6)}


CASES += [
    FnCase("v2_bi_matmul_persistent", "vllm batch_invariant.matmul_persistent", "a @ b + bias (fp16)", _bi_mm_make,
           _bi_mm_run, lambda inp: {"c": inp["a"] @ inp["b"] + inp["bias"]}),
    FnCase("v2_bi_log_softmax_mean_rms", "vllm batch_invariant log_softmax / mean_dim / rms_norm",
           "log_softmax(x, -1), mean(x, 1), x * rsqrt(mean(x^2) + eps) * w", lambda g: {"x": 2 * rn(g, 9, 3000),
                                                                                   "w": rn(g, 3000)},
           _bi_lsm_run, lambda inp: {"log_softmax": torch.log_softmax(inp["x"], -1), "mean": inp["x"].mean(1),
                                     "rms": inp["x"] * torch.rsqrt((inp["x"] ** 2).mean(-1, keepdim=True) + 1e-6)
                                     * inp["w"]}),
]
