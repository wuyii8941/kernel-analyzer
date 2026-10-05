"""vLLM main (2026-10-05, 51eeb0c) Triton kernels for tool_spec_check.

Imported from .cache/pylibs/vllm_shim (scripts/build_vllm_shim.py: the kernel modules verbatim, the rest of vLLM
stubbed).  Specifications are the documented semantics, evaluated in float64 eager PyTorch on the same inputs
(declared bound 2^-40 * max|f|):

* unified_attention (v1/attention/ops/triton_unified_attention.py), paged KV cache: per sequence, query i of a chunk
  of length L at absolute position kv_len - L + i attends to key j when (causal: j <= pos) and
  (window (wl, wr) in the FlashAttention convention: pos - j <= wl, and j - pos <= wr when not causal);
  scores q.k * scale, softcap c*tanh(s/c), ALiBi slope*(j - pos), sinks as an extra logit in the denominator only.

Environment: ka_main + PYTHONPATH=.cache/pylibs/vllm_shim.
"""

from __future__ import annotations

import torch

from tool_spec_check import Case, f64_point_spec


def P(t):
    return f64_point_spec(t.detach().cpu().numpy())


CASES = []

# ---------------------------------------------------------------------------------------------------------------
# unified_attention
# ---------------------------------------------------------------------------------------------------------------


def _attn_ref(q, kd, vd, qlens, kvlens, causal, window, scale, softcap=0.0, sinks=None, alibi=None):
    """q (T, H, D); kd/vd (B, Kmax, Hk, D) dense per-sequence keys/values; causal: bool or list per sequence."""
    outs, qs = [], 0
    H, Hk = q.shape[1], kd.shape[2]
    for b, (ql, kl) in enumerate(zip(qlens, kvlens)):
        qq = q[qs:qs + ql].double()
        qs += ql
        kk = kd[b][:kl].double().repeat_interleave(H // Hk, 1)
        vv = vd[b][:kl].double().repeat_interleave(H // Hk, 1)
        s = torch.einsum("qhd,khd->hqk", qq, kk) * scale
        if softcap > 0:
            s = softcap * torch.tanh(s / softcap)
        qpos = torch.arange(kl - ql, kl, device=q.device)[:, None]
        kpos = torch.arange(kl, device=q.device)[None, :]
        if alibi is not None:
            s = s + alibi.double()[:, None, None] * (kpos - qpos)[None]
        c = causal[b] if isinstance(causal, (list, tuple)) else causal
        m = torch.ones(ql, kl, dtype=torch.bool, device=q.device)
        if c:
            m &= kpos <= qpos
        if window[0] >= 0:
            m &= (qpos - kpos) <= window[0]
        if window[1] >= 0 and not c:
            m &= (kpos - qpos) <= window[1]
        s = s.masked_fill(~m, float("-inf"))
        if sinks is not None:
            sk = sinks.double()[:, None, None].expand(H, ql, 1)
            p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :-1]
        else:
            p = torch.softmax(s, -1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vv))
    return torch.cat(outs)


class UACase(Case):
    implementation = "vllm.v1.attention.ops.triton_unified_attention.unified_attention (paged KV cache)"
    spec_bound = "float64 attention on the dense keys/values, declared bound 2^-40 * max|f|"

    def __init__(self, name, qlens, kvlens, H, Hk, D, causal=True, window=(-1, -1), softcap=0.0, sinks=False,
                 alibi=False, block_size=16, dtype=torch.float16, decode_3d=False, null_block_nan=False):
        self.name = name
        self.cfg = dict(qlens=qlens, kvlens=kvlens, H=H, Hk=Hk, D=D, causal=causal, window=window, softcap=softcap,
                        sinks=sinks, alibi=alibi, block_size=block_size, dtype=dtype, decode_3d=decode_3d,
                        null_block_nan=null_block_nan)
        self.specification = (f"softmax attention, causal={causal}, window={window}, softcap={softcap}, "
                              f"sinks={sinks}, alibi={alibi}, GQA {H}/{Hk}, D={D}, query lens {qlens}, kv lens {kvlens}")

    def setup(self):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        self._fn = unified_attention
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        c = self.cfg
        g = torch.Generator(device="cpu").manual_seed(seed)
        B, bs = len(c["qlens"]), c["block_size"]
        nblk = (max(c["kvlens"]) + bs - 1) // bs
        H, Hk, D, dt = c["H"], c["Hk"], c["D"], c["dtype"]
        q = torch.randn(sum(c["qlens"]), H, D, generator=g).to(dt)
        kd = torch.randn(B, nblk * bs, Hk, D, generator=g).to(dt)
        vd = torch.randn(B, nblk * bs, Hk, D, generator=g).to(dt)
        num_blocks = B * nblk + 2  # block 0 stays unassigned ("null block")
        perm = torch.randperm(num_blocks - 1, generator=g)[: B * nblk] + 1
        bt = perm.view(B, nblk).int()
        fill = float("nan") if c["null_block_nan"] else 0.0
        kc = torch.full((num_blocks, bs, Hk, D), fill).to(dt)
        vc = torch.full((num_blocks, bs, Hk, D), fill).to(dt)
        for b in range(B):
            kl = c["kvlens"][b]
            kc[bt[b].long()] = kd[b].view(nblk, bs, Hk, D)
            vc[bt[b].long()] = vd[b].view(nblk, bs, Hk, D)
            if c["null_block_nan"]:  # slots past kv_len hold stale data
                flat_k = kc[bt[b].long()].view(-1, Hk, D)
                flat_k[kl:] = float("nan")
                kc[bt[b].long()] = flat_k.view(nblk, bs, Hk, D)
                flat_v = vc[bt[b].long()].view(-1, Hk, D)
                flat_v[kl:] = float("nan")
                vc[bt[b].long()] = flat_v.view(nblk, bs, Hk, D)
        inp = dict(q=q, kd=kd, vd=vd, kc=kc, vc=vc, bt=bt,
                   cu=torch.tensor([0] + torch.tensor(c["qlens"]).cumsum(0).tolist(), dtype=torch.int32),
                   sk=torch.tensor(c["kvlens"], dtype=torch.int32))
        if c["sinks"]:
            inp["sinks"] = torch.randn(H, generator=g).float() * 2
        if c["alibi"]:
            inp["alibi"] = (2.0 ** (-8.0 * torch.arange(1, H + 1) / H)).float()
        return {k: v.cuda() for k, v in inp.items()}

    def launch(self, inp):
        c = self.cfg
        q = inp["q"]
        out = torch.empty_like(q)
        causal = c["causal"]
        if isinstance(causal, (list, tuple)):
            causal = torch.tensor(causal, dtype=torch.bool, device=q.device)
        kw = {}
        if c["decode_3d"]:
            segs = 4
            T, H, Dp = q.shape[0], q.shape[1], 1 << (q.shape[2] - 1).bit_length()
            kw = dict(seq_threshold_3D=64, num_par_softmax_segments=segs,
                      softmax_segm_output=torch.empty(T, H, segs, Dp, device=q.device, dtype=torch.float32),
                      softmax_segm_max=torch.empty(T, H, segs, device=q.device, dtype=torch.float32),
                      softmax_segm_expsum=torch.empty(T, H, segs, device=q.device, dtype=torch.float32))
        self._fn(q, inp["kc"], inp["vc"], out, inp["cu"], max(c["qlens"]), inp["sk"], max(c["kvlens"]),
                 q.shape[-1] ** -0.5, causal, c["window"], inp["bt"], c["softcap"], None, None, None,
                 sinks=inp.get("sinks"), alibi_slopes=inp.get("alibi"), **kw)
        return {"out": out}

    def spec(self, inp):
        c = self.cfg
        ref = _attn_ref(inp["q"], inp["kd"], inp["vd"], c["qlens"], c["kvlens"], c["causal"], c["window"],
                        inp["q"].shape[-1] ** -0.5, c["softcap"], inp.get("sinks"), inp.get("alibi"))
        return {"out": P(ref)}


_Q, _K = [48, 21], [80, 21]   # sequence 0: 32 cached tokens + a 48-token chunk; sequence 1: a fresh 21-token prompt
CASES += [
    UACase("ua_causal", _Q, _K, 8, 8, 64),
    UACase("ua_causal_gqa4_d80", _Q, _K, 8, 2, 80),
    UACase("ua_causal_sw24", _Q, _K, 8, 2, 64, window=(23, 0)),
    UACase("ua_causal_sw24_nan_stale", _Q, _K, 8, 2, 64, window=(23, 0), null_block_nan=True),
    UACase("ua_causal_softcap", _Q, _K, 8, 2, 64, softcap=5.0),
    UACase("ua_causal_sinks", _Q, _K, 8, 2, 64, sinks=True),
    UACase("ua_causal_sinks_sw24", _Q, _K, 8, 2, 64, sinks=True, window=(23, 0)),
    UACase("ua_causal_alibi", _Q, _K, 8, 8, 64, alibi=True),
    UACase("ua_bidir", [48, 21], [48, 21], 8, 8, 64, causal=False),
    UACase("ua_bidir_sw8_mha", [48, 21], [48, 21], 8, 8, 64, causal=False, window=(7, 7)),
    UACase("ua_bidir_sw24_gqa4", [48, 21], [48, 21], 8, 2, 64, causal=False, window=(23, 23)),
    UACase("ua_bidir_sw8_qpkv16", [48, 21], [48, 21], 16, 1, 64, causal=False, window=(7, 7)),
    UACase("ua_perseq_causal_sw8", [48, 21], [48, 21], 8, 2, 64, causal=[False, True], window=(7, 7)),
    UACase("ua_decode_3d", [1, 1, 1], [300, 77, 129], 8, 2, 64, decode_3d=True),
    UACase("ua_decode_3d_sinks_sw", [1, 1, 1], [300, 77, 129], 8, 2, 64, decode_3d=True, sinks=True,
           window=(63, 0)),
]
