"""flash-linear-attention (fla-org, main) gated delta rule kernels for tool_spec_check.

Implementation: fla.ops.gated_delta_rule.chunk_gated_delta_rule (training path; chunked WY form, several Triton
launches) and fused_recurrent_gated_delta_rule (inference path).  Specification: the documented recurrence
(fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule, generalised to the documented options),
per value head hv with query/key head hv // (HV / H), in float64:

    g   <- -exp(A_log) * softplus(g + dt_bias)            (use_gate_in_kernel)
    beta <- sigmoid(beta) * (2 if allow_neg_eigval else 1) (use_beta_sigmoid_in_kernel)
    q, k <- x / sqrt(sum x^2 + 1e-6)                       (use_qk_l2norm_in_kernel)
    h <- h * exp(g_t);  u = beta_t (v_t - h^T k_t);  h <- h + k_t u^T;  o_t = (scale q_t)^T h

with h [K, V] (or [V, K] for state_v_first in the initial / final state), per-sequence states with cu_seqlens,
scale = K^-0.5.  Declared bound 2^-40 * max|f|.  Environment: ka_main + PYTHONPATH=.cache/pylibs/shim and the FLA
source tree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_cases_tridao import FnCase, rn


def gdr_ref(q, k, v, g, beta, h0=None, cu=None, l2=False, beta_sig=False, neg=False, gate=False, A_log=None,
            dt_bias=None, v_first=False):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if gate:
        g = -torch.exp(A_log) * F.softplus(g + dt_bias)
    if beta_sig:
        beta = torch.sigmoid(beta) * (2.0 if neg else 1.0)
    if l2:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    rep = HV // H
    q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
    q = q * K ** -0.5
    segs = [(b, 0, T) for b in range(B)] if cu is None else [(0, int(cu[i]), int(cu[i + 1])) for i in range(len(cu) - 1)]
    o = torch.zeros(B, T, HV, V, dtype=q.dtype, device=q.device)
    finals = []
    for n, (b, s0, s1) in enumerate(segs):
        h = torch.zeros(HV, K, V, dtype=q.dtype, device=q.device)
        if h0 is not None:
            h = h0[n].transpose(-1, -2) if v_first else h0[n]
        outs = []
        for t in range(s0, s1):
            h = h * torch.exp(g[b, t])[:, None, None]
            u = beta[b, t][:, None] * (v[b, t] - torch.einsum("hk,hkv->hv", k[b, t], h))
            h = h + k[b, t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", q[b, t], h))
        o[b, s0:s1] = torch.stack(outs)
        finals.append(h.transpose(-1, -2) if v_first else h)
    return o, torch.stack(finals)


def _gdr_case(name, path="chunk", B=2, T=128, H=2, HV=2, K=32, V=32, l2=False, beta_sig=False, neg=False,
              gate=False, h0=False, final=False, v_first=False, varlen=None, backward=False):
    def make(gen):
        b = 1 if varlen else B
        t = varlen[-1] if varlen else T
        d = {"q": rn(gen, b, t, H, K), "k": rn(gen, b, t, H, K), "v": rn(gen, b, t, HV, V)}
        if not l2:
            d["k"] = F.normalize(d["k"], dim=-1)
        if gate:
            d["g"] = rn(gen, b, t, HV)
            d["A_log"] = 0.3 * rn(gen, HV)
            d["dt_bias"] = 0.5 * rn(gen, HV)
        else:
            d["g"] = F.logsigmoid(rn(gen, b, t, HV) + 3.0)
        d["beta"] = rn(gen, b, t, HV) if beta_sig else torch.rand(b, t, HV, generator=gen)
        if h0:
            n = len(varlen) - 1 if varlen else B
            d["h0"] = 0.5 * rn(gen, n, HV, V, K) if v_first else 0.5 * rn(gen, n, HV, K, V)
        if varlen:
            d["cu"] = torch.tensor(varlen, dtype=torch.long)
        return d

    def run(inp):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        kw = dict(initial_state=inp.get("h0"), output_final_state=final, use_qk_l2norm_in_kernel=l2,
                  use_beta_sigmoid_in_kernel=beta_sig, allow_neg_eigval=neg, state_v_first=v_first,
                  cu_seqlens=inp.get("cu"))
        if gate:
            kw.update(use_gate_in_kernel=True, A_log=inp["A_log"], dt_bias=inp["dt_bias"])
        if path == "chunk":
            if varlen:
                kw["cu_seqlens_cpu"] = inp["cu"].cpu()
            o, ht = chunk_gated_delta_rule(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], **kw)
        else:
            o, ht = fused_recurrent_gated_delta_rule(inp["q"], inp["k"], inp["v"], g=inp["g"], beta=inp["beta"], **kw)
        return {"o": o, "final_state": ht} if final and not backward else {"o": o}

    def ref(inp):
        o, ht = gdr_ref(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp.get("h0"), inp.get("cu"), l2,
                        beta_sig, neg, gate, inp.get("A_log"), inp.get("dt_bias"), v_first)
        return {"o": o, "final_state": ht} if final and not backward else {"o": o}

    grads = ()
    if backward:
        grads = ["q", "k", "v", "g", "beta"] + (["A_log", "dt_bias"] if gate else []) + (["h0"] if h0 else [])
    impl = ("fla.ops.gated_delta_rule.chunk_gated_delta_rule" if path == "chunk"
            else "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule") + " (main)"
    return FnCase(f"fla_gdr_{path}_{name}" + ("_bwd" if backward else ""), impl,
                  f"gated delta rule recurrence (float64): B={B} T={T} H={H} HV={HV} K={K} V={V} l2={l2} "
                  f"beta_sigmoid={beta_sig} neg_eig={neg} gate_in_kernel={gate} h0={h0} final={final} "
                  f"state_v_first={v_first} varlen={varlen}", make, run, ref, grad_of=grads, loss_of="o")


_VL = [0, 50, 128, 200]
CASES = []
for _path in ("chunk", "recurrent"):
    CASES += [
        _gdr_case("basic", _path),
        _gdr_case("gva", _path, H=2, HV=4),
        _gdr_case("l2_sigmoid", _path, l2=True, beta_sig=True),
        _gdr_case("neg_eig", _path, beta_sig=True, neg=True),
        _gdr_case("gate", _path, gate=True),
        _gdr_case("h0_final", _path, h0=True, final=True),
        _gdr_case("vfirst", _path, h0=True, final=True, v_first=True),
        _gdr_case("varlen", _path, varlen=_VL, h0=True, final=True),
        _gdr_case("t100", _path, T=100),
    ]
CASES += [
    _gdr_case("basic", "chunk", backward=True),
    _gdr_case("l2_sigmoid", "chunk", l2=True, beta_sig=True, backward=True),
    _gdr_case("gate", "chunk", gate=True, backward=True),
    _gdr_case("h0", "chunk", h0=True, backward=True),
    _gdr_case("varlen", "chunk", varlen=_VL, backward=True),
    _gdr_case("t100", "chunk", T=100, backward=True),
]
