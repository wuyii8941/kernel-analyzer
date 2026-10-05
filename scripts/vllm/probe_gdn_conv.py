"""Differential probes (float64 specs) for vLLM's vendored FLA gated delta rule, causal_conv1d and merge_attn_states.
Screening only: a finding is then checked with tool_spec_check."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool_spec_cases_fla import gdr_ref  # noqa: E402

from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, fused_sigmoid_gating_delta_rule_update)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: E402
from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states  # noqa: E402

dev = "cuda"


def rel(a, b):
    a, b = a.double(), b.double()
    f = torch.isfinite(b)
    if not torch.equal(torch.isfinite(a), f):
        return float("nan")
    return ((a[f] - b[f]).norm() / b[f].norm().clamp_min(1e-30)).item()


def gdn_chunk(T_list, H, HV, K, V, l2, h0, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    T = sum(T_list)
    cu = torch.tensor([0] + torch.tensor(T_list).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    q = torch.randn(1, T, H, K, device=dev).to(dtype)
    k = torch.randn(1, T, H, K, device=dev)
    k = (k if l2 else F.normalize(k, dim=-1)).to(dtype)
    v = torch.randn(1, T, HV, V, device=dev).to(dtype)
    g = F.logsigmoid(torch.randn(1, T, HV, device=dev) + 3).float()
    beta = torch.rand(1, T, HV, device=dev).to(dtype)
    N = len(T_list)
    st = (0.5 * torch.randn(N, HV, V, K, device=dev)).float() if h0 else None
    o, ht = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=st, output_final_state=True, cu_seqlens=cu,
                                   use_qk_l2norm_in_kernel=l2)
    ro, rh = gdr_ref(q.double(), k.double(), v.double(), g.double(), beta.double(),
                     st.double() if h0 else None, cu.cpu(), l2=l2, v_first=True)
    return rel(o, ro), rel(ht, rh)


def gdn_recurrent_spec(qlens, num_acc, H, HV, K, V, sigmoid_gating, dtype=torch.bfloat16, seed=0):
    """Spec decoding: sequence n has qlens[n] tokens; its state slots ssm_state_indices[n, :], initial state read
    from slot index num_acc[n]-1, the state after token t written to slot t."""
    torch.manual_seed(seed)
    N, T = len(qlens), sum(qlens)
    maxq = max(qlens)
    cu = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    q = torch.randn(1, T, H, K, device=dev).to(dtype)
    k = torch.randn(1, T, H, K, device=dev).to(dtype)
    v = torch.randn(1, T, HV, V, device=dev).to(dtype)
    nslots = 1 + N * maxq + 3
    perm = torch.randperm(nslots - 1)[: N * maxq] + 1
    idx = perm.view(N, maxq).int().to(dev)
    pool = (0.5 * torch.randn(nslots, HV, V, K, device=dev)).float()
    pool0 = pool.clone()
    acc = torch.tensor(num_acc, device=dev, dtype=torch.int32)
    if sigmoid_gating:
        A_log = 0.3 * torch.randn(HV, device=dev)
        a = torch.randn(1, T, HV, device=dev).to(dtype)
        b = torch.randn(1, T, HV, device=dev).to(dtype)
        dt_bias = 0.5 * torch.randn(HV, device=dev)
        o, _ = fused_sigmoid_gating_delta_rule_update(A_log, a, b, dt_bias, q, k, v, initial_state=pool,
                                                     inplace_final_state=True, cu_seqlens=cu,
                                                     ssm_state_indices=idx, num_accepted_tokens=acc,
                                                     use_qk_l2norm_in_kernel=True)
        g = -torch.exp(A_log.double()) * F.softplus(a.double() + dt_bias.double())
        beta = torch.sigmoid(b.double())
        l2 = True
    else:
        g = F.logsigmoid(torch.randn(1, T, HV, device=dev) + 3).float()
        beta = torch.rand(1, T, HV, device=dev).to(dtype)
        o, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=pool, inplace_final_state=True,
                                                cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=acc,
                                                use_qk_l2norm_in_kernel=True)
        g, beta, l2 = g.double(), beta.double(), True
    errs_o, errs_s = [], []
    for n in range(N):
        s0, s1 = int(cu[n]), int(cu[n + 1])
        h0 = pool0[idx[n, num_acc[n] - 1].long()].double()[None]
        # reference per token to get the per-token states
        hs = []
        for t in range(s0, s1):
            ro_t, rh_t = gdr_ref(q[:, s0:t + 1].double(), k[:, s0:t + 1].double(), v[:, s0:t + 1].double(),
                                 g[:, s0:t + 1].double(), beta[:, s0:t + 1].double(), h0, None, l2=l2, v_first=True)
            hs.append(rh_t[0])
        ro, _ = gdr_ref(q[:, s0:s1].double(), k[:, s0:s1].double(), v[:, s0:s1].double(), g[:, s0:s1].double(),
                        beta[:, s0:s1].double(), h0, None, l2=l2, v_first=True)
        errs_o.append(rel(o[:, s0:s1], ro))
        errs_s.append(max(rel(pool[idx[n, t].long()], hs[t]) for t in range(s1 - s0)))
    return errs_o, errs_s


def conv_fn(qlens, has_init, W, dim=320, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    T = sum(qlens)
    x = torch.randn(T, dim, device=dev).to(dtype).t()  # (dim, T) channel-last
    w = torch.randn(dim, W, device=dev).to(dtype)
    bias = torch.randn(dim, device=dev).to(dtype)
    N = len(qlens)
    lines = N + 3
    cs = torch.randn(lines, W - 1, dim, device=dev).to(dtype).transpose(1, 2)  # (lines, dim, W-1), dim stride 1
    cs0 = cs.clone()
    ci = (torch.randperm(lines - 1)[:N] + 1).int().to(dev)
    hi = torch.tensor(has_init, device=dev, dtype=torch.bool)
    cu = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    out = causal_conv1d_fn(x, w, bias, conv_states=cs, query_start_loc=cu, cache_indices=ci, has_initial_state=hi,
                           activation="silu")
    eo, es = [], []
    for n in range(N):
        s0, s1 = int(cu[n]), int(cu[n + 1])
        init = cs0[ci[n].long()].double() if has_init[n] else torch.zeros(dim, W - 1, device=dev, dtype=torch.float64)
        full = torch.cat([init, x[:, s0:s1].double()], 1)
        y = F.conv1d(full[None], w.double()[:, None, :], bias.double(), groups=dim)[0]
        y = F.silu(y)
        eo.append(rel(out[:, s0:s1], y))
        es.append(rel(cs[ci[n].long()], full[:, -(W - 1):]))
    return eo, es


def conv_update_spec(qlens, num_acc, W, dim=320, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    N, T, maxq = len(qlens), sum(qlens), max(qlens)
    x = torch.randn(T, dim, device=dev).to(dtype)
    w = torch.randn(dim, W, device=dev).to(dtype)
    bias = torch.randn(dim, device=dev).to(dtype)
    L = W - 1 + maxq - 1
    lines = N + 3
    cs = torch.randn(lines, L, dim, device=dev).to(dtype).transpose(1, 2)
    cs0 = cs.clone()
    ci = (torch.randperm(lines - 1)[:N] + 1).int().to(dev)
    cu = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    acc = torch.tensor(num_acc, device=dev, dtype=torch.int32)
    out = causal_conv1d_update(x.clone(), cs, w, bias, "silu", conv_state_indices=ci, num_accepted_tokens=acc,
                               query_start_loc=cu, max_query_len=maxq)
    eo, es = [], []
    for n in range(N):
        s0, s1 = int(cu[n]), int(cu[n + 1])
        S = cs0[ci[n].long()].double()
        a = num_acc[n]
        hist = S[:, a - 1: a - 1 + W - 1]
        full = torch.cat([hist, x[s0:s1].double().t()], 1)
        y = F.silu(F.conv1d(full[None], w.double()[:, None, :], bias.double(), groups=dim)[0])
        eo.append(rel(out[s0:s1].t(), y))
        new = torch.cat([S[:, a: a + W - 2], x[s0:s1].double().t()], 1)
        es.append(rel(cs[ci[n].long()][:, : new.shape[1]], new))
    return eo, es


def merge(n_tok, H, D, special, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    pa = torch.randn(n_tok, H, D, device=dev).to(dtype)
    sa = torch.randn(n_tok, H, D, device=dev).to(dtype)
    pl = torch.randn(H, n_tok, device=dev) * 3
    sl = torch.randn(H, n_tok, device=dev) * 3
    if special == "prefix_empty":
        pl[:, ::3] = float("-inf")
    if special == "suffix_empty":
        sl[:, ::3] = float("-inf")
    if special == "both_empty":
        pl[:, ::3] = float("-inf")
        sl[:, ::3] = float("-inf")
    if special == "pinf":
        pl[:, ::3] = float("inf")
    out = torch.empty_like(pa)
    out_lse = torch.empty_like(pl)
    merge_attn_states(out, pa, pl, sa, sl, out_lse)
    pl64, sl64 = pl.double(), sl.double()
    m = torch.maximum(pl64, sl64)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    wp, ws = torch.exp(pl64 - m), torch.exp(sl64 - m)
    lse = m + torch.log(wp + ws)
    ref = (pa.double() * (wp / (wp + ws)).t()[..., None] + sa.double() * (ws / (wp + ws)).t()[..., None])
    return rel(out, ref), rel(out_lse, lse)


if __name__ == "__main__" and False:
    print("== vendored chunk_gated_delta_rule (o rel, final state rel), bf16 noise ~1e-2")
    for T_list in ([64], [100, 37], [1, 200, 63], [64, 64, 1]):
        for H, HV in ((2, 2), (2, 4)):
            print(T_list, H, HV, "l2" , gdn_chunk(T_list, H, HV, 64, 64, True, True),
                  "nol2", gdn_chunk(T_list, H, HV, 64, 64, False, True))
    print("== fused_recurrent spec decoding (per-seq o rel, max per-token-state rel)")
    for qlens, acc in (([1, 1, 1], [1, 1, 1]), ([3, 3, 2], [1, 2, 3]), ([4, 1, 4], [4, 1, 2])):
        print(qlens, acc, "recurrent", gdn_recurrent_spec(qlens, acc, 2, 4, 64, 64, False),
              "sigmoid", gdn_recurrent_spec(qlens, acc, 2, 4, 64, 64, True))
if __name__ == "__main__":
    print("== causal_conv1d_fn (per-seq out rel, state rel)")
    for W in (2, 3, 4):
        for qlens, hi in (([20, 1, 2, 9], [True, True, True, False]), ([2, 1, 3], [False, True, True]),
                          ([17, 300], [True, False])):
            print(W, qlens, hi, conv_fn(qlens, hi, W))
    print("== causal_conv1d_update spec decoding")
    for W in (2, 3, 4):
        for qlens, acc in (([1, 1, 1], [1, 1, 1]), ([3, 3, 2], [1, 2, 3]), ([4, 1, 4], [4, 1, 2])):
            print(W, qlens, acc, conv_update_spec(qlens, acc, W))
    print("== merge_attn_states (out rel, lse rel)")
    for sp in ("none", "prefix_empty", "suffix_empty", "both_empty", "pinf"):
        print(sp, merge(37, 8, 128, sp), merge(37, 8, 80, sp))
