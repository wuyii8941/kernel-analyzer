"""Quick differential probe of vLLM unified_attention against float64 attention (paged KV cache)."""
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def ref_attn(q, kd, vd, qlens, kvlens, causal, window, scale, softcap=0.0):
    outs, qs = [], 0
    H, Hk = q.shape[1], kd.shape[2]
    for b, (ql, kl) in enumerate(zip(qlens, kvlens)):
        qq = q[qs:qs + ql].double(); qs += ql
        kk = kd[b][:kl].double().repeat_interleave(H // Hk, 1)
        vv = vd[b][:kl].double().repeat_interleave(H // Hk, 1)
        s = torch.einsum("qhd,khd->hqk", qq, kk) * scale
        if softcap > 0:
            s = softcap * torch.tanh(s / softcap)
        qpos = torch.arange(kl - ql, kl, device=q.device)[:, None]
        kpos = torch.arange(kl, device=q.device)[None, :]
        m = torch.ones(ql, kl, dtype=torch.bool, device=q.device)
        if causal:
            m &= kpos <= qpos
        if window[0] >= 0:
            m &= (qpos - kpos) <= window[0]
        if window[1] >= 0 and not causal:
            m &= (kpos - qpos) <= window[1]
        p = torch.softmax(s.masked_fill(~m, float("-inf")), -1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vv))
    return torch.cat(outs)


def make(qlens, kvlens, H, Hk, D, block_size=16, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B, maxk = len(qlens), max(kvlens)
    nblk = (maxk + block_size - 1) // block_size
    q = torch.randn(sum(qlens), H, D, device="cuda", generator=g).to(dtype)
    kd = torch.randn(B, nblk * block_size, Hk, D, device="cuda", generator=g).to(dtype)
    vd = torch.randn(B, nblk * block_size, Hk, D, device="cuda", generator=g).to(dtype)
    num_blocks = B * nblk + 3
    perm = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(seed))[: B * nblk]
    bt = perm.view(B, nblk).int().cuda()
    kc = torch.randn(num_blocks, block_size, Hk, D, device="cuda", generator=g).to(dtype)
    vc = torch.randn(num_blocks, block_size, Hk, D, device="cuda", generator=g).to(dtype)
    for b in range(B):
        kc[bt[b].long()] = kd[b].view(nblk, block_size, Hk, D)
        vc[bt[b].long()] = vd[b].view(nblk, block_size, Hk, D)
    cu = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    sk = torch.tensor(kvlens, device="cuda", dtype=torch.int32)
    return dict(q=q, kd=kd, vd=vd, kc=kc, vc=vc, bt=bt, cu=cu, sk=sk, qlens=qlens, kvlens=kvlens)


def run(d, causal, window, softcap=0.0):
    q = d["q"]
    out = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    unified_attention(q, d["kc"], d["vc"], out, d["cu"], max(d["qlens"]), d["sk"], max(d["kvlens"]), scale, causal,
                      window, d["bt"], softcap, None, None, None)
    ref = ref_attn(q, d["kd"], d["vd"], d["qlens"], d["kvlens"], causal, window, scale, softcap)
    return out, ref


if __name__ == "__main__":
    for causal in (True, False):
        for window in ((-1, -1), (7, 7), (31, 31), (100, 100)):
            for H, Hk in ((8, 8), (8, 2), (16, 1)):
                d = make([64, 37], [64, 37], H, Hk, 64)
                out, ref = run(d, causal, window)
                err = (out.double() - ref).abs()
                print(f"causal={causal!s:5} window={window!s:10} H={H:2} Hk={Hk}: max abs {err.max().item():.2e} "
                      f"rel {(err.norm() / ref.norm()).item():.2e}")
