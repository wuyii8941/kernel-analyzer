"""FlexAttention (torch.nn.attention.flex_attention, compiled by Inductor into Triton templates) for tool_spec_check.

Specification f: the documented FlexAttention semantics, evaluated in float64 eager PyTorch on the same inputs:

    S = (Q K^T) * scale,  S'[b,h,i,j] = score_mod(S[b,h,i,j], b, h, i, j),  S' = -inf where mask_mod is False,
    out = softmax(S') V   (rows with every entry masked give 0, as documented),

with GQA by repeating K/V heads; the backward specification is float64 autograd of the same expression for the
same upstream gradient.  Environment: ka_main (torch 2.10.0, triton 3.6.0); fresh Inductor cache per process,
float32 matmul precision "highest".
"""

from __future__ import annotations

import torch

from tool_spec_check import Case, f64_point_spec


def _doc_ids(n, device):
    ids = torch.zeros(n, dtype=torch.long, device=device)
    for cut in (n // 5, n // 2, (4 * n) // 5):
        ids[cut:] += 1
    return ids


class FlexCase(Case):
    implementation = "torch.compile(torch.nn.attention.flex_attention.flex_attention) -> Inductor Triton templates"
    spec_bound = "float64 eager evaluation, declared bound 2^-40 * max|f|"

    def __init__(self, name, B=1, H=2, HKV=None, Q=256, KV=256, D=64, DV=None, score="none", mask="none",
                 backward=False, bias_grad=False):
        self.name = name
        self.B, self.H, self.HKV, self.Q, self.KV, self.D = B, H, HKV or H, Q, KV, D
        self.DV = DV or D
        self.score, self.mask, self.backward, self.bias_grad = score, mask, backward, bias_grad
        self.specification = (f"FlexAttention semantics in float64: B={B} H={H} H_kv={self.HKV} Q={Q} KV={KV} D={D} "
                              f"Dv={self.DV} score_mod={score} mask_mod={mask}" + ("; backward dq dk dv" if backward else ""))

    # ---- mods, shared by the kernel call and the float64 specification
    def _score_mod(self, bias=None):
        kind = self.score
        if kind == "none":
            return None
        if kind == "softcap":
            return lambda s, b, h, i, j: 20.0 * torch.tanh(s / 20.0)
        if kind == "alibi":
            slopes = self._slopes

            def alibi(s, b, h, i, j):
                return s + slopes[h] * (j - i)
            return alibi
        if kind == "bias":
            return lambda s, b, h, i, j: s + bias[h, i, j]
        raise ValueError(kind)

    def _mask_mod(self):
        kind = self.mask
        if kind == "none":
            return None
        if kind == "causal":
            return lambda b, h, i, j: i >= j
        if kind == "sliding_causal":
            return lambda b, h, i, j: (i >= j) & (i - j < 64)
        if kind == "document":
            doc = self._doc

            def document(b, h, i, j):
                return (doc[i] == doc[j]) & (i >= j)
            return document
        raise ValueError(kind)

    def setup(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        dev = "cuda"
        self._slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / self.H) for h in range(self.H)], device=dev)
        self._doc = _doc_ids(max(self.Q, self.KV), dev)
        mm = self._mask_mod()
        self._block_mask = None if mm is None else create_block_mask(mm, self.B, self.H, self.Q, self.KV, device=dev)
        self._flex = torch.compile(flex_attention)
        inp = self.inputs(10_000)
        self.launch(inp)  # compile (and compile the backward) outside the recorder
        torch.cuda.synchronize()

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(self.B, self.H, self.Q, self.D, generator=g).cuda()
        k = torch.randn(self.B, self.HKV, self.KV, self.D, generator=g).cuda()
        v = torch.randn(self.B, self.HKV, self.KV, self.DV, generator=g).cuda()
        out = {"q": q, "k": k, "v": v}
        if self.score == "bias":
            out["bias"] = 0.5 * torch.randn(self.H, self.Q, self.KV, generator=g).cuda()
        if self.backward:
            out["g"] = torch.randn(self.B, self.H, self.Q, self.DV, generator=g).cuda()
        return out

    def launch(self, inp):
        q, k, v = inp["q"].clone(), inp["k"].clone(), inp["v"].clone()
        bias = inp.get("bias")
        if self.backward:
            for t in (q, k, v):
                t.requires_grad_(True)
            if bias is not None and self.bias_grad:
                bias = bias.clone().requires_grad_(True)
        out = self._flex(q, k, v, score_mod=self._score_mod(bias), block_mask=self._block_mask,
                         enable_gqa=self.HKV != self.H)
        if not self.backward:
            return {"out": out}
        out.backward(inp["g"])
        res = {"dq": q.grad, "dk": k.grad, "dv": v.grad}
        if bias is not None and self.bias_grad:
            res["dbias"] = bias.grad
        return res

    def spec(self, inp):
        q, k, v = (inp[n].double() for n in ("q", "k", "v"))
        bias = inp.get("bias")
        bias = None if bias is None else bias.double()
        leaves = [q, k, v]
        if self.backward:
            for t in leaves:
                t.requires_grad_(True)
            if bias is not None and self.bias_grad:
                bias.requires_grad_(True)
        rep = self.H // self.HKV
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = (q @ kk.transpose(-1, -2)) * (self.D ** -0.5)
        b_idx = torch.arange(self.B, device=s.device)[:, None, None, None]
        h_idx = torch.arange(self.H, device=s.device)[None, :, None, None]
        i_idx = torch.arange(self.Q, device=s.device)[None, None, :, None]
        j_idx = torch.arange(self.KV, device=s.device)[None, None, None, :]
        sm = self._score_mod(bias)
        if sm is not None:
            if self.score == "alibi":
                s = s + self._slopes.double()[h_idx] * (j_idx - i_idx)
            else:
                s = sm(s, b_idx, h_idx, i_idx, j_idx)
        mm = self._mask_mod()
        if mm is not None:
            keep = mm(b_idx, h_idx, i_idx, j_idx).expand_as(s)
            s = s.masked_fill(~keep, float("-inf"))
            dead = ~keep.any(-1, keepdim=True)
            p = torch.softmax(s.masked_fill(dead, 0.0), -1).masked_fill(dead, 0.0)
        else:
            p = torch.softmax(s, -1)
        out = p @ vv
        if not self.backward:
            return {"out": f64_point_spec(out.detach().cpu().numpy())}
        grads = torch.autograd.grad(out, leaves + ([bias] if bias is not None and self.bias_grad else []),
                                    inp["g"].double())
        res = {n: f64_point_spec(gr.detach().cpu().numpy()) for n, gr in zip(("dq", "dk", "dv"), grads[:3])}
        if bias is not None and self.bias_grad:
            res["dbias"] = f64_point_spec(grads[3].detach().cpu().numpy())
        return res


_CONFIGS = [
    ("plain", {}),
    ("causal", {"mask": "causal"}),
    ("sliding_causal", {"mask": "sliding_causal"}),
    ("softcap", {"score": "softcap", "mask": "causal"}),
    ("alibi", {"score": "alibi", "mask": "causal"}),
    ("gqa", {"H": 4, "HKV": 2, "mask": "causal"}),
    ("document", {"mask": "document"}),
    ("len200_causal", {"Q": 200, "KV": 200, "mask": "causal"}),
    ("q100_kv300", {"Q": 100, "KV": 300}),
    ("d96", {"D": 96, "mask": "causal"}),
    ("dv32", {"DV": 32, "mask": "causal"}),
    ("bias", {"score": "bias", "Q": 128, "KV": 128, "bias_grad": True}),
]

CASES = []
for _name, _kw in _CONFIGS:
    CASES.append(FlexCase(f"flex_fwd_{_name}", **{k: v for k, v in _kw.items() if k != "bias_grad"}))
    CASES.append(FlexCase(f"flex_bwd_{_name}", backward=True, **_kw))
