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
                 backward=False, bias_grad=False, kernel_options=None):
        self.name = name
        self.B, self.H, self.HKV, self.Q, self.KV, self.D = B, H, HKV or H, Q, KV, D
        self.DV = DV or D
        self.score, self.mask, self.backward, self.bias_grad = score, mask, backward, bias_grad
        self.kernel_options = kernel_options
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
        # decode-style masks: query i sits at absolute position i + (KV - Q)
        off = self.KV - self.Q
        if kind == "decode_causal":
            return lambda b, h, i, j: i + off >= j
        if kind == "decode_sliding":
            return lambda b, h, i, j: (i + off >= j) & (i + off - j < 128)
        if kind == "decode_document":
            doc = self._doc

            def decode_document(b, h, i, j):
                return (doc[i + off] == doc[j]) & (i + off >= j)
            return decode_document
        raise ValueError(kind)

    def setup(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        dev = "cuda"
        self._slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / self.H) for h in range(self.H)], device=dev)
        self._doc = _doc_ids(max(self.Q, self.KV), dev)
        mm = self._mask_mod()
        # one block-mask head (broadcast) for the decode cases: flex decoding with GQA requires it
        heads = None if self.mask.startswith("decode") else self.H
        self._block_mask = None if mm is None else create_block_mask(mm, self.B, heads, self.Q, self.KV, device=dev)
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
                         enable_gqa=self.HKV != self.H, kernel_options=self.kernel_options)
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


# smaller tiles so the templates fit the A6000's 101 KB of shared memory (the default configs do not)
_SMALL = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1}

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
    ("d96", {"D": 96, "mask": "causal", "kernel_options": _SMALL}),
    ("dv32", {"DV": 32, "mask": "causal"}),
    ("bias", {"score": "bias", "Q": 128, "KV": 128, "bias_grad": True, "kernel_options": _SMALL}),
]

CASES = []
for _name, _kw in _CONFIGS:
    CASES.append(FlexCase(f"flex_fwd_{_name}", **{k: v for k, v in _kw.items() if k != "bias_grad"}))
    CASES.append(FlexCase(f"flex_bwd_{_name}", backward=True, **_kw))


class FlexNewer(FlexCase):
    """Newer FlexAttention paths: a captured per-head ALiBi slope tensor that requires grad (its gradient is a
    reduction over every (q, kv) pair, accumulated by the backward template), and the gradient through the returned
    log-sum-exp (return_lse=True; loss = <out, g> + <lse, g2>)."""

    def __init__(self, name, kind, **kw):
        super().__init__(name, backward=True, **kw)
        self.kind = kind
        self.specification += f"; {kind}"

    def inputs(self, seed):
        inp = super().inputs(seed)
        g = torch.Generator(device="cpu").manual_seed(seed + 77_777)
        if self.kind == "learned_alibi":
            inp["slopes"] = (0.05 * torch.rand(self.H, generator=g) + 0.01).cuda()
        else:
            inp["g2"] = torch.randn(self.B, self.H, self.Q, generator=g).cuda()
        return inp

    def launch(self, inp):
        q, k, v = (inp[n].clone().requires_grad_(True) for n in ("q", "k", "v"))
        if self.kind == "learned_alibi":
            slopes = inp["slopes"].clone().requires_grad_(True)
            out = self._flex(q, k, v, score_mod=lambda s, b, h, i, j: s + slopes[h] * (j - i),
                             block_mask=self._block_mask, kernel_options=self.kernel_options)
            out.backward(inp["g"])
            return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dslopes": slopes.grad}
        out, lse = self._flex(q, k, v, block_mask=self._block_mask, return_lse=True,
                              kernel_options=self.kernel_options)
        ((out * inp["g"]).sum() + (lse * inp["g2"]).sum()).backward()
        return {"dq": q.grad, "dk": k.grad, "dv": v.grad}

    def spec(self, inp):
        q, k, v = (inp[n].double().requires_grad_(True) for n in ("q", "k", "v"))
        s = (q @ k.transpose(-1, -2)) * (self.D ** -0.5)
        i_idx = torch.arange(self.Q, device=s.device)[:, None]
        j_idx = torch.arange(self.KV, device=s.device)[None, :]
        leaves = [q, k, v]
        if self.kind == "learned_alibi":
            slopes = inp["slopes"].double().requires_grad_(True)
            leaves.append(slopes)
            s = s + slopes[None, :, None, None] * (j_idx - i_idx)
        mm = self._mask_mod()
        if mm is not None:
            keep = mm(None, None, i_idx, j_idx)
            s = s.masked_fill(~keep, float("-inf"))
        lse = torch.logsumexp(s, -1)
        out = torch.softmax(s, -1) @ v
        loss = (out * inp["g"].double()).sum()
        if self.kind == "lse":
            loss = loss + (lse * inp["g2"].double()).sum()
        grads = torch.autograd.grad(loss, leaves)
        names = ["dq", "dk", "dv"] + (["dslopes"] if self.kind == "learned_alibi" else [])
        return {n: f64_point_spec(gr.detach().cpu().numpy()) for n, gr in zip(names, grads)}

    def setup(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        mm = self._mask_mod()
        self._doc = None
        self._block_mask = None if mm is None else create_block_mask(mm, self.B, self.H, self.Q, self.KV, device="cuda")
        self._flex = torch.compile(flex_attention)
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()


CASES += [FlexNewer("flex_bwd_learned_alibi", "learned_alibi", mask="causal", kernel_options=_SMALL),
          FlexNewer("flex_bwd_lse_grad", "lse", mask="causal"),
          FlexNewer("flex_bwd_lse_grad_plain", "lse")]


# Flex decoding (query length < 128 selects the split-KV decoding template; GQA needs a power-of-two group size and
# a single block-mask head).  Forward only: the decoding template has no backward of its own.
_DECODE = [
    ("dec_q1_kv1027", {"Q": 1, "KV": 1027}),
    ("dec_q1_kv1027_causal", {"Q": 1, "KV": 1027, "mask": "decode_causal"}),
    ("dec_q7_kv300_causal", {"Q": 7, "KV": 300, "mask": "decode_causal"}),
    ("dec_q7_kv1000_sliding", {"Q": 7, "KV": 1000, "mask": "decode_sliding"}),
    ("dec_q64_kv777_document", {"Q": 64, "KV": 777, "mask": "decode_document"}),
    ("dec_q4_kv513_gqa4", {"Q": 4, "KV": 513, "H": 8, "HKV": 2, "mask": "decode_causal"}),
    ("dec_q4_kv513_gqa4_softcap", {"Q": 4, "KV": 513, "H": 8, "HKV": 2, "score": "softcap"}),
    ("dec_q3_kv700_alibi", {"Q": 3, "KV": 700, "score": "alibi", "mask": "decode_causal"}),
    ("dec_q2_kv400_dv32", {"Q": 2, "KV": 400, "DV": 32, "mask": "decode_causal"}),
    ("dec_q16_kv2048_b3", {"B": 3, "Q": 16, "KV": 2048, "mask": "decode_sliding"}),
]
for _name, _kw in _DECODE:
    CASES.append(FlexCase(f"flex_{_name}", **_kw))
