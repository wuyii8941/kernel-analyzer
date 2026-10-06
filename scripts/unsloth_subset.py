"""Bindings for the unfamiliar Triton subset: Unsloth @ 88cfd06 (docs/unfamiliar_subset_protocol_20261006.md).

Run with the shim on the path (scripts/build_unsloth_shim.py):

    PYTHONPATH=.cache/pylibs/unsloth_shim:src python -m kernel_analyzer.cli check scripts/unsloth_subset.py \
        --out results/external/unsloth --seeds 96

Each case calls a public Unsloth entry point on bf16 inputs (one independent draw per seed) and returns the tensors
that its Triton kernels write; the specification is written from the public mathematical definition in float64
(torch eager, autograd for the backward outputs), not from the kernel code.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kernel_analyzer.check import Case, f64_point_spec

BF16 = torch.bfloat16
COMMIT = "88cfd06"


def _gen(seed, salt):
    return torch.Generator().manual_seed(1_000_003 * salt + seed)


def _bf16(t):
    return t.to(BF16).cuda()


def _spec(d):
    return {k: f64_point_spec(v.detach().cpu().numpy()) for k, v in d.items()}


class UnslothCase(Case):
    spec_bound = "float64 torch evaluation of the public definition, declared bound 2^-40 max|f| (screening)"

    def __init__(self, uid, name, salt, implementation, specification):
        self.name = f"{uid}_{name}"
        self.salt = salt
        self.implementation = f"unsloth@{COMMIT} {implementation}"
        self.specification = specification


class RMSNormCase(UnslothCase):
    def __init__(self, uid, gemma):
        super().__init__(uid, "rms_layernorm_gemma" if gemma else "rms_layernorm", 11 + gemma,
                         f"fast_rms_layernorm(gemma={gemma}) forward + backward",
                         "y = x / sqrt(mean(x^2) + eps) * " + ("(1 + w)" if gemma else "w") + "; dX by autograd")
        self.gemma = gemma

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        w = (0.1 * torch.randn(4096, generator=g)) if self.gemma else (1 + 0.1 * torch.randn(4096, generator=g))
        return {"X": _bf16(torch.randn(16, 4096, generator=g)), "W": _bf16(w),
                "dY": _bf16(torch.randn(16, 4096, generator=g))}

    def launch(self, inp):
        from unsloth.kernels.rms_layernorm import fast_rms_layernorm

        X = inp["X"].clone().requires_grad_(True)
        Y = fast_rms_layernorm(SimpleNamespace(weight=inp["W"], variance_epsilon=1e-6), X, gemma=self.gemma)
        Y.backward(inp["dY"].clone())  # the backward writes dX over the gradient it receives
        return {"Y": Y, "dX": X.grad}

    def spec(self, inp):
        X = inp["X"].double().requires_grad_(True)
        w = inp["W"].double()
        Y = X / torch.sqrt((X * X).mean(-1, keepdim=True) + 1e-6) * ((1 + w) if self.gemma else w)
        Y.backward(inp["dY"].double())
        return _spec({"Y": Y, "dX": X.grad})


class LayerNormCase(UnslothCase):
    def __init__(self, uid):
        super().__init__(uid, "layernorm", 13, "fast_layernorm forward + backward",
                         "(x - mean) / sqrt(var + eps) * w + b (biased variance); dX by autograd")

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        return {"X": _bf16(torch.randn(16, 4096, generator=g)), "W": _bf16(1 + 0.1 * torch.randn(4096, generator=g)),
                "b": _bf16(0.1 * torch.randn(4096, generator=g)), "dY": _bf16(torch.randn(16, 4096, generator=g))}

    def launch(self, inp):
        from unsloth.kernels.layernorm import fast_layernorm

        ln = SimpleNamespace(weight=inp["W"], bias=inp["b"], eps=1e-5, elementwise_affine=True)
        X = inp["X"].clone().requires_grad_(True)
        Y = fast_layernorm(ln, X)
        Y.backward(inp["dY"].clone())
        return {"Y": Y, "dX": X.grad}

    def spec(self, inp):
        X = inp["X"].double().requires_grad_(True)
        Y = F.layer_norm(X, (4096,), inp["W"].double(), inp["b"].double(), eps=1e-5)
        Y.backward(inp["dY"].double())
        return _spec({"Y": Y, "dX": X.grad})


def _gelu(e, approx):
    if approx:
        return 0.5 * e * (1 + torch.tanh(math.sqrt(2 / math.pi) * (e + 0.044715 * e ** 3)))
    return 0.5 * e * (1 + torch.erf(e / math.sqrt(2)))


class GatedForwardCase(UnslothCase):
    """h = act(e) * g; act = silu (SwiGLU) or gelu (GeGLU exact / tanh approximation)."""

    def __init__(self, uid, act, salt):
        fn = {"silu": "swiglu_fg_kernel", "gelu_exact": "geglu_exact_forward_kernel",
              "gelu_approx": "geglu_approx_forward_kernel"}[act]
        super().__init__(uid, fn, salt, fn, f"h = {act}(e) * g")
        self.act, self.fn = act, fn

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        return {"e": _bf16(torch.randn(1, 4, 11008, generator=g)), "g": _bf16(torch.randn(1, 4, 11008, generator=g))}

    def launch(self, inp):
        import unsloth.kernels.geglu as geglu
        import unsloth.kernels.swiglu as swiglu

        f = getattr(swiglu if self.act == "silu" else geglu, self.fn)
        return {"h": f(inp["e"], inp["g"])}

    def spec(self, inp):
        e, g = inp["e"].double(), inp["g"].double()
        a = F.silu(e) if self.act == "silu" else _gelu(e, self.act == "gelu_approx")
        return _spec({"h": a * g})


class GatedBackwardCase(UnslothCase):
    """Backward entry points that write (h, df, de) over (DW, e, g): h = f(e) g, df = DW f(e), de = DW g f'(e)."""

    def __init__(self, uid, act, salt):
        fn = {"silu": "swiglu_DWf_DW_dfg_kernel", "gelu_exact": "geglu_exact_backward_kernel",
              "gelu_approx": "geglu_approx_backward_kernel"}[act]
        super().__init__(uid, fn, salt, fn, f"h = f(e) g, df = DW f(e), de = DW g f'(e), f = {act}")
        self.act, self.fn = act, fn

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        return {k: _bf16(torch.randn(4, 11008, generator=g)) for k in ("DW", "e", "g")}

    def launch(self, inp):
        import unsloth.kernels.geglu as geglu
        import unsloth.kernels.swiglu as swiglu

        DW, e, g = (inp[k].clone() for k in ("DW", "e", "g"))
        getattr(swiglu if self.act == "silu" else geglu, self.fn)(DW, e, g)
        return {"h": DW, "df": e, "de": g}

    def spec(self, inp):
        DW, g = inp["DW"].double(), inp["g"].double()
        e = inp["e"].double().requires_grad_(True)
        f = F.silu(e) if self.act == "silu" else _gelu(e, self.act == "gelu_approx")
        (fp,) = torch.autograd.grad(f.sum(), e)
        f = f.detach()
        return _spec({"h": f * g, "df": DW * f, "de": DW * g * fp})


class CrossEntropyCase(UnslothCase):
    def __init__(self, uid, name, rows, vocab, softcap, salt):
        super().__init__(uid, name, salt, f"Fast_CrossEntropyLoss forward + backward (vocab {vocab}, softcap {softcap})",
                         "loss = logsumexp(z) - z_label" + (f", z <- {softcap} tanh(z / {softcap})" if softcap else "")
                         + "; dlogits by autograd")
        self.rows, self.vocab, self.softcap = rows, vocab, softcap

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        return {"logits": _bf16(2 * torch.randn(self.rows, self.vocab, generator=g)),
                "labels": torch.randint(0, self.vocab, (self.rows,), generator=g).cuda(),
                "dloss": torch.randn(self.rows, generator=g).float().cuda()}

    def launch(self, inp):
        from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

        z = inp["logits"].clone().requires_grad_(True)
        loss = Fast_CrossEntropyLoss.apply(z, inp["labels"], self.softcap, 0)
        loss.backward(inp["dloss"].clone())
        return {"loss": loss, "dlogits": z.grad}

    def spec(self, inp):
        z = inp["logits"].double().requires_grad_(True)
        zz = self.softcap * torch.tanh(z / self.softcap) if self.softcap else z
        loss = torch.logsumexp(zz, -1) - zz.gather(1, inp["labels"][:, None]).squeeze(1)
        loss.backward(inp["dloss"].double())
        return _spec({"loss": loss, "dlogits": z.grad})


def _rope_tables(seq, dim):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.arange(seq, dtype=torch.float32)[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], -1)
    return _bf16(torch.cos(ang)), _bf16(torch.sin(ang))


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


class RopeCase(UnslothCase):
    def __init__(self, uid, indices):
        super().__init__(uid, "rope_qk_indices" if indices else "rope", 21 + indices,
                         f"fast_rope_embedding forward + backward ({'with' if indices else 'without'} rope indices)",
                         "q' = q cos + rotate_half(q) sin (cos, sin given), backward = transpose; batch, heads, seq, dim")
        self.indices = indices

    def inputs(self, seed):
        g = _gen(seed, self.salt)
        cos, sin = _rope_tables(64, 128)
        return {"Q": _bf16(torch.randn(1, 8, 64, 128, generator=g)), "K": _bf16(torch.randn(1, 8, 64, 128, generator=g)),
                "dQ": _bf16(torch.randn(1, 8, 64, 128, generator=g)), "dK": _bf16(torch.randn(1, 8, 64, 128, generator=g)),
                "cos": cos, "sin": sin}

    def launch(self, inp):
        from unsloth.kernels.rope_embedding import fast_rope_embedding

        Q = inp["Q"].clone().requires_grad_(True)
        K = inp["K"].clone().requires_grad_(True)
        idx = torch.arange(64, dtype=torch.int32, device="cuda")[None] if self.indices else None
        Qo, Ko = fast_rope_embedding(Q, K, inp["cos"], inp["sin"], idx)
        # autograd.grad returns the backward's own buffers (here views of what the kernel wrote); .backward() would
        # copy them into the leaves' layout with a torch op
        dQ, dK = torch.autograd.grad([Qo, Ko], [Q, K], [inp["dQ"].clone(), inp["dK"].clone()])
        return {"Q_out": Qo, "K_out": Ko, "dQ": dQ, "dK": dK}

    def spec(self, inp):
        cos, sin = inp["cos"].double(), inp["sin"].double()
        Q, K = inp["Q"].double().requires_grad_(True), inp["K"].double().requires_grad_(True)
        Qo = Q * cos + _rotate_half(Q) * sin
        Ko = K * cos + _rotate_half(K) * sin
        torch.autograd.backward([Qo, Ko], [inp["dQ"].double(), inp["dK"].double()])
        return _spec({"Q_out": Qo, "K_out": Ko, "dQ": Q.grad, "dK": K.grad})


CASES = [
    RMSNormCase("U01", False), RMSNormCase("U02", True), LayerNormCase("U03"),
    GatedForwardCase("U04", "silu", 14), GatedBackwardCase("U05", "silu", 15),
    GatedForwardCase("U06", "gelu_exact", 16), GatedBackwardCase("U07", "gelu_exact", 17),
    GatedForwardCase("U08", "gelu_approx", 18), GatedBackwardCase("U09", "gelu_approx", 19),
    CrossEntropyCase("U10", "cross_entropy", 4, 32000, 0, 31),
    CrossEntropyCase("U11", "cross_entropy_chunked", 2, 128256, 0, 32),
    CrossEntropyCase("U12", "cross_entropy_softcap", 4, 32000, 30.0, 33),
    RopeCase("U13", False), RopeCase("U14", True),
]
