"""Triton's official tutorials (triton-lang/triton v3.6.0, python/tutorials) for tool_spec_check: 05-layer-norm and
06-fused-attention, the two reference kernels most often copied into other projects.

The tutorial files are loaded from .cache/src with their module-level test / benchmark code cut off.  Specifications
(float64, declared bound 2^-40 * max|f|):

* layer norm: y = (x - mean) / sqrt(var + eps) * w + b (biased variance), backward = float64 autograd;
* fused attention: o = softmax(q k^T * sm_scale [causal: j <= i]) v on the float16 inputs, backward = float64
  autograd for the same upstream gradient.
Environment: ka_main.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn.functional as F

from tool_spec_cases_tridao import FnCase, rn

SRC = Path(__file__).resolve().parents[1] / ".cache" / "src"
_MODULES = {}


def tutorial(name, cut):
    if name not in _MODULES:
        text = (SRC / f"triton_tut_{name}").read_text()
        text = text[: text.index(cut)]
        mod = types.ModuleType(f"triton_tutorial_{name.split('-')[0]}")
        mod.__file__ = str(SRC / f"triton_tut_{name}")
        exec(compile(text, mod.__file__, "exec"), mod.__dict__)
        _MODULES[name] = mod
    return _MODULES[name]


def _ln_mod():
    return tutorial("05-layer-norm.py", "def test_layer_norm(")


def _attn_mod():
    return tutorial("06-fused-attention.py", "try:\n    from flash_attn")


def _ln_case(name, M=256, N=512, backward=False):
    def make(g):
        return {"x": rn(g, M, N, scale=2) + 0.5, "w": 1 + 0.1 * rn(g, N), "b": 0.1 * rn(g, N)}

    def run(inp):
        return {"y": _ln_mod().layer_norm(inp["x"], (N,), inp["w"], inp["b"], 1e-5)}

    def ref(inp):
        return {"y": F.layer_norm(inp["x"], (N,), inp["w"], inp["b"], 1e-5)}

    return FnCase(f"tut_layernorm_{name}" + ("_bwd" if backward else ""), "Triton tutorial 05-layer-norm (v3.6.0)",
                  f"layer norm M={M} N={N} (biased variance, eps 1e-5)" + ("; backward dx dw db" if backward else ""),
                  make, run, ref, grad_of=(["x", "w", "b"] if backward else ()), loss_of="y")


def _attn_case(name, Z=1, H=2, N=256, D=64, causal=False, backward=False):
    def make(g):
        return {"q": rn(g, Z, H, N, D).half(), "k": rn(g, Z, H, N, D).half(), "v": rn(g, Z, H, N, D).half()}

    def run(inp):
        return {"o": _attn_mod().attention(inp["q"], inp["k"], inp["v"], causal, D ** -0.5, False)}

    def ref(inp):
        s = inp["q"] @ inp["k"].transpose(-1, -2) * (D ** -0.5)
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
        return {"o": torch.softmax(s, -1) @ inp["v"]}

    return FnCase(f"tut_attention_{name}" + ("_bwd" if backward else ""), "Triton tutorial 06-fused-attention (v3.6.0)",
                  f"softmax attention Z={Z} H={H} N={N} D={D} causal={causal} on float16 inputs"
                  + ("; backward dq dk dv" if backward else ""),
                  make, run, ref, grad_of=(["q", "k", "v"] if backward else ()), loss_of="o")


class _Half(FnCase):
    """The float16 kernel's leaves must stay float16 on the kernel side; the base class clones them as they are."""


CASES = [
    _ln_case("fp32"), _ln_case("fp32", backward=True), _ln_case("n1000", N=1000), _ln_case("n1000", N=1000, backward=True),
    _attn_case("plain"), _attn_case("causal", causal=True), _attn_case("plain", backward=True),
    _attn_case("causal", causal=True, backward=True), _attn_case("d128_causal", D=128, causal=True),
]
