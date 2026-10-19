"""Independent chain-rule and public PyTorch checks for the reviewed SDPA VJP."""
import copy
from fractions import Fraction as F

import flint
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernel_spec import registry
from kernel_spec.num import ends, evaluate


PROFILE = "torch210_sdpa_defined_real_vjp"
DTYPES = ((torch.float16, "f16", 6e-3), (torch.bfloat16, "bf16", 4e-2),
          (torch.float32, "f32", 3e-6), (torch.float64, "f64", 2e-12))


def base(**updates):
    value = {"q": [[[[1, 0], [0, 1]]]], "k": [[[[1, 0], [0, 1], [1, 1]]]],
             "v": [[[[1, 2], [3, 4], [5, 6]]]], "upstream": [[[[1, -1], [2, 1]]]],
             "dtype": "f64", "attn_mask": None, "attn_mask_dtype": None,
             "is_causal": False, "scale": None, "enable_gqa": False, "dropout_p": 0}
    value.update(updates)
    return value


def arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def independent_chain_rule(inp):
    """Unshifted high-precision formula, separate from d1_attention and the adapter."""
    rank3 = np.asarray(inp["q"], dtype=object).ndim == 3
    q, k, v, upstream = (inp[name] for name in ("q", "k", "v", "upstream"))
    if rank3:
        q, k, v, upstream = [q], [k], [v], [upstream]
    B, Hq, L, E = np.asarray(q, dtype=object).shape
    Hkv, S, Ev = len(k[0]), len(k[0][0]), len(v[0][0][0])
    group = Hq // Hkv
    scale = 1 / arb(E).sqrt() if inp["scale"] is None else arb(inp["scale"])
    zero = arb(0)
    out = [[[[zero for _ in range(Ev)] for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    dq = [[[[zero for _ in range(E)] for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    dk = [[[[zero for _ in range(E)] for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    dv = [[[[zero for _ in range(Ev)] for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    mask = inp["attn_mask"]
    for b in range(B):
        for h in range(Hq):
            g = h // group
            for i in range(L):
                columns, scores = [], []
                for j in range(S):
                    if inp["is_causal"] and j > i:
                        continue
                    bias = 0
                    if mask is not None:
                        entry = mask[i][j]
                        if isinstance(entry, bool) and not entry or entry == "-inf":
                            continue
                        if not isinstance(entry, bool):
                            bias = entry
                    columns.append(j)
                    scores.append(sum((arb(q[b][h][i][e]) * arb(k[b][g][j][e]) for e in range(E)), zero)
                                  * scale + arb(bias))
                weights = [score.exp() for score in scores]
                total = sum(weights[1:], weights[0])
                probabilities = [weight / total for weight in weights]
                do = [arb(x) for x in upstream[b][h][i]]
                dp = []
                for probability, j in zip(probabilities, columns):
                    for d in range(Ev):
                        out[b][h][i][d] += probability * arb(v[b][g][j][d])
                        dv[b][g][j][d] += probability * do[d]
                    dp.append(sum((do[d] * arb(v[b][g][j][d]) for d in range(Ev)), zero))
                mean = sum((probability * value for probability, value in zip(probabilities, dp)), zero)
                for probability, value, j in zip(probabilities, dp, columns):
                    ds = probability * (value - mean) * scale
                    for e in range(E):
                        dq[b][h][i][e] += ds * arb(k[b][g][j][e])
                        dk[b][g][j][e] += ds * arb(q[b][h][i][e])
    result = {"out": out, "dq": dq, "dk": dk, "dv": dv}
    return {name: value if not rank3 else value[0] for name, value in result.items()}


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def assert_encloses(actual, independent):
    if isinstance(independent, list):
        assert len(actual) == len(independent)
        for left, right in zip(actual, independent):
            assert_encloses(left, right)
    else:
        lo, hi = ends(actual)
        if lo == hi:
            assert independent.contains(arb(lo))
            return
        assert lo <= endpoint(independent.lower())
        assert hi >= endpoint(independent.upper())


def midpoint_tree(value):
    if isinstance(value, list):
        return [midpoint_tree(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


def cases():
    return [
        base(is_causal=True),
        base(attn_mask=[[True, False, True], [True, True, False]], attn_mask_dtype="b8", scale=F(1, 2)),
        base(q=[[[[1, 0]], [[0, 1]], [[1, 1]], [[1, -1]]]],
             k=[[[[1, 0], [0, 1]], [[1, 1], [1, -1]]]],
             v=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
             upstream=[[[[1, 2]], [[-1, 1]], [[2, 0]], [[1, -2]]]], enable_gqa=True, scale=F(1, 2)),
        base(q=[[[1, 0]]], k=[[[1, 0], [0, 1]]], v=[[[1, 2], [3, 4]]],
             upstream=[[[2, -1]]], scale=F(1, 2)),
    ]


def test_reference_encloses_independent_chain_rule_for_all_four_outputs():
    adapter = registry.load("ATTN").adapter
    saved = flint.ctx.prec
    try:
        flint.ctx.prec = 768
        for inp in cases():
            before = copy.deepcopy(inp)
            actual = adapter.reference(PROFILE, inp).fields
            independent = independent_chain_rule(inp)
            assert set(actual) == {"out", "dq", "dk", "dv"}
            for name in actual:
                assert_encloses(actual[name], independent[name])
            assert inp == before
    finally:
        flint.ctx.prec = saved


@pytest.mark.parametrize("torch_dtype,dtype_name,tolerance", DTYPES)
@pytest.mark.parametrize("template", cases())
def test_public_math_backend_forward_and_all_operand_gradients(template, torch_dtype, dtype_name, tolerance):
    adapter = registry.load("ATTN").adapter
    inp = copy.deepcopy(template)
    inp["dtype"] = dtype_name
    tensors = []
    for name in ("q", "k", "v"):
        tensor = torch.tensor(inp[name], dtype=torch_dtype, requires_grad=True)
        inp[name] = tensor.detach().tolist()
        tensors.append(tensor)
    upstream = torch.tensor(inp["upstream"], dtype=torch_dtype)
    inp["upstream"] = upstream.tolist()
    mask = inp["attn_mask"]
    if mask is not None:
        mask_dtype = torch.bool if inp["attn_mask_dtype"] == "b8" else torch_dtype
        mask = torch.tensor(mask, dtype=mask_dtype)
        inp["attn_mask"], inp["attn_mask_dtype"] = mask.tolist(), "b8" if mask_dtype is torch.bool else dtype_name
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        public_scale = None if inp["scale"] is None else float(inp["scale"])
        out = torch_f.scaled_dot_product_attention(*tensors, attn_mask=mask, dropout_p=0.0,
            is_causal=inp["is_causal"], scale=public_scale, enable_gqa=inp["enable_gqa"])
        gradients = torch.autograd.grad(out, tensors, upstream)
    expected = adapter.reference(PROFILE, inp).fields
    for name, actual in zip(("out", "dq", "dk", "dv"), (out, *gradients)):
        target = torch.tensor(midpoint_tree(expected[name]), dtype=torch.float64)
        torch.testing.assert_close(actual.double(), target, rtol=tolerance, atol=tolerance)


def test_vjp_requires_complete_upstream_and_complete_result():
    adapter = registry.load("ATTN").adapter
    inp = base(q=[[[0]]], k=[[[1], [3]]], v=[[[1], [3]]], upstream=[[[2]]], scale=1, dtype="f32")
    fields = {name: {"t": "f32", "shape": list(np.asarray(value, dtype=object).shape), "v": value}
              for name, value in {"out": [[[2]]], "dq": [[[2]]], "dk": [[[0], [0]]],
                                         "dv": [[[1], [1]]]}.items()}
    assert adapter.check_output(PROFILE, inp, fields)
    for missing in fields:
        assert not adapter.check_output(PROFILE, inp, {k: v for k, v in fields.items() if k != missing})
    assert not adapter.check_output(PROFILE, inp, dict(fields, lse=[]))
    assert evaluate(adapter.reference, PROFILE, dict(inp, upstream=[[2]])).outcome == "invalid_input"
    assert evaluate(adapter.reference, PROFILE, {k: v for k, v in inp.items() if k != "upstream"}).outcome == "invalid_input"
