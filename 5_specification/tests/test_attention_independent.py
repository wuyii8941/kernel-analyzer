"""Independent and public PyTorch checks for the reviewed SDPA forward profile."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends, evaluate


PROFILE = "torch210_sdpa_defined_real"
DTYPES = (
    (torch.float16, "f16", 4e-3),
    (torch.bfloat16, "bf16", 3e-2),
    (torch.float32, "f32", 2e-6),
    (torch.float64, "f64", 1e-12),
)


def _base(**updates):
    value = {
        "q": [[[[1, 0], [0, 1]]]],
        "k": [[[[1, 0], [0, 1], [1, 1]]]],
        "v": [[[[1, 2], [3, 4], [5, 6]]]],
        "dtype": "f64", "attn_mask": None, "attn_mask_dtype": None,
        "is_causal": False, "scale": None, "enable_gqa": False, "dropout_p": 0,
    }
    value.update(updates)
    return value


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _independent(inp):
    q, k, v = inp["q"], inp["k"], inp["v"]
    rank3 = len(np.asarray(q, dtype=object).shape) == 3
    if rank3:
        q, k, v = [q], [k], [v]
    batch, hq, length, head_dim = np.asarray(q, dtype=object).shape
    hkv, source = len(k[0]), len(k[0][0])
    ev = len(v[0][0][0])
    group = hq // hkv
    scale = 1 / _arb(head_dim).sqrt() if inp["scale"] is None else _arb(inp["scale"])
    mask = inp["attn_mask"]
    result = [[[[None for _ in range(ev)] for _ in range(length)] for _ in range(hq)] for _ in range(batch)]
    for b in range(batch):
        for h in range(hq):
            g = h // group
            for i in range(length):
                scores, columns = [], []
                for j in range(source):
                    if inp["is_causal"] and j > i:
                        continue
                    bias = 0
                    if mask is not None:
                        entry = mask[i][j]
                        if isinstance(entry, bool) and not entry or entry == "-inf":
                            continue
                        if not isinstance(entry, bool):
                            bias = entry
                    score = sum((_arb(q[b][h][i][d]) * _arb(k[b][g][j][d]) for d in range(head_dim)), _arb(0))
                    scores.append(score * scale + _arb(bias))
                    columns.append(j)
                weights = [score.exp() for score in scores]
                total = sum(weights[1:], weights[0])
                for d in range(ev):
                    result[b][h][i][d] = sum(
                        (weight * _arb(v[b][g][j][d]) for weight, j in zip(weights, columns)), _arb(0)
                    ) / total
    return result if not rank3 else result[0]


def _midpoint_tree(value):
    if isinstance(value, list):
        return [_midpoint_tree(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


class IndependentAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("ATTN").adapter

    def test_reference_encloses_independent_high_precision_formula(self):
        cases = [
            _base(is_causal=True),
            _base(attn_mask=[[True, False, True], [True, True, False]], attn_mask_dtype="b8", scale=F(1, 2)),
            _base(attn_mask=[[0, -1, "-inf"], [1, 0, -2]], attn_mask_dtype="f64", scale=F(-1, 2)),
            _base(
                q=[[[[1, 0]], [[0, 1]], [[1, 1]], [[1, -1]]]],
                k=[[[[1, 0], [0, 1]], [[1, 1], [1, -1]]]],
                v=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], enable_gqa=True, scale=F(1, 2),
            ),
            _base(q=[[[[1, 0]], [[0, 1]]]], k=[[[[1, 0], [0, 1]]]],
                  v=[[[[1, 2], [3, 4]]]], scale=F(1, 2)),
        ]
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(cases):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                observed = self.adapter.reference(PROFILE, inp).fields["out"]
                expected = _independent(inp)

                def visit(actual, independent):
                    if isinstance(independent, list):
                        self.assertEqual(len(actual), len(independent))
                        for left, right in zip(actual, independent):
                            visit(left, right)
                    else:
                        lo, hi = ends(actual)
                        if lo == hi:
                            self.assertTrue(independent.contains(_arb(lo)))
                            return
                        self.assertLessEqual(lo, _fraction(independent.lower()))
                        self.assertGreaterEqual(hi, _fraction(independent.upper()))
                visit(observed, expected)
                self.assertEqual(inp, before)

    def test_twenty_public_math_backend_calls_follow_the_same_task(self):
        torch_cases = [
            _base(is_causal=True),
            _base(attn_mask=[[True, False, True], [True, True, False]], attn_mask_dtype="b8", scale=0.5),
            _base(attn_mask=[[0, -1, -2], [1, 0, -2]], attn_mask_dtype="f64", scale=-0.5),
            _base(q=[[[[1, 0]], [[0, 1]], [[1, 1]], [[1, -1]]]],
                  k=[[[[1, 0], [0, 1]], [[1, 1], [1, -1]]]],
                  v=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], enable_gqa=True, scale=0.5),
            _base(q=[[[1, 0]], [[0, 1]]], k=[[[1, 0], [0, 1]]], v=[[[1, 2], [3, 4]]], scale=0.5),
        ]
        checked = 0
        for torch_dtype, dtype_name, tolerance in DTYPES:
            for index, template in enumerate(torch_cases):
                with self.subTest(dtype=str(torch_dtype), case=index):
                    inp = copy.deepcopy(template)
                    inp["dtype"] = dtype_name
                    q = torch.tensor(inp["q"], dtype=torch_dtype)
                    k = torch.tensor(inp["k"], dtype=torch_dtype)
                    v = torch.tensor(inp["v"], dtype=torch_dtype)
                    mask = inp["attn_mask"]
                    if mask is not None:
                        mask_dtype = torch.bool if inp["attn_mask_dtype"] == "b8" else torch_dtype
                        mask = torch.tensor(mask, dtype=mask_dtype)
                        inp["attn_mask"] = mask.tolist()
                        inp["attn_mask_dtype"] = "b8" if mask_dtype is torch.bool else dtype_name
                    inp["q"], inp["k"], inp["v"] = q.tolist(), k.tolist(), v.tolist()
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        actual = torch_f.scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=inp["is_causal"],
                            scale=inp["scale"], enable_gqa=inp["enable_gqa"],
                        )
                    expected = torch.tensor(
                        _midpoint_tree(self.adapter.reference(PROFILE, inp).fields["out"]), dtype=torch.float64
                    )
                    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
                    checked += 1
        self.assertEqual(checked, 20)

    def test_complete_output_checker_enforces_field_dtype_and_shape(self):
        inp = _base(q=[[[0]]], k=[[[0]]], v=[[[3]]], dtype="f32")
        exact = {"out": {"t": "f32", "shape": [1, 1, 1], "v": [[[3]]]}}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, exact))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": exact["out"], "lse": []}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": dict(exact["out"], t="f64")}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": dict(exact["out"], shape=[1, 1])}))

    def test_uncertain_public_boundaries_are_refused(self):
        empty = _base(q=[[[0]]], k=[[[0], [0]]], v=[[[1], [3]]],
                      attn_mask=[[False, False]], attn_mask_dtype="b8")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, empty).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(_base(), lse=True)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(_base(), dropout_p=F(1, 10))).outcome,
                         "unsupported")


if __name__ == "__main__":
    unittest.main()
