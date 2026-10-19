"""Independent checks for the reviewed FlashAttention variable-length contract."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends, evaluate


CAUSAL = "flashattn283_varlen_causal_real"
BIDIRECTIONAL = "flashattn283_varlen_bidirectional_real"


def _custom(causal=True):
    return {
        "q": [
            [[1, 0], [0, 1], [1, 1], [1, -1]],
            [[0, 1], [1, 0], [-1, 1], [1, 1]],
            [[1, 1], [1, 0], [0, 1], [-1, 0]],
            [[-1, 1], [0, -1], [1, 0], [1, 1]],
            [[1, -1], [1, 1], [0, -1], [-1, 0]],
        ],
        "k": [
            [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [1, -1]],
            [[-1, 0], [0, -1]], [[0, -1], [1, 1]],
        ],
        "v": [
            [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],
            [[13, 14], [15, 16]], [[17, 18], [19, 20]],
        ],
        "cu_seqlens_q": [0, 2, 5],
        "cu_seqlens_k": [0, 3, 5],
        "max_seqlen_q": 3,
        "max_seqlen_k": 3,
        "dropout_p": 0,
        "softmax_scale": "1/2",
        "causal": causal,
        "window_size": [-1, -1],
        "softcap": 0,
        "alibi_slopes": None,
        "deterministic": True,
        "return_attn_probs": False,
        "block_table": None,
        "dtype": "f16",
    }


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _independent(inp):
    tq, hq, dim = len(inp["q"]), len(inp["q"][0]), len(inp["q"][0][0])
    hkv = len(inp["k"][0])
    group = hq // hkv
    scale = 1 / flint.arb(dim).sqrt() if inp["softmax_scale"] is None else _arb(inp["softmax_scale"])
    output = [[[flint.arb(0) for _ in range(dim)] for _ in range(hq)] for _ in range(tq)]
    for sequence in range(len(inp["cu_seqlens_q"]) - 1):
        q_start, q_stop = inp["cu_seqlens_q"][sequence:sequence + 2]
        k_start, k_stop = inp["cu_seqlens_k"][sequence:sequence + 2]
        q_len, k_len = q_stop - q_start, k_stop - k_start
        for i, token in enumerate(range(q_start, q_stop)):
            for head in range(hq):
                kv_head = head // group
                keys = list(range(k_len))
                if inp["causal"]:
                    keys = [j for j in keys if j <= i + k_len - q_len]
                if not keys:
                    continue
                scores = []
                for j in keys:
                    score = sum(
                        (_arb(inp["q"][token][head][d]) * _arb(inp["k"][k_start + j][kv_head][d])
                         for d in range(dim)),
                        flint.arb(0),
                    ) * scale
                    scores.append(score)
                weights = [score.exp() for score in scores]
                total = sum(weights, flint.arb(0))
                for d in range(dim):
                    output[token][head][d] = sum(
                        (weight * _arb(inp["v"][k_start + j][kv_head][d])
                         for weight, j in zip(weights, keys)),
                        flint.arb(0),
                    ) / total
    return output


def _visit_enclosure(test, actual, independent):
    if isinstance(independent, list):
        test.assertEqual(len(actual), len(independent))
        for left, right in zip(actual, independent):
            _visit_enclosure(test, left, right)
        return
    lo, hi = ends(actual)
    if lo == hi:
        test.assertTrue(independent.contains(_arb(lo)))
    else:
        test.assertLessEqual(lo, _fraction(independent.lower()))
        test.assertGreaterEqual(hi, _fraction(independent.upper()))


def _midpoint(value):
    if isinstance(value, list):
        return [_midpoint(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


def _torch_dense(inp):
    q = torch.tensor(inp["q"], dtype=torch.float64)
    k = torch.tensor(inp["k"], dtype=torch.float64)
    v = torch.tensor(inp["v"], dtype=torch.float64)
    out = torch.zeros_like(q)
    group = q.shape[1] // k.shape[1]
    scale = q.shape[-1] ** -0.5 if inp["softmax_scale"] is None else float(F(inp["softmax_scale"]))
    for sequence in range(len(inp["cu_seqlens_q"]) - 1):
        q_start, q_stop = inp["cu_seqlens_q"][sequence:sequence + 2]
        k_start, k_stop = inp["cu_seqlens_k"][sequence:sequence + 2]
        q_len, k_len = q_stop - q_start, k_stop - k_start
        for i, token in enumerate(range(q_start, q_stop)):
            keys = list(range(k_len))
            if inp["causal"]:
                keys = [j for j in keys if j <= i + k_len - q_len]
            if not keys:
                continue
            for head in range(q.shape[1]):
                kv_head = head // group
                index = torch.tensor([k_start + j for j in keys], dtype=torch.long)
                scores = k[index, kv_head] @ q[token, head] * scale
                out[token, head] = torch.softmax(scores, dim=0) @ v[index, kv_head]
    return out


class IndependentPackedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = registry.load("PACK")
        cls.adapter = cls.contract.adapter

    def test_reference_encloses_independent_768_bit_element_loop(self):
        cases = [(item["profile"], copy.deepcopy(item["input"]))
                 for item in self.contract.data["examples"][:8]]
        cases += [(CAUSAL, _custom(True)), (BIDIRECTIONAL, _custom(False))]
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, (profile, inp) in enumerate(cases):
            with self.subTest(case=index, profile=profile):
                before = copy.deepcopy(inp)
                actual = self.adapter.reference(profile, inp).fields["out"]
                _visit_enclosure(self, actual, _independent(inp))
                self.assertEqual(inp, before)

    def test_dense_pytorch_formula_checks_every_coordinate(self):
        cases = [(CAUSAL, _custom(True)), (BIDIRECTIONAL, _custom(False))]
        cases.append((BIDIRECTIONAL, copy.deepcopy(self.contract.data["examples"][6]["input"])))
        for profile, inp in cases:
            with self.subTest(profile=profile, default_scale=inp["softmax_scale"] is None):
                expected = torch.tensor(_midpoint(self.adapter.reference(profile, inp).fields["out"]),
                                        dtype=torch.float64)
                torch.testing.assert_close(_torch_dense(inp), expected, rtol=1e-12, atol=1e-12)

    def test_complete_output_checker_requires_dtype_shape_and_every_coordinate(self):
        inp = copy.deepcopy(self.contract.data["examples"][0]["input"])
        candidate = {"out": {"t": "f16", "shape": [3, 1, 1], "v": [[[1]], [[2]], [[3]]]}}
        self.assertTrue(self.adapter.check_output(CAUSAL, inp, candidate))
        self.assertFalse(self.adapter.check_output(CAUSAL, inp, {}))
        self.assertFalse(self.adapter.check_output(CAUSAL, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(
            CAUSAL, inp, {"out": {**candidate["out"], "t": "bf16"}}))
        self.assertFalse(self.adapter.check_output(
            CAUSAL, inp, {"out": {**candidate["out"], "shape": [3, 1]}}))
        wrong = copy.deepcopy(candidate)
        wrong["out"]["v"][2][0][0] = 4
        self.assertFalse(self.adapter.check_output(CAUSAL, inp, wrong))

    def test_paths_outside_reviewed_forward_are_refused(self):
        base = _custom(True)
        changes = (
            ("dtype", "f32", "unsupported"),
            ("causal", False, "unsupported"),
            ("window_size", [1, -1], "unsupported"),
            ("dropout_p", "1/10", "unsupported"),
            ("softcap", 1, "unsupported"),
            ("alibi_slopes", [1, 2, 3, 4], "unsupported"),
            ("return_attn_probs", True, "unsupported"),
            ("block_table", [[0]], "unsupported"),
            ("deterministic", 1, "invalid_input"),
            ("max_seqlen_q", 4, "invalid_input"),
            ("cu_seqlens_k", [0, 0, 5], "invalid_input"),
        )
        for field, value, outcome in changes:
            with self.subTest(field=field):
                result = evaluate(self.adapter.reference, CAUSAL, dict(base, **{field: value}))
                self.assertEqual(result.outcome, outcome, result.note)


if __name__ == "__main__":
    unittest.main()
