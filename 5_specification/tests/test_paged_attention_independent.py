"""Independent high-precision and public dense checks for reviewed paged attention."""
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


PROFILE = "vllm016_causal_2d_real"


def _custom(dtype="f16", window=(2, -1), softcap=1):
    return {
        "query": [
            [[1, 0], [0, 1], [1, 1], [1, -1]],
            [[0, 1], [1, 0], [-1, 1], [1, 1]],
            [[1, 1], [1, 0], [0, 1], [-1, 0]],
        ],
        "cu_seqlens_q": [0, 2, 3],
        "seq_lens": [4, 3],
        # Physical blocks 2,0 form sequence 0; blocks 3,1 form sequence 1.
        "key_cache": [
            [[[1, 1], [1, -1]], [[-1, 1], [0, 1]]],
            [[[1, 0], [0, -1]], [[0, 0], [0, 0]]],
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[-1, 0], [1, 1]], [[0, -1], [1, 0]]],
        ],
        "value_cache": [
            [[[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            [[[17, 18], [19, 20]], [[0, 0], [0, 0]]],
            [[[1, 2], [3, 4]], [[-1, 0], [2, 1]]],
            [[[4, 3], [2, 1]], [[8, 7], [6, 5]]],
        ],
        "block_table": [[2, 0], [3, 1]],
        "scale": "1/2",
        "causal": True,
        "window": list(window),
        "softcap": softcap,
        "dtype": dtype,
    }


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _logical(inp, sequence):
    block_size = len(inp["key_cache"][0])
    keys, values = [], []
    for position in range(inp["seq_lens"][sequence]):
        block = inp["block_table"][sequence][position // block_size]
        slot = position % block_size
        keys.append(inp["key_cache"][block][slot])
        values.append(inp["value_cache"][block][slot])
    return keys, values


def _independent(inp):
    hq, dim = len(inp["query"][0]), len(inp["query"][0][0])
    hkv = len(inp["key_cache"][0][0])
    group = hq // hkv
    output = [[[None for _ in range(dim)] for _ in range(hq)] for _ in inp["query"]]
    for sequence, seq_len in enumerate(inp["seq_lens"]):
        start, stop = inp["cu_seqlens_q"][sequence:sequence + 2]
        query_len = stop - start
        context = seq_len - query_len
        keys, values = _logical(inp, sequence)
        for local, token in enumerate(range(start, stop)):
            position = context + local
            for head in range(hq):
                kv_head = head // group
                scores, columns = [], []
                for key_position in range(seq_len):
                    if key_position > position:
                        continue
                    left = inp["window"][0]
                    if left >= 0 and position - key_position > left:
                        continue
                    score = sum(
                        (_arb(inp["query"][token][head][coordinate])
                         * _arb(keys[key_position][kv_head][coordinate]) for coordinate in range(dim)),
                        flint.arb(0),
                    ) * _arb(inp["scale"])
                    if F(inp["softcap"]) > 0:
                        cap = _arb(inp["softcap"])
                        score = cap * (score / cap).tanh()
                    scores.append(score)
                    columns.append(key_position)
                weights = [score.exp() for score in scores]
                total = sum(weights, flint.arb(0))
                for coordinate in range(dim):
                    output[token][head][coordinate] = sum(
                        (weight * _arb(values[key_position][kv_head][coordinate])
                         for weight, key_position in zip(weights, columns)),
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
    query = torch.tensor([[[float(F(x)) for x in row] for row in token]
                          for token in inp["query"]], dtype=torch.float64)
    hq, dim = query.shape[1:]
    hkv = len(inp["key_cache"][0][0])
    group = hq // hkv
    output = torch.empty_like(query)
    for sequence, seq_len in enumerate(inp["seq_lens"]):
        start, stop = inp["cu_seqlens_q"][sequence:sequence + 2]
        context = seq_len - (stop - start)
        keys_raw, values_raw = _logical(inp, sequence)
        keys = torch.tensor(keys_raw, dtype=torch.float64)
        values = torch.tensor(values_raw, dtype=torch.float64)
        for local, token in enumerate(range(start, stop)):
            position = context + local
            first = 0 if inp["window"][0] < 0 else max(0, position - inp["window"][0])
            for head in range(hq):
                kv_head = head // group
                scores = keys[first:position + 1, kv_head] @ query[token, head]
                scores *= float(F(inp["scale"]))
                if F(inp["softcap"]) > 0:
                    cap = float(F(inp["softcap"]))
                    scores = cap * torch.tanh(scores / cap)
                output[token, head] = torch.softmax(scores, dim=0) @ values[first:position + 1, kv_head]
    return output


class IndependentPagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = registry.load("PAGEDATTN")
        cls.adapter = cls.contract.adapter

    def test_reference_encloses_independent_768_bit_element_loop(self):
        cases = [copy.deepcopy(item["input"]) for item in self.contract.data["examples"][:4]]
        cases += [_custom(), _custom("bf16", (-1, -1), 0), _custom("f32", (0, -1), 2)]
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(cases):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                actual = self.adapter.reference(PROFILE, inp).fields["out"]
                _visit_enclosure(self, actual, _independent(inp))
                self.assertEqual(inp, before)

    def test_public_dense_pytorch_formula_checks_all_coordinates(self):
        for index, inp in enumerate((_custom(), _custom("bf16", (-1, -1), 0), _custom("f32", (0, -1), 2))):
            with self.subTest(case=index):
                expected = torch.tensor(_midpoint(self.adapter.reference(PROFILE, inp).fields["out"]),
                                        dtype=torch.float64)
                torch.testing.assert_close(_torch_dense(inp), expected, rtol=1e-12, atol=1e-12)

    def test_complete_output_checker_requires_dtype_shape_and_every_coordinate(self):
        inp = copy.deepcopy(self.contract.data["examples"][0]["input"])
        candidate = {"out": {"t": "f16", "shape": [2, 1, 1], "v": [[['5/2']], [['7/2']]]}}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, candidate))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": {**candidate["out"], "t": "f32"}}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": {**candidate["out"], "shape": [12, 2]}}))
        wrong = copy.deepcopy(candidate)
        wrong["out"]["v"][1][0][0] = '9/2'
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong))

    def test_inputs_outside_the_reviewed_release_path_are_refused(self):
        base = _custom()
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, causal=False)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, window=[2, 0])).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, dtype="f64")).outcome, "unsupported")
        bad = copy.deepcopy(base)
        bad["query"][0][0][0] = "1/3"
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, bad).outcome, "invalid_input")
        bad = copy.deepcopy(base)
        bad["block_table"][1][1] = 9
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, bad).outcome, "invalid_input")


if __name__ == "__main__":
    unittest.main()
