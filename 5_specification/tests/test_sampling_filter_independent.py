"""Independent high-precision and public PyTorch checks for deterministic top-k/top-p filtering."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


PROFILE = "vllm016_topk_topp_keep_real"
P_VALUES = ("1/2", "5/8", "3/4", "7/8", 1)


def _case(index):
    vocabulary = 8
    base = list(range(-4, 4))
    shift0, shift1 = index % vocabulary, (3 * index + 1) % vocabulary
    row0 = base[shift0:] + base[:shift0]
    row1 = list(reversed(base[shift1:] + base[:shift1]))
    k = None if index % 3 == 0 else [1 + index % vocabulary, 1 + (index + 3) % vocabulary]
    p = None if index % 4 == 0 else [P_VALUES[index % len(P_VALUES)], P_VALUES[(index + 2) % len(P_VALUES)]]
    if k is None and p is None:
        k = [3, 5]
    return {"logits": [row0, row1], "k": k, "p": p, "dtype": "f32"}


CASES = tuple(_case(index) for index in range(20))


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _independent(inp):
    result = []
    for row_index, row in enumerate(inp["logits"]):
        order = sorted(range(len(row)), key=lambda token: F(row[token]), reverse=True)
        if inp["k"] is not None:
            order = order[:inp["k"][row_index]]
        keep = set(order)
        if inp["p"] is not None and F(inp["p"][row_index]) < 1:
            maximum = _arb(row[order[0]])
            masses = {token: (_arb(row[token]) - maximum).exp() for token in order}
            total = sum(masses.values(), flint.arb(0))
            threshold = _arb(inp["p"][row_index])
            keep, above = set(), flint.arb(0)
            for token in order:
                if (above - threshold).upper() < 0:
                    keep.add(token)
                elif (above - threshold).lower() <= 0:
                    raise AssertionError("independent comparison did not separate from the threshold")
                above += masses[token] / total
        result.append([token in keep for token in range(len(row))])
    return result


def _public_formula(inp):
    logits = torch.tensor(inp["logits"], dtype=torch.float32)
    logits_before = logits.clone()
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if inp["k"] is not None:
        k = torch.tensor(inp["k"], dtype=torch.int64)
        threshold = logits_sort.gather(1, (logits_sort.size(1) - k).unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < threshold, -float("inf"))
    if inp["p"] is not None:
        p = torch.tensor([float(F(value)) for value in inp["p"]], dtype=torch.float32)
        cumulative = torch.cumsum(torch.softmax(logits_sort, dim=-1), dim=-1)
        mask = cumulative <= 1 - p.unsqueeze(1)
        mask[:, -1] = False
        logits_sort.masked_fill_(mask, -float("inf"))
    filtered = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return torch.isfinite(filtered), logits, logits_before


class IndependentSamplingFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SAMPLING").adapter

    def test_reference_matches_independent_768_bit_formula(self):
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                self.assertEqual(self.adapter.reference(PROFILE, inp).fields["keep"], _independent(inp))
                self.assertEqual(inp, before)

    def test_twenty_public_pytorch_formula_calls_follow_the_same_task(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                public, logits, before = _public_formula(inp)
                reference = self.adapter.reference(PROFILE, inp).fields["keep"]
                self.assertTrue(torch.equal(public, torch.tensor(reference, dtype=torch.bool)))
                self.assertTrue(torch.equal(logits, before))

    def test_complete_output_checker_requires_boolean_dtype_shape_and_all_coordinates(self):
        inp = CASES[1]
        expected = self.adapter.reference(PROFILE, inp).fields["keep"]
        candidate = {"keep": {"t": "b8", "shape": [2, 8], "v": expected}}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, candidate))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"keep": {**candidate["keep"], "t": "i8"}}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"keep": {**candidate["keep"], "shape": [16]}}))
        wrong = copy.deepcopy(candidate)
        wrong["keep"]["v"][0][0] = not wrong["keep"]["v"][0][0]
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong))

    def test_ties_random_sampling_and_nonphysical_inputs_are_refused(self):
        base = CASES[1]
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, logits=[[1, 1], [2, 1]])).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, q=[[1] * 8, [1] * 8])).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, dtype="f16")).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, p=["1/3", "1/2"])).outcome, "invalid_input")


if __name__ == "__main__":
    unittest.main()
