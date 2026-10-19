"""Independent high-precision and public PyTorch checks for the reviewed BSR softmax values task."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, ends, evaluate


PROFILE = "torch210_bsr_softmax_values_real"
DTYPES = (
    (torch.float16, "f16", 2e-3),
    (torch.bfloat16, "bf16", 2e-2),
    (torch.float32, "f32", 2e-6),
)


def _task(shape, blocksize, crow, col, values, dtype="f32"):
    return {"shape": shape, "blocksize": blocksize, "crow": crow, "col": col,
            "values": values, "dtype": dtype}


CASES = (
    _task([2, 4], [1, 2], [0, 2, 3], [0, 1, 1], [[[0, 1]], [[1, 0]], [[5, 5]]]),
    _task([2, 2], [2, 2], [0, 1], [0], [[[0, 1], [2, 0]]]),
    _task([4, 8], [2, 2], [0, 2, 2], [0, 3], [[[-1, 2], [3, 0]], [[1, -2], [0, 4]]]),
    _task([2, 6], [2, 1], [0, 3], [0, 2, 5], [[[0], [2]], [[1], [0]], [[-2], [3]]]),
    _task([2, 4], [1, 2], [0, 1, 2], [1, 0], [[[3, 3]], [[-1, -1]]]),
)


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _independent(inp):
    rows, columns = inp["blocksize"]
    result = [[[flint.arb(0) for _ in range(columns)] for _ in range(rows)] for _ in inp["col"]]
    for block_row in range(inp["shape"][0] // rows):
        entries = list(range(inp["crow"][block_row], inp["crow"][block_row + 1]))
        for row in range(rows):
            positions = [(entry, column) for entry in entries for column in range(columns)]
            if not positions:
                continue
            xs = [_arb(inp["values"][entry][row][column]) for entry, column in positions]
            maximum = max(xs, key=lambda value: value.mid())
            exponentials = [(value - maximum).exp() for value in xs]
            total = sum(exponentials, flint.arb(0))
            for (entry, column), value in zip(positions, exponentials):
                result[entry][row][column] = value / total
    return result


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


def _exact_midpoint(value):
    if isinstance(value, list):
        return [_exact_midpoint(item) for item in value]
    lo, hi = ends(value)
    return (lo + hi) / 2


def _torch_formula(values, crow):
    result = torch.empty_like(values)
    rows, columns = values.shape[1:]
    for block_row in range(len(crow) - 1):
        start, end = crow[block_row], crow[block_row + 1]
        if start == end:
            continue
        row_values = values[start:end].transpose(0, 1).reshape(rows, -1)
        row_result = torch.softmax(row_values, dim=-1, dtype=torch.float32).to(values.dtype)
        result[start:end] = row_result.reshape(rows, end - start, columns).transpose(0, 1)
    return result


class IndependentBsrSoftmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("BSR").adapter

    def test_reference_encloses_independent_768_bit_formula(self):
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                reference = self.adapter.reference(PROFILE, inp).fields["out"]
                _visit_enclosure(self, reference, _independent(inp))
                self.assertEqual(inp, before)

    def test_fifteen_public_pytorch_formula_calls_follow_the_same_task(self):
        checked = 0
        for torch_dtype, dtype_name, tolerance in DTYPES:
            for index, template in enumerate(CASES):
                with self.subTest(dtype=str(torch_dtype), case=index):
                    values = torch.tensor(template["values"], dtype=torch_dtype)
                    before = values.clone()
                    public = _torch_formula(values, template["crow"])
                    inp = dict(template, values=values.tolist(), dtype=dtype_name)
                    reference = self.adapter.reference(PROFILE, inp).fields["out"]
                    torch.testing.assert_close(
                        public.double(), torch.tensor(_midpoint(reference), dtype=torch.float64),
                        rtol=tolerance, atol=tolerance,
                    )
                    self.assertTrue(torch.equal(values, before))
                    checked += 1
        self.assertEqual(checked, 15)

    def test_complete_output_checker_enforces_field_dtype_and_shape(self):
        inp = CASES[0]
        reference = self.adapter.reference(PROFILE, inp).fields["out"]
        candidate = {"out": {"t": "f32", "shape": [3, 1, 2], "v": _exact_midpoint(reference)}}
        with self.assertRaises(NotEstablished):
            self.adapter.check_output(PROFILE, inp, candidate)
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"out": {**candidate["out"], "t": "f16"}}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"out": {**candidate["out"], "shape": [3, 2]}}))
        wrong = copy.deepcopy(candidate)
        wrong["out"]["v"][0][0][0] = 99
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong))

    def test_other_sparse_tasks_special_values_and_noncanonical_patterns_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, op="sampled_addmm")).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, dtype="f64")).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE,
            dict(base, values=[[['inf', 1]], [[1, 0]], [[5, 5]]])).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, col=[0, 0, 1])).outcome, "invalid_input")


if __name__ == "__main__":
    unittest.main()
