import importlib.util
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "ARITH.py"
    spec = importlib.util.spec_from_file_location("reviewed_arith_independent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def q(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def nested(array):
    if not isinstance(array, np.ndarray):
        return q(array)
    if array.ndim == 0:
        return q(array.item())
    return [nested(x) for x in array]


def encoded(array):
    return nested(np.asarray(array, dtype=object))


def record(array, dtype="f64"):
    return {"t": dtype, "v": encoded(np.asarray(array, dtype=object))}


def normalize(value):
    if isinstance(value, list):
        return [normalize(x) for x in value]
    return q(value)


class ArithmeticIndependentEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def result(self, op, args, kwargs=None):
        call = {"op": op, "args": args}
        if kwargs:
            call["kwargs"] = kwargs
        return normalize(self.adapter.reference("real", call).fields["out"])

    def test_random_broadcast_arithmetic_matches_separate_fraction_formulas(self):
        rng = random.Random(20261018)
        for _ in range(160):
            left = np.asarray([[Fraction(rng.randrange(-16, 17), 4)] for _ in range(2)], dtype=object)
            right = np.asarray([[Fraction(rng.randrange(-16, 17), 4) for _ in range(3)]], dtype=object)
            right_nonzero = np.vectorize(lambda x: x if x else Fraction(1, 2), otypes=[object])(right)
            alpha = Fraction(rng.choice((-4, -2, -1, 1, 2, 4)), 2)
            a, b = record(left), record(right)
            cases = (
                ("aten.add", [a, b], {"alpha": str(alpha)}, left + alpha * right),
                ("aten.sub", [a, b], {"alpha": str(alpha)}, left - alpha * right),
                ("aten.rsub", [a, b], {"alpha": str(alpha)}, right - alpha * left),
                ("aten.mul", [a, b], None, left * right),
                ("aten.div", [a, record(right_nonzero)], None, left / right_nonzero),
            )
            for op, args, kwargs, expected in cases:
                with self.subTest(op=op, left=left.tolist(), right=right.tolist(), alpha=alpha):
                    self.assertEqual(self.result(op, args, kwargs), encoded(expected))

    def test_division_rounding_uses_the_real_quotient(self):
        numerators = np.asarray([[7, -7, 1], [-1, 5, -5]], dtype=object)
        denominators = np.asarray([[2, 2, -2]], dtype=object)
        quotient = np.vectorize(lambda x, y: q(x) / q(y), otypes=[object])(numerators, denominators)
        trunc = np.vectorize(lambda x: math.trunc(x), otypes=[object])(quotient)
        floor = np.vectorize(lambda x: math.floor(x), otypes=[object])(quotient)
        args = [record(numerators), record(denominators)]
        self.assertEqual(self.result("aten.div", args, {"rounding_mode": "trunc"}), encoded(trunc))
        self.assertEqual(self.result("aten.div", args, {"rounding_mode": "floor"}), encoded(floor))
        self.assertEqual(self.result("aten.floor_divide", args), encoded(floor))

    def test_unary_clamp_extrema_selection_and_fma_match_separate_formulas(self):
        x = np.asarray([Fraction(-5, 2), -1, 0, Fraction(3, 2), 4], dtype=object)
        xr = record(x)
        self.assertEqual(self.result("aten.neg", [xr]), encoded(-x))
        self.assertEqual(self.result("aten.abs", [xr]), encoded(np.vectorize(abs, otypes=[object])(x)))
        self.assertEqual(self.result("aten.sign", [xr]), [-1, -1, 0, 1, 1])
        self.assertEqual(self.result("aten.floor", [xr]), [-3, -1, 0, 1, 4])
        self.assertEqual(self.result("aten.ceil", [xr]), [-2, -1, 0, 2, 4])
        clamped = np.vectorize(lambda v: min(max(v, Fraction(-1)), Fraction(3, 2)),
                               otypes=[object])(x)
        self.assertEqual(self.result("aten.clamp", [xr], {"min": -1, "max": "3/2"}), encoded(clamped))

        y = np.asarray([2, -2, 3, 1, 0], dtype=object)
        self.assertEqual(self.result("aten.maximum", [xr, record(y)]),
                         encoded(np.vectorize(max, otypes=[object])(x, y)))
        self.assertEqual(self.result("aten.minimum", [xr, record(y)]),
                         encoded(np.vectorize(min, otypes=[object])(x, y)))
        condition = np.asarray([True, False, True, False, True], dtype=object)
        cond = {"t": "b8", "v": condition.tolist()}
        self.assertEqual(self.result("aten.where", [cond, xr, record(y)]),
                         encoded(np.where(condition.astype(bool), x, y)))
        self.assertEqual(self.result("aten.masked_fill", [xr, cond, "1/2"]),
                         encoded(np.where(condition.astype(bool), Fraction(1, 2), x)))

        a = np.asarray([Fraction(3, 2), -2, 0], dtype=object)
        b = np.asarray([2, Fraction(-1, 2), 4], dtype=object)
        c = np.asarray([Fraction(1, 2), 3, -1], dtype=object)
        self.assertEqual(self.result("prims.fma", [record(a), record(b), record(c)]), encoded(a * b + c))

    def test_public_torch_calls_match_exact_formula_after_output_rounding(self):
        dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16,
                     "f32": torch.float32, "f64": torch.float64}
        for name, dtype in dtype_map.items():
            x = torch.tensor([[-2.0], [1.5]], dtype=dtype)
            y = torch.tensor([[0.5, 2.0, -1.0]], dtype=dtype)
            mask = torch.tensor([[True, False, True], [False, True, False]])
            calls = (
                ("aten.add", [record(x.tolist(), name), record(y.tolist(), name)], {"alpha": "1/2"},
                 torch.add(x, y, alpha=0.5)),
                ("aten.sub", [record(x.tolist(), name), record(y.tolist(), name)], None, torch.sub(x, y)),
                ("aten.mul", [record(x.tolist(), name), record(y.tolist(), name)], None, torch.mul(x, y)),
                ("aten.div", [record(x.tolist(), name), record(y.tolist(), name)], None, torch.div(x, y)),
                ("aten.maximum", [record(x.tolist(), name), record(y.tolist(), name)], None, torch.maximum(x, y)),
                ("aten.where", [{"t": "b8", "v": mask.tolist()}, record(x.tolist(), name),
                                record(y.tolist(), name)], None, torch.where(mask, x, y)),
            )
            for op, args, kwargs, actual in calls:
                with self.subTest(dtype=name, op=op):
                    exact = np.asarray(self.result(op, args, kwargs), dtype=object)
                    expected = torch.tensor([[float(q(v)) for v in row] for row in exact.tolist()], dtype=dtype)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                    self.assertTrue(torch.equal(actual, expected), (name, op, actual, expected))


if __name__ == "__main__":
    unittest.main()
