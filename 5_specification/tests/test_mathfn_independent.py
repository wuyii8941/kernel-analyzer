import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path

import flint

from kernel_spec.num import ends


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "MATHFN.py"
    spec = importlib.util.spec_from_file_location("reviewed_mathfn_independent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arb_fraction(value):
    value = Fraction(value)
    return flint.arb(int(value.numerator)) / flint.arb(int(value.denominator))


def exact_endpoint(value):
    mantissa, exponent = value.man_exp()
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def arb_bounds(ball):
    return exact_endpoint(ball.lower()), exact_endpoint(ball.upper())


def independent(op, x):
    a = arb_fraction(x)
    table = {
        "aten.exp": lambda: a.exp(),
        "aten.log": lambda: a.log(),
        "aten.log1p": lambda: a.log1p(),
        "aten.expm1": lambda: a.expm1(),
        "aten.sqrt": lambda: a.sqrt(),
        "aten.rsqrt": lambda: 1 / a.sqrt(),
        "aten.sin": lambda: a.sin(),
        "aten.cos": lambda: a.cos(),
        "aten.tanh": lambda: a.tanh(),
        "aten.sigmoid": lambda: 1 / (1 + (-a).exp()),
    }
    return table[op]()


class MathFunctionIndependentEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def test_reference_encloses_independent_arb_results(self):
        cases = {
            "aten.exp": (Fraction(-3, 2), 0, Fraction(7, 3)),
            "aten.log": (Fraction(1, 4), 1, 3),
            "aten.log1p": (Fraction(-1, 2), 0, Fraction(3, 2)),
            "aten.expm1": (Fraction(-3, 2), 0, Fraction(7, 3)),
            "aten.sqrt": (0, Fraction(1, 4), 2),
            "aten.rsqrt": (Fraction(1, 4), 2, 9),
            "aten.sin": (-2, Fraction(1, 3), 2),
            "aten.cos": (-2, Fraction(1, 3), 2),
            "aten.tanh": (-2, Fraction(1, 3), 2),
            "aten.sigmoid": (-2, Fraction(1, 3), 2),
        }
        saved = flint.ctx.prec
        flint.ctx.prec = 768
        try:
            for op, values in cases.items():
                for x in values:
                    with self.subTest(op=op, x=x):
                        encoded = str(x) if isinstance(x, Fraction) and x.denominator != 1 else int(x)
                        actual = self.adapter.reference(
                            "real", {"op": op, "args": [{"t": "f64", "v": [encoded]}]}).fields["out"][0]
                        lo, hi = ends(actual)
                        independent_lo, independent_hi = arb_bounds(independent(op, x))
                        if lo == hi:
                            self.assertLessEqual(independent_lo, lo)
                            self.assertGreaterEqual(independent_hi, hi)
                        else:
                            self.assertLessEqual(lo, independent_lo)
                            self.assertGreaterEqual(hi, independent_hi)
        finally:
            flint.ctx.prec = saved

    def test_erf_has_a_separate_rational_series_check(self):
        # test_reference_guarantees.ErfGuarantees compares the same evaluator
        # against a rational integral series. This assertion keeps the reviewed
        # MATHFN adapter on that evaluator rather than a point approximation.
        from kernel_spec import real
        result = self.adapter.reference(
            "real", {"op": "aten.erf", "args": [{"t": "f64", "v": ["1/2"]}]}).fields["out"][0]
        self.assertEqual(ends(result), real.erf(Fraction(1, 2)).bounds())


if __name__ == "__main__":
    unittest.main()
