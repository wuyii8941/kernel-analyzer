"""Decimal interval powers must enclose exact rational extrema at every ambient precision."""
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import v24

Interval = v24.module("spec_optimizers").Interval


class DecimalPowers(unittest.TestCase):
    def test_rational_square_does_not_collapse_to_ambient_precision(self):
        for precision in (6, 28, 100):
            with localcontext() as context:
                context.prec = precision
                out = Interval.exact(Q(1, 3)) ** 2
                self.assertLessEqual(Q(out.lo), Q(1, 9))
                self.assertGreaterEqual(Q(out.hi), Q(1, 9))
                self.assertEqual(context.prec, precision)

    def test_powers_enclose_exact_extrema(self):
        for low, high in (("-1.23456789123456789", "2.34567891234567891"),
                          ("-2.34567891234567891", "-1.23456789123456789"),
                          ("0", "0"), ("0.33333333333333333333333333333333333", "0.4")):
            lo, hi = Decimal(low), Decimal(high)
            for power in range(2, 7):
                values = [Q(lo) ** power, Q(hi) ** power]
                if power % 2 == 0 and lo <= 0 <= hi:
                    values.append(Q(0))
                for precision in (6, 28, 100):
                    with localcontext() as context:
                        context.prec = precision
                        out = Interval(lo, hi) ** power
                        self.assertLessEqual(Q(out.lo), min(values))
                        self.assertGreaterEqual(Q(out.hi), max(values))
                        if power % 2 == 0:
                            self.assertGreaterEqual(out.lo, 0)

    def test_square_underflow_is_outward_not_zero(self):
        out = Interval(Decimal("1e-600000")) ** 2
        self.assertEqual(out.lo, 0)
        self.assertGreater(out.hi, 0)
        self.assertLessEqual(out.hi.adjusted(), -999999)
