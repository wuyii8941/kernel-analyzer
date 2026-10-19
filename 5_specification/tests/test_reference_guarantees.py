"""Checks of mathematical guarantees, independent of contract example counts."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from mpmath import iv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec.expr import iv_erf
from kernel_spec.num import ends, from_interval, to_iv
from kernel_spec import real as rl
from kernel_spec.num import NotEstablished, Undefined


class SqrtGuarantees(unittest.TestCase):
    def test_nonnegative_interval_including_zero_has_a_finite_enclosure(self):
        for upper in (Fraction(1), Fraction(2), Fraction(1, 3), Fraction(10**100)):
            with self.subTest(upper=upper):
                result = rl.sqrt(rl.R(from_interval(0, upper)))
                lo, hi = result.bounds()
                self.assertEqual(lo, 0)
                self.assertGreaterEqual(hi * hi, upper)

    def test_sqrt_of_squared_uncertainty_keeps_zero_in_domain(self):
        x = rl.R(from_interval(-Fraction(1, 3), Fraction(1, 2)))
        lo, hi = rl.sqrt(x ** 2).bounds()
        self.assertEqual(lo, 0)
        self.assertGreaterEqual(hi, Fraction(1, 2))

    def test_negative_domain_and_uncertain_domain_remain_distinct(self):
        for a, b in ((-2, -1), (-1, -1)):
            with self.assertRaises(Undefined):
                rl.sqrt(rl.R(from_interval(a, b)))
        for a, b in ((-1, 0), (-1, 2)):
            with self.assertRaises(NotEstablished):
                rl.sqrt(rl.R(from_interval(a, b)))
        self.assertEqual(rl.sqrt(rl.R(from_interval(0, 0))).v, 0)


def erf_series(x):
    """Independent integral series with a rigorous alternating-series remainder.

    erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1)/(n! (2n+1)).
    For |x| <= 2, after n >= 4 the term magnitudes decrease to zero.
    All series arithmetic is rational; only the final pi factor is enclosed.
    """
    assert abs(x) <= 2
    term, total = x, Fraction(0)
    for n in range(320):
        total += term
        term *= -x*x*Fraction(2*n+1, (n+1)*(2*n+3))
    a, b = sorted((total, total + term))
    return from_interval(a, b) * 2 / iv.sqrt(iv.pi)


class ErfGuarantees(unittest.TestCase):
    def test_contains_independent_integral_series(self):
        saved = iv.prec
        try:
            for x in map(Fraction, ("-2", "-1", "-1/2", "0", "1/2", "1", "2")):
                with self.subTest(x=x):
                    iv.prec = 256
                    lo, hi = ends(iv_erf(to_iv(x)))
                    iv.prec = 768
                    a, b = ends(erf_series(x))
                    self.assertLessEqual(lo, a)
                    self.assertGreaterEqual(hi, b)
        finally:
            iv.prec = saved

    def test_interval_endpoints_and_precision_restoration(self):
        import flint
        saved = flint.ctx.prec
        lo, hi = ends(iv_erf(from_interval(Fraction(-1, 2), Fraction(1))))
        self.assertEqual(flint.ctx.prec, saved)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        a, _ = ends(iv_erf(to_iv(Fraction(-1, 2))))
        _, b = ends(iv_erf(to_iv(1)))
        self.assertEqual((lo, hi), (a, b))

    def test_far_tail_and_zero_do_not_use_point_approximation(self):
        from unittest.mock import patch
        # The old implementation silently depended on this unbounded point oracle.
        with patch("mpmath.erf", side_effect=AssertionError("point oracle used")):
            self.assertEqual(ends(iv_erf(to_iv(0))), (0, 0))
            for sign in (-1, 1):
                lo, hi = ends(iv_erf(to_iv(sign * 10**30)))
                self.assertLessEqual(-1, lo)
                self.assertLessEqual(lo, hi)
                self.assertLessEqual(hi, 1)
                self.assertLess(hi-lo, Fraction(1, 10**50))
                self.assertLess(lo, 1) if sign > 0 else self.assertGreater(hi, -1)


if __name__ == "__main__":
    unittest.main()
