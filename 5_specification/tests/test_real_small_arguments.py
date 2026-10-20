"""kernel_spec.real enclosures must contain the true value for tiny arguments.

Found by path A package D (2026-10-21): with mpmath 1.3.0, iv.exp at 256 bits returns point intervals that exclude
exp(x) for |x| between about 2^-271 and 2^-55 (exp(2^-100) came back 8.2e-92 too low); exp, expm1, tanh, sigmoid and
log1p inherited it. The oracle is python-flint Arb at 600 bits.
"""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import flint

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC)]
from kernel_spec import real as rl  # noqa: E402
from kernel_spec.real import R  # noqa: E402


def arb_of(q):
    return flint.arb(q.numerator) / flint.arb(q.denominator)


ORACLE = {
    "exp": lambda a: a.exp(),
    "expm1": lambda a: a.exp() - 1,
    "tanh": lambda a: a.tanh(),
    "sigmoid": lambda a: 1 / (1 + (-a).exp()),
    "log1p": lambda a: (1 + a).log(),
    "log": lambda a: a.log(),
}


class SmallArguments(unittest.TestCase):
    def check(self, name, x):
        old = flint.ctx.prec
        flint.ctx.prec = 600
        try:
            truth = ORACLE[name](arb_of(x))
            lo, hi = getattr(rl, name)(R(x)).bounds()
            self.assertFalse(arb_of(lo) > truth, f"{name}({x}) lower bound above the true value")
            self.assertFalse(arb_of(hi) < truth, f"{name}({x}) upper bound below the true value")
        finally:
            flint.ctx.prec = old

    def test_tiny_arguments_are_enclosed(self):
        for k in (20, 40, 55, 56, 60, 100, 149, 200, 270, 272, 300):
            for sign in (1, -1):
                x = Fraction(sign, 2 ** k)
                for name in ("exp", "expm1", "tanh", "sigmoid", "log1p"):
                    with self.subTest(name=name, x=f"{sign}*2^-{k}"):
                        self.check(name, x)

    def test_near_one_logarithm(self):
        for k in (20, 55, 100, 200):
            for x in (1 + Fraction(1, 2 ** k), 1 - Fraction(1, 2 ** k)):
                with self.subTest(x=str(x)[:30]):
                    self.check("log", x)

    def test_ordinary_arguments_unchanged_in_meaning(self):
        for x in (Fraction(1, 3), Fraction(-7, 2), Fraction(20), Fraction(-20), Fraction(1, 10)):
            for name in ("exp", "tanh", "sigmoid", "log1p", "expm1"):
                if name == "log1p" and x <= -1:
                    continue          # outside the domain (Undefined), not an enclosure question
                with self.subTest(name=name, x=str(x)):
                    self.check(name, x)


if __name__ == "__main__":
    unittest.main()
