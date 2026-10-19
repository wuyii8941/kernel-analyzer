"""Independent exact checks for finite round-to-nearest-even conversion.

Expected results below use only Fraction arithmetic and integer bit lengths.
They do not import the production rounding oracle or CAST adapter helpers.
"""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry

FORMAT = {
    "f16": (11, -14, 15),
    "bf16": (8, -126, 127),
    "f32": (24, -126, 127),
    "f64": (53, -1022, 1023),
}


def floor_log2(value):
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Q(2) ** exponent > value:
        exponent -= 1
    return exponent


def rne(value, dtype):
    if value == 0:
        return Q(0)
    sign, magnitude = (-1, -value) if value < 0 else (1, value)
    precision, emin, emax = FORMAT[dtype]
    exponent = max(floor_log2(magnitude), emin)
    step = Q(2) ** (exponent - precision + 1)
    quotient, remainder = divmod(magnitude, step)
    twice = 2 * remainder
    if twice > step or (twice == step and quotient % 2):
        quotient += 1
    rounded = quotient * step
    maximum = (2 - Q(2) ** (1 - precision)) * Q(2) ** emax
    if rounded > maximum:
        raise OverflowError(dtype)
    return sign * rounded


def plain(result):
    return list(result.fields["out"])


class IndependentCast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = staticmethod(registry.load("CAST").adapter.reference)

    def convert(self, values, source, target):
        inp = {"op": "prims.convert_element_type",
               "args": [{"t": source, "v": [str(v) for v in values]}, target]}
        return self.reference("finite_rne", inp)

    def test_normal_midpoints_neighbors_and_signs(self):
        for target in ("f16", "bf16", "f32"):
            precision = FORMAT[target][0]
            ulp = Q(2) ** (1 - precision)
            points = [
                Q(1), 1 + ulp / 4, 1 + ulp / 2, 1 + 3 * ulp / 4,
                1 + ulp, 1 + 3 * ulp / 2, 1 + 2 * ulp,
            ]
            points += [-value for value in points]
            with self.subTest(target=target):
                actual = plain(self.convert(points, "f64", target))
                self.assertEqual(actual, [rne(value, target) for value in points])

    def test_subnormal_edges_for_each_narrow_target(self):
        for target in ("f16", "bf16", "f32"):
            precision, emin, _ = FORMAT[target]
            smallest = Q(2) ** (emin - precision + 1)
            points = [0, smallest / 2, 3 * smallest / 4, smallest,
                      3 * smallest / 2, 2 * smallest, -smallest / 2, -3 * smallest / 2]
            with self.subTest(target=target):
                actual = plain(self.convert(points, "f64", target))
                self.assertEqual(actual, [rne(Q(value), target) for value in points])

    def test_widening_and_identity_preserve_source_values(self):
        for source in ("f16", "bf16", "f32", "f64"):
            precision, emin, _ = FORMAT[source]
            values = [Q(0), Q(-3, 2), Q(1) + Q(2) ** (1 - precision),
                      Q(2) ** (emin - precision + 1)]
            for target in ("f16", "bf16", "f32", "f64"):
                if FORMAT[target][0] < precision or FORMAT[target][1] > emin:
                    continue
                with self.subTest(source=source, target=target):
                    self.assertEqual(plain(self.convert(values, source, target)), values)

    def test_format_boundaries_and_adjacent_values(self):
        for target in ("f16", "bf16", "f32"):
            precision, emin, emax = FORMAT[target]
            minimum_normal = Q(2) ** emin
            smallest = Q(2) ** (emin - precision + 1)
            maximum = (2 - Q(2) ** (1 - precision)) * Q(2) ** emax
            values = [minimum_normal - smallest, minimum_normal, minimum_normal + smallest, maximum]
            with self.subTest(target=target):
                self.assertEqual(plain(self.convert(values, "f64", target)), values)


if __name__ == "__main__":
    unittest.main()
