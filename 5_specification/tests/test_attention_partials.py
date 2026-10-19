"""Partial attention identities, checked without the full attention reference."""
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import attention_partials as P
from kernel_spec.num import Invalid, Undefined, from_interval
from kernel_spec.real import R


class PartialAttentionTests(unittest.TestCase):
    def test_empty_identity_and_complete_coordinates(self):
        p = P.scaled(0, 3, [1, 5, -2])
        for parts in ([p, P.empty(3)], [P.empty(3), p], [P.empty(3), p, P.empty(3)]):
            result = P.merge(parts)
            self.assertEqual(result, p)
            self.assertEqual(P.finish(result)['out'], [F(1, 3), F(5, 3), F(-2, 3)])

    def test_scaled_numerator_is_not_normalized_output(self):
        # Three equally weighted values: [2,7], [4,-1], [9,3].
        p = P.scaled(0, 2, [6, 6])
        q = P.normalized([9, 3], 0)
        result = P.finish(P.merge([p, q]))
        self.assertEqual(result['out'], [5, 3])

    def test_empty_policy_and_infinity_boundary_are_explicit(self):
        for sentinel in ('+inf', '-inf'):
            p = P.normalized([0, 0], sentinel, is_empty=True)
            self.assertEqual(p, P.empty(2))
            with self.assertRaises(Invalid):
                P.normalized([0, 0], sentinel)
        with self.assertRaises(Undefined):
            P.finish(P.merge([P.empty(2), P.empty(2)]))
        self.assertEqual(P.finish(P.empty(2), empty_policy='zero'), {'out': [0, 0], 'lse': '-inf'})

    def test_malformed_empty_or_late_part_is_not_hidden(self):
        good = P.scaled(0, 1, [1, 2])
        for bad in (P.empty(1), {'shift': None, 'mass': 0, 'numerator': [0, 1]},
                    {'shift': None, 'mass': 0, 'numerator': [0, 0], 'extra': 1},
                    {'shift': 0, 'mass': 0, 'numerator': [0, 0]}):
            with self.assertRaises(Invalid):
                P.merge([good, bad])
        for parts in ([], None, 'parts'):
            with self.assertRaises(Invalid):
                P.merge(parts)
        with self.assertRaises(Invalid):
            P.normalized([0, True], '+inf', is_empty=True)
        with self.assertRaises(Invalid):
            P.normalized([0], 0, is_empty=1)

    def test_independent_exponential_sum_and_interval_inputs(self):
        import flint
        saved = flint.ctx.prec
        try:
            flint.ctx.prec = 512
            # Independent unshifted sums, including negative values and unequal masses.
            masses = [flint.arb(-2).exp()*2, flint.arb(3).exp()*3]
            expected = [(masses[0]*a + masses[1]*b)/sum(masses)
                        for a, b in ((2, -4), (-1, 7), (5, 3))]
            result = P.finish(P.merge([P.scaled(-2, 2, [4, -2, 10]),
                                       P.scaled(3, 3, [-12, 21, 9])]))
            def endpoint(x):
                m, e = x.man_exp()
                return F(int(m))*F(2)**int(e)
            for actual, target in zip(result['out'] + [result['lse']],
                                      expected + [sum(masses).log()]):
                lo, hi = R(actual).bounds()
                self.assertLessEqual(lo, endpoint(target.lower()))
                self.assertGreaterEqual(hi, endpoint(target.upper()))
            # Equal unknown shifts cancel mathematically. Enclosure must retain 1/3.
            p = P.scaled(from_interval(0, F(1, 100)), 3, [1])
            lo, hi = R(P.finish(P.merge([p, P.empty(1)]))['out'][0]).bounds()
            self.assertLessEqual(lo, F(1, 3))
            self.assertGreaterEqual(hi, F(1, 3))
        finally:
            flint.ctx.prec = saved


if __name__ == '__main__':
    unittest.main()
