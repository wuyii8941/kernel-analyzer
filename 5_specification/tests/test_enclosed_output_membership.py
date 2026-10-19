"""An enclosure can reject a point, but containment alone cannot prove equality."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import compare, registry
from kernel_spec.num import NotEstablished, ends, hull


class EnclosedOutput(unittest.TestCase):
    def test_exact_boolean_outputs_are_compared_as_booleans(self):
        self.assertTrue(compare.check_enclosed_output([True, False], [True, False]))
        self.assertFalse(compare.check_enclosed_output([True, False], [True, True]))
        self.assertFalse(compare.check_enclosed_output([True], [1]))

    def test_exact_infinities_are_compared_without_finite_substitutes(self):
        for reference, candidate in (("-inf", float("-inf")), ("+inf", "inf")):
            self.assertTrue(compare.check_enclosed_output([reference], [candidate]))
        for reference, candidate in (("-inf", 0), ("+inf", "-inf"), (0, float("inf")),
                                     (0, float("nan")), ("-inf", float("nan"))):
            self.assertFalse(compare.check_enclosed_output([reference], [candidate]))

    def test_unsupported_coordinate_does_not_hide_a_later_contradiction(self):
        for reference, candidate in (([{0, 1}, 2], [0, 3]), ([1, 2], [hull(0, 2), 3])):
            self.assertFalse(compare.check_enclosed_output(reference, candidate))
        with self.assertRaises(NotEstablished):
            compare.check_enclosed_output([1, 2], [hull(0, 2), 2])

    def test_inside_nonpoint_interval_is_not_equality(self):
        with self.assertRaises(NotEstablished):
            compare.check_enclosed_output({"x": hull(0, 2)}, {"x": 1})
        self.assertFalse(compare.check_enclosed_output({"x": hull(0, 2)}, {"x": 3}))
        self.assertTrue(compare.check_enclosed_output({"x": hull(1, 1)}, {"x": 1}))

    def test_later_contradiction_is_not_hidden_by_overlap(self):
        target = {"loss": hull(0, 2), "gradient": [Fraction(0), Fraction(1)]}
        self.assertFalse(compare.check_enclosed_output(target, {"loss": 1, "gradient": [0, 2]}))
        self.assertFalse(compare.check_enclosed_output(target, {"loss": 1}))
        self.assertFalse(compare.check_enclosed_output(target, {"loss": 1, "gradient": [0]}))

    def test_exact_float_input_is_not_replaced_by_decimal_intent(self):
        self.assertTrue(compare.check_enclosed_output({"x": Fraction(0.1)}, {"x": 0.1}))
        self.assertFalse(compare.check_enclosed_output({"x": Fraction(1, 10)}, {"x": 0.1}))

    def test_allowed_sets_are_not_compared_as_independent_intervals(self):
        with self.assertRaises(NotEstablished):
            compare.check_enclosed_output({"indices": [{0, 1}, {0, 1}]}, {"indices": [0, 0]})

    def test_ce_reduction_none_checks_every_loss_and_gradient_row(self):
        adapter = registry.load("CE").adapter
        inp = dict(logits=[[0, 0], [1, -1]], target=[0, -100], weights=None, eps=0,
                   ignore_index=-100, reduction="none", upstream=[2, 7])
        result = adapter.reference("index_unsmoothed", inp)
        low, high = ends(result.fields["loss"][0])
        candidate = {"loss": [(low + high) / 2, 0], "gradient": [[-1, 1], [0, 0]]}
        with self.assertRaises(NotEstablished):
            adapter.check_output("index_unsmoothed", inp, candidate)
        self.assertFalse(adapter.check_output("index_unsmoothed", inp,
                                             dict(candidate, gradient=[[-1, 1], [0, 1]])))
        self.assertFalse(adapter.check_output("index_unsmoothed", inp, dict(candidate, loss=[candidate["loss"][0]])))

    def test_actual_ce_does_not_certify_an_interval_midpoint(self):
        adapter = registry.load("CE").adapter
        inp = dict(logits=[[0, 0]], target=[0], weights=None, eps=0,
                   ignore_index=-100, reduction="sum", upstream=1)
        result = adapter.reference("index_unsmoothed", inp)
        low, high = ends(result.fields["loss"])
        self.assertLess(low, high)
        candidate = {"loss": (low + high) / 2, "gradient": [[Fraction(-1, 2), Fraction(1, 2)]]}
        with patch.object(registry, "activation", return_value=(True, [], {})):
            self.assertEqual(registry.check_output("CE", "index_unsmoothed", inp, candidate)[0], "not_established")
            bad = dict(candidate, gradient=[[0, 0]])
            self.assertEqual(registry.check_output("CE", "index_unsmoothed", inp, bad)[0], "contradiction")

    def test_ce_all_ignored_sum_has_exact_zero_output(self):
        adapter = registry.load("CE").adapter
        inp = dict(logits=[[1, -1]], target=[-100], weights=None, eps=0,
                   ignore_index=-100, reduction="sum", upstream=7)
        self.assertTrue(adapter.check_output("index_unsmoothed", inp, {"loss": 0, "gradient": [[0, 0]]}))


if __name__ == "__main__":
    unittest.main()
