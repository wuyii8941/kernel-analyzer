"""Closed-form uniform-logit CE values with an independent rational logarithm remainder."""
import copy
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends, evaluate


def log_integer_bounds(value):
    # ln(value) = 2*atanh((value-1)/(value+1)); positive series tail bounded geometrically.
    y = Q(value - 1, value + 1)
    power, total = y, Q(0)
    terms = 400
    for k in range(terms):
        total += 2 * power / (2*k + 1)
        power *= y*y
    tail = 2 * power / ((2*terms + 1) * (1 - y*y))
    return total, total + tail


class CEIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("CE").adapter

    def inputs(self):
        return dict(logits=[[7, 7, 7], [-3, -3, -3], [0, 0, 0]], target=[1, -7, 0], weights=[0, 2, 3],
                    ignore_index=-7, eps=Q(0), reduction="mean", upstream=Q(-3, 2))

    def test_loss_and_complete_vjp_against_closed_form(self):
        log_lo, log_hi = log_integer_bounds(3)
        for profile in ("index_unsmoothed", "index_smoothing_unweighted", "index_smoothing_R_A", "probability_unsmoothed"):
            for reduction in ("none", "sum", "mean"):
                inp = self.inputs()
                inp["reduction"] = reduction
                if reduction == "none":
                    inp["upstream"] = [Q(-3, 2), Q(0), Q(2)]
                if "smoothing" in profile:
                    inp["eps"] = Q(2, 5)
                if profile == "index_smoothing_unweighted":
                    inp["weights"] = None
                if profile == "probability_unsmoothed":
                    inp["target"] = [[Q(1, 3)] * 3, [Q(0), Q(1), Q(0)], [Q(1, 4), Q(1, 4), Q(1, 2)]]
                weights = inp["weights"] if inp["weights"] is not None else [Q(1)] * 3
                coefficients, mass = [], Q(0)
                for target in inp["target"]:
                    if isinstance(target, list):
                        target_distribution = target
                        mass += 1
                    elif target == -7:
                        coefficients.append([Q(0)] * 3)
                        continue
                    else:
                        target_distribution = [inp["eps"]/3 + (1-inp["eps"])*(c == target) for c in range(3)]
                        mass += weights[target]
                    coefficients.append([w*t for w, t in zip(weights, target_distribution)])
                divisor = mass if reduction == "mean" else Q(1)
                saved = copy.deepcopy(inp)
                result = self.adapter.reference(profile, inp).fields
                self.assertEqual(inp, saved)
                row_factors = [sum(row)/divisor for row in coefficients]
                factors = row_factors if reduction == "none" else [sum(row_factors)]
                losses = result["loss"] if reduction == "none" else [result["loss"]]
                self.assertEqual(len(losses), len(factors))
                for loss, factor in zip(losses, factors):
                    lo, hi = ends(loss)
                    self.assertLessEqual(lo, factor*log_lo)
                    self.assertGreaterEqual(hi, factor*log_hi)
                self.assertEqual(len(result["gradient"]), 3)
                for n, row in enumerate(coefficients):
                    self.assertEqual(len(result["gradient"][n]), 3)
                    upstream = inp["upstream"][n] if reduction == "none" else inp["upstream"]
                    for c, coefficient in enumerate(row):
                        expected = (sum(row)/3 - coefficient)*upstream/divisor
                        lo, hi = ends(result["gradient"][n][c])
                        self.assertLessEqual(lo, expected)
                        self.assertGreaterEqual(hi, expected)

    def test_missing_inputs_are_invalid(self):
        complete = self.inputs()
        for key in ("logits", "target", "ignore_index", "eps", "reduction", "upstream"):
            inp = {k: v for k, v in complete.items() if k != key}
            self.assertEqual(evaluate(self.adapter.reference, "index_unsmoothed", inp).outcome, "invalid_input")
        for inp in (None, [], "task"):
            self.assertEqual(evaluate(self.adapter.reference, "index_unsmoothed", inp).outcome, "invalid_input")

    def test_undeclared_task_options_are_not_ignored(self):
        for option, value in (("size_average", True), ("reduce", False), ("reading", "R_B")):
            inp = self.inputs()
            inp[option] = value
            self.assertEqual(evaluate(self.adapter.reference, "index_unsmoothed", inp).outcome, "unsupported")

    def test_zero_mean_mass_is_decided_before_transcendental_evaluation(self):
        for profile, weights, target, eps in (
                ("index_unsmoothed", [0, 2, 3], [0, 0, -7], Q(0)),
                ("index_smoothing_R_A", [0, 2, 3], [0, 0, -7], Q(1, 2)),
                ("index_smoothing_unweighted", None, [-7, -7, -7], Q(1, 2))):
            inp = self.inputs()
            inp.update(weights=weights, target=target, eps=eps)
            with patch.object(self.adapter.ce, "cross_entropy", return_value={"established": False, "reason": "budget"}) as backend:
                for function in (self.adapter.reference, self.adapter.reading_R_B):
                    result = evaluate(function, profile, inp)
                    self.assertEqual(result.outcome, "domain_undefined")
                backend.assert_not_called()
            inp["upstream"] = [1, 2, 3]
            self.assertEqual(evaluate(self.adapter.reference, profile, inp).outcome, "invalid_input")

    def test_zero_weights_do_not_make_sum_or_probability_mean_undefined(self):
        inp = self.inputs()
        inp.update(weights=[0, 0, 0], reduction="sum")
        for profile in ("index_unsmoothed", "probability_unsmoothed"):
            if profile == "probability_unsmoothed":
                inp.update(target=[[Q(1, 3)]*3]*3, reduction="mean")
            result = self.adapter.reference(profile, inp)
            self.assertEqual(ends(result.fields["loss"]), (0, 0))
            for row in result.fields["gradient"]:
                for value in row:
                    self.assertEqual(ends(value), (0, 0))

    def test_nonuniform_binary_case_against_independent_logistic_formula(self):
        import flint
        def arb(q):
            q = Q(q)
            return flint.arb(q.numerator) / flint.arb(q.denominator)
        def endpoint(value):
            mantissa, exponent = value.man_exp()
            return Q(int(mantissa)) * Q(2)**int(exponent)
        def enclosed(reference, independent):
            low, high = ends(reference)
            self.assertLessEqual(low, endpoint(independent.lower()))
            self.assertGreaterEqual(high, endpoint(independent.upper()))
        saved_precision = flint.ctx.prec
        try:
            flint.ctx.prec = 512
            for profile in ("index_unsmoothed", "index_smoothing_unweighted", "index_smoothing_R_A", "probability_unsmoothed"):
                for reduction in ("sum", "none", "mean"):
                    for difference in (Q(-5, 3), Q(30)):
                        inp = dict(logits=[[10000003+difference, 10000003], [-2, 1]],
                                   target=[0, 1], weights=[Q(2, 3), Q(5, 2)], eps=Q(0),
                                   ignore_index=-100, reduction=reduction,
                                   upstream=[Q(-2), Q(3)] if reduction == "none" else Q(-2))
                        if "smoothing" in profile:
                            inp["eps"] = Q(1, 2)
                        if profile == "index_smoothing_unweighted":
                            inp["weights"] = None
                        if profile == "probability_unsmoothed":
                            inp["target"] = [[Q(1, 3), Q(2, 3)], [Q(1, 4), Q(3, 4)]]
                        weights = inp["weights"] or [Q(1), Q(1)]
                        losses, gradients, mass = [], [], Q(0)
                        for row, target in zip(inp["logits"], inp["target"]):
                            if isinstance(target, list):
                                distribution = target
                                mass += 1
                            else:
                                distribution = [inp["eps"]/2 + (1-inp["eps"])*(c == target) for c in range(2)]
                                mass += weights[target]
                            a, b = [arb(w*t) for w, t in zip(weights, distribution)]
                            ratio = arb(row[0] - row[1]).exp()
                            losses.append(a*(1+1/ratio).log() + b*(1+ratio).log())
                            dx = (b*ratio-a)/(1+ratio)
                            gradients.append([dx, -dx])
                        divisor = arb(mass if reduction == "mean" else 1)
                        result = self.adapter.reference(profile, inp).fields
                        if reduction == "none":
                            for actual, expected in zip(result["loss"], losses):
                                enclosed(actual, expected)
                        else:
                            enclosed(result["loss"], sum(losses)/divisor)
                        for n, row in enumerate(gradients):
                            upstream = inp["upstream"][n] if reduction == "none" else inp["upstream"]
                            for c, expected in enumerate(row):
                                enclosed(result["gradient"][n][c], expected*arb(upstream)/divisor)
        finally:
            flint.ctx.prec = saved_precision

    def test_api_valid_ranks_outside_profile_are_not_called_invalid(self):
        for logits, target in (([1, 2, 3], 1), ([[[1, 2], [3, 4], [5, 6]]], [[1, 0]]),
                               ([[[], []]], [[]])):
            inp = self.inputs()
            inp.update(logits=logits, target=target)
            self.assertEqual(evaluate(self.adapter.reference, "index_unsmoothed", inp).outcome, "unsupported")
        for logits in ([[1, 2], [3]], [[1, 2], 3], [], [[], []]):
            inp = self.inputs()
            inp["logits"] = logits
            self.assertEqual(evaluate(self.adapter.reference, "index_unsmoothed", inp).outcome, "invalid_input")
