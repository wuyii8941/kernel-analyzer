"""AdamW enclosure checks using rational square-root bounds, independently of Decimal/mpmath."""
import copy
import itertools
import math
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry, v24
from kernel_spec.num import ends, evaluate, from_interval


class AdamWIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("ADAMW").adapter

    def inputs(self):
        return dict(state=dict(theta=[Q(3, 2), -2], m=[Q(1, 3), -1], v=[Q(2, 3), 2], t=4),
                    gradient=[Q(2, 5), -3], lr=Q(1, 7), beta1=Q(2, 3), beta2=Q(3, 4),
                    eps=Q(1, 10), weight_decay=Q(1, 9), maximize=False, amsgrad=False)

    def test_reference_contains_independent_integer_sqrt_bounds(self):
        for time, beta2, lr, eps in itertools.product((0, 4), (Q(0), Q(3, 4)), (Q(0), Q(1, 7)), (Q(0), Q(1, 10))):
            inp = self.inputs()
            inp.update(beta2=beta2, lr=lr, eps=eps)
            inp["state"]["t"] = time
            saved = copy.deepcopy(inp)
            result = self.adapter.reference("base", inp).fields
            self.assertEqual(inp, saved)
            self.assertEqual(result["t"], time + 1)
            for i, gradient in enumerate(inp["gradient"]):
                state, b1 = inp["state"], inp["beta1"]
                moment = b1 * state["m"][i] + (1 - b1) * gradient
                variance = beta2 * state["v"][i] + (1 - beta2) * gradient**2
                self.assertEqual(result["m"][i], moment)
                self.assertEqual(result["v"][i], variance)
                vh = variance / (1 - beta2**(time + 1))
                scale = 2**512
                root_floor = math.isqrt(vh.numerator * scale**2 // vh.denominator)
                lo, hi = Q(root_floor, scale), Q(root_floor + 1, scale)
                if lo * lo == vh:
                    hi = lo
                self.assertLessEqual(lo * lo, vh)
                self.assertGreaterEqual(hi * hi, vh)
                numerator = lr * moment / (1 - b1**(time + 1))
                decay = state["theta"][i] * (1 - lr * inp["weight_decay"])
                targets = [decay - numerator / (root + eps) for root in (lo, hi)]
                lower, upper = ends(result["theta"][i])
                self.assertLessEqual(lower, min(targets))
                self.assertGreaterEqual(upper, max(targets))

    def test_missing_inputs_and_nonmapping_inputs_are_invalid(self):
        complete = self.inputs()
        for key in complete:
            inp = {k: v for k, v in complete.items() if k != key}
            with self.subTest(missing=key):
                self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "invalid_input")
        for inp in (None, [], "task"):
            self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "invalid_input")

    def test_backend_exponent_limits_are_not_mathematical_undefined(self):
        backend = v24.module("rigorous")
        original_context = backend._ctx
        def limited_context(*args, **kwargs):
            context = original_context(*args, **kwargs)
            context.Emax, context.Emin = 5, -5
            return context
        for theta in (Q(10**8), Q(1, 10**8)):
            inp = self.inputs()
            inp["state"]["theta"][0] = theta
            saved = copy.deepcopy(inp)
            with patch.object(backend, "_ctx", side_effect=limited_context):
                result = evaluate(self.adapter.reference, "base", inp)
            self.assertEqual(result.outcome, "not_established", result.note)
            self.assertEqual(inp, saved)

    def test_legacy_refusal_and_unexpected_errors_stay_distinct(self):
        backend = v24.module("rigorous")
        for error, expected in ((backend.SpecNotEstablished("precision exhausted"), "not_established"),
                                (ValueError("precision exhausted"), "implementation_error"),
                                (ZeroDivisionError("unexpected internal division"), "implementation_error")):
            with patch.object(self.adapter.opt, "adam_step", side_effect=error):
                result = evaluate(self.adapter.reference, "base", self.inputs())
            self.assertEqual(result.outcome, expected)
        inp = self.inputs()
        inp.update(beta2=Q(0), eps=Q(0), gradient=[Q(0), Q(0)])
        self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "domain_undefined")

    def test_explicit_algorithm_options_are_not_ignored(self):
        for option, value, expected in (("decoupled", False, "unsupported"),
                                        ("differentiable", True, "unsupported"),
                                        ("decoupled", 1, "invalid_input"),
                                        ("differentiable", 0, "invalid_input"),
                                        ("unknown_option", True, "unsupported")):
            inp = self.inputs()
            inp[option] = value
            with self.subTest(option=option, value=value):
                self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, expected)
        inp = self.inputs()
        inp.update(decoupled=True, differentiable=False)
        self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "value")

    def test_interval_prestate_remains_explicitly_unsupported(self):
        for field in ("theta", "m", "v"):
            for interval in (from_interval(Q(1), Q(2)), self.adapter.interval_backend.Interval.exact(Q(1, 3))):
                inp = self.inputs()
                inp["state"][field][0] = interval
                with self.subTest(field=field):
                    self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "unsupported")

    def test_none_gradient_preserves_full_state_without_mutation(self):
        inp = self.inputs()
        inp["gradient"] = None
        saved = copy.deepcopy(inp)
        self.assertEqual(self.adapter.reference("base", inp).fields, inp["state"])
        self.assertEqual(inp, saved)
