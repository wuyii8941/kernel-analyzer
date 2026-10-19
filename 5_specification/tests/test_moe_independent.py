"""Independent high-precision and public PyTorch checks for Mixtral top-2 routing."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, ends, evaluate


PROFILE = "mixtral_top2_real"
DTYPES = (
    (torch.float16, "f16", 4e-3),
    (torch.bfloat16, "bf16", 3e-2),
    (torch.float32, "f32", 2e-6),
    # Transformers explicitly computes routing softmax in float32 even when hidden states are float64.
    (torch.float64, "f64", 2e-6),
)


def _task(logits, experts, dtype="f64"):
    return {"router_logits": logits, "expert_outputs": experts, "dtype": dtype}


CASES = (
    _task([[1, 0, -1]], [[[1, 0], [0, 1], [9, 9]]]),
    _task([[0, 2, 1], [3, 0, 2]], [[[100], [2], [4]], [[5], [100], [1]]]),
    _task([[2, 1, -8, 0]], [[[1, 2, 3], [3, 5, 7], [100, 100, 100], [9, 11, 13]]]),
    _task([[-4, -1, -3]], [[[2, -2], [4, 1], [8, 3]]]),
    _task([[-1, 1], [4, 2]], [[[2, 3], [5, 7]], [[11, 13], [17, 19]]]),
)


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _independent(inp):
    selected, dense_weights, outputs = [], [], []
    for logits, experts in zip(inp["router_logits"], inp["expert_outputs"]):
        order = sorted(range(len(logits)), key=lambda expert: F(logits[expert]), reverse=True)
        chosen = order[:2]
        exponentials = [_arb(logits[expert]).exp() for expert in chosen]
        total = exponentials[0] + exponentials[1]
        pair_weights = [value / total for value in exponentials]
        selected.append([expert in chosen for expert in range(len(logits))])
        dense = [flint.arb(0) for _ in logits]
        for expert, weight in zip(chosen, pair_weights):
            dense[expert] = weight
        dense_weights.append(dense)
        outputs.append([
            sum((dense[expert] * _arb(experts[expert][coordinate]) for expert in chosen), flint.arb(0))
            for coordinate in range(len(experts[0]))
        ])
    return selected, dense_weights, outputs


def _visit_enclosure(test, actual, independent):
    if isinstance(independent, list):
        test.assertEqual(len(actual), len(independent))
        for left, right in zip(actual, independent):
            _visit_enclosure(test, left, right)
        return
    lo, hi = ends(actual)
    if lo == hi:
        test.assertTrue(independent.contains(_arb(lo)))
    else:
        test.assertLessEqual(lo, _fraction(independent.lower()))
        test.assertGreaterEqual(hi, _fraction(independent.upper()))


def _midpoint(value):
    if isinstance(value, list):
        return [_midpoint(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


def _exact_midpoint(value):
    if isinstance(value, list):
        return [_exact_midpoint(item) for item in value]
    lo, hi = ends(value)
    return (lo + hi) / 2


class IndependentMixtralRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("MOE").adapter

    def test_reference_encloses_independent_high_precision_formula(self):
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                reference = self.adapter.reference(PROFILE, inp).fields
                selected, weights, out = _independent(inp)
                self.assertEqual(reference["selected"], selected)
                _visit_enclosure(self, reference["weights"], weights)
                _visit_enclosure(self, reference["out"], out)
                self.assertEqual(inp, before)

    def test_twenty_public_pytorch_formula_calls_follow_the_same_task(self):
        checked = 0
        for torch_dtype, dtype_name, tolerance in DTYPES:
            for index, template in enumerate(CASES):
                with self.subTest(dtype=str(torch_dtype), case=index):
                    logits = torch.tensor(template["router_logits"], dtype=torch_dtype)
                    experts = torch.tensor(template["expert_outputs"], dtype=torch_dtype)
                    logits_before, experts_before = logits.clone(), experts.clone()
                    probabilities = torch.softmax(logits, dim=1, dtype=torch.float32)
                    selected_probabilities, indices = torch.topk(probabilities, 2, dim=-1)
                    selected_probabilities = selected_probabilities / selected_probabilities.sum(dim=-1, keepdim=True)
                    selected_probabilities = selected_probabilities.to(torch_dtype)
                    selected = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
                    weights = torch.zeros_like(logits).scatter(1, indices, selected_probabilities)
                    out = (weights.unsqueeze(-1) * experts).sum(dim=1)
                    inp = _task(logits.tolist(), experts.tolist(), dtype_name)
                    reference = self.adapter.reference(PROFILE, inp).fields
                    self.assertTrue(torch.equal(selected, torch.tensor(reference["selected"], dtype=torch.bool)))
                    torch.testing.assert_close(
                        weights.double(), torch.tensor(_midpoint(reference["weights"]), dtype=torch.float64),
                        rtol=tolerance, atol=tolerance,
                    )
                    torch.testing.assert_close(
                        out.double(), torch.tensor(_midpoint(reference["out"]), dtype=torch.float64),
                        rtol=tolerance, atol=tolerance,
                    )
                    self.assertTrue(torch.equal(logits, logits_before))
                    self.assertTrue(torch.equal(experts, experts_before))
                    checked += 1
        self.assertEqual(checked, 20)

    def test_complete_output_checker_enforces_fields_dtype_shape_and_boolean_values(self):
        inp = _task([[1, 0, -1]], [[[1], [2], [3]]], "f32")
        reference = self.adapter.reference(PROFILE, inp).fields
        candidate = {
            "selected": {"t": "b8", "shape": [1, 3], "v": [[True, True, False]]},
            "weights": {"t": "f32", "shape": [1, 3], "v": _exact_midpoint(reference["weights"])},
            "out": {"t": "f32", "shape": [1, 1], "v": _exact_midpoint(reference["out"])},
        }
        with self.assertRaises(NotEstablished):
            self.adapter.check_output(PROFILE, inp, candidate)
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"selected": candidate["selected"]}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {**candidate, "selected": {**candidate["selected"], "t": "i8"}}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {**candidate, "weights": {**candidate["weights"], "shape": [3]}}))
        wrong = copy.deepcopy(candidate)
        wrong["out"]["v"] = [[99]]
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong))

    def test_other_moe_tasks_and_uncertain_ties_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, capacity=2)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, aux=True)).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, _task([[1, 1, 0]], [[[1], [2], [3]]])
        ).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, _task([["inf", 0, -1]], [[[1], [2], [3]]])
        ).outcome, "unsupported")


if __name__ == "__main__":
    unittest.main()
