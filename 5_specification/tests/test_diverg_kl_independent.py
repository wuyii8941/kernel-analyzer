"""Independent checks for the reviewed PyTorch 2.10 KL batchmean profile."""
from __future__ import annotations

import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch
import torch.nn.functional as torch_functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends, evaluate


PROFILE = "torch210_probability_batchmean_f32"


def _task(values, target, upstream=1):
    return {
        "input": values,
        "target": target,
        "input_dtype": "f32",
        "target_dtype": "f32",
        "reduction": "batchmean",
        "log_target": False,
        "upstream": upstream,
    }


CASES = (
    _task([[-1, -2], [0, -1]], [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]),
    _task([[0, -1]], [[1, F(1, 2)]]),
    _task([[F(1, 2), -4, 2], [1, -2, F(-1, 2)]],
          [[F(1, 4), F(1, 2), 2], [F(1, 8), 1, F(3, 2)]], upstream=2),
    _task([[0, 1, -1, 2]], [[1, 2, 4, 8]], upstream=F(1, 2)),
)


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _independent(inp):
    batch = len(inp["input"])
    total = flint.arb(0)
    grad = []
    for values, targets in zip(inp["input"], inp["target"]):
        row = []
        for value, target in zip(values, targets):
            target = F(target)
            total += _arb(target) * (_arb(target).log() - _arb(value))
            row.append(-target * F(inp["upstream"]) / batch)
        grad.append(row)
    return total / batch, grad


def _midpoint(value):
    lo, hi = ends(value)
    return float((lo + hi) / 2)


class IndependentKlBatchmean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("DIVERG").adapter

    def test_reference_encloses_independent_768_bit_loss_and_matches_exact_vjp(self):
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                result = self.adapter.reference(PROFILE, inp).fields
                loss, grad = _independent(inp)
                lo, hi = ends(result["loss"])
                self.assertLessEqual(lo, _fraction(loss.lower()))
                self.assertGreaterEqual(hi, _fraction(loss.upper()))
                self.assertEqual(result["grad"], grad)
                self.assertEqual(inp, before)

    def test_public_pytorch_function_and_autograd_follow_the_same_selected_task(self):
        generator = torch.Generator().manual_seed(20261019)
        checked = 0
        for batch, support in ((1, 7), (3, 5), (8, 16), (4, 33)):
            for upstream in (0.5, 1.0, -2.0):
                values = torch.randn((batch, support), generator=generator, dtype=torch.float32, requires_grad=True)
                target = torch.rand((batch, support), generator=generator, dtype=torch.float32) * 0.9 + 0.05
                values_before, target_before = values.detach().clone(), target.clone()
                loss = torch_functional.kl_div(values, target, reduction="batchmean", log_target=False)
                (loss * upstream).backward()
                inp = _task(values_before.tolist(), target.tolist(), upstream)
                result = self.adapter.reference(PROFILE, inp).fields
                self.assertAlmostEqual(float(loss.detach()), _midpoint(result["loss"]), delta=3e-5)
                torch.testing.assert_close(
                    values.grad.double(), torch.tensor(result["grad"], dtype=torch.float64), rtol=2e-6, atol=2e-7)
                self.assertTrue(torch.equal(values.detach(), values_before))
                self.assertTrue(torch.equal(target, target_before))
                checked += 1
        self.assertEqual(checked, 12)

    def test_complete_output_checker_requires_both_float32_outputs(self):
        inp = _task([[0, 1]], [[1, 1]], upstream=2)
        candidate = {
            "loss": {"t": "f32", "shape": [], "v": -1},
            "grad": {"t": "f32", "shape": [1, 2], "v": [[-2, -2]]},
        }
        self.assertTrue(self.adapter.check_output(PROFILE, inp, candidate))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"loss": candidate["loss"]}))
        wrong_dtype = copy.deepcopy(candidate)
        wrong_dtype["grad"]["t"] = "f16"
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_dtype))
        wrong_shape = copy.deepcopy(candidate)
        wrong_shape["grad"]["shape"] = [2]
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_shape))
        wrong_value = copy.deepcopy(candidate)
        wrong_value["grad"]["v"][0][1] = 2
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_value))

    def test_unselected_calls_and_nonphysical_records_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, reduction="mean")).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, log_target=True)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, target=[[0, 1], [1, 1]])).outcome,
                         "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, input_dtype="f16")).outcome,
                         "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, upstream=F(1, 3))).outcome,
                         "invalid_input")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, input=[["nan", 0], [0, 1]])).outcome,
                         "unsupported")


if __name__ == "__main__":
    unittest.main()
