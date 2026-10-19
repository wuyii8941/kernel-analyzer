"""Independent exact and public PyTorch checks for vLLM LoRA shrink."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends, evaluate


PROFILE = "vllm016_shrink_f32out_real"


def _nested(shape, fn, prefix=()):
    if not shape:
        return fn(prefix)
    return [_nested(shape[1:], fn, prefix + (index,)) for index in range(shape[0])]


def _case(seed, *, dtype="f16", all_inactive=False):
    tokens, hidden, slices, loras, rank = 4, 5, 2, 3, 3
    values = (-2, -1, "-1/2", 0, "1/2", 1, 2)
    pick = lambda index, shift=0: values[(sum((axis + 1) * value for axis, value in enumerate(index)) + seed + shift) % len(values)]
    mapping = [-1] * tokens if all_inactive else [-1, seed % loras, (seed + 1) % loras, (seed + 2) % loras]
    return {
        "x": _nested((tokens, hidden), lambda index: pick(index)),
        "weights": _nested((slices, loras, rank, hidden), lambda index: pick(index, 2)),
        "mapping": mapping,
        "scaling": ("1/2", -1, 2)[seed % 3],
        "out_before": _nested((slices, tokens, rank), lambda index: pick(index, 4)),
        "input_dtype": dtype,
        "output_dtype": "f32",
    }


CASES = tuple(_case(index, dtype="bf16" if index % 2 else "f16") for index in range(10)) + (
    _case(10, all_inactive=True),
    _case(11, dtype="bf16", all_inactive=True),
)


def _independent(inp):
    if all(index == -1 for index in inp["mapping"]):
        return copy.deepcopy(inp["out_before"])
    slices = len(inp["weights"])
    tokens, rank = len(inp["x"]), len(inp["weights"][0][0])
    hidden = len(inp["x"][0])
    output = [[[F(0) for _ in range(rank)] for _ in range(tokens)] for _ in range(slices)]
    for slice_index in range(slices):
        for token in range(tokens):
            lora = inp["mapping"][token]
            if lora < 0:
                continue
            for rank_index in range(rank):
                output[slice_index][token][rank_index] = F(inp["scaling"]) * sum(
                    (F(inp["weights"][slice_index][lora][rank_index][hidden_index])
                     * F(inp["x"][token][hidden_index]) for hidden_index in range(hidden)),
                    F(0),
                )
    return output


def _assert_exact(test, actual, expected):
    if isinstance(expected, list):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            _assert_exact(test, left, right)
        return
    lo, hi = ends(actual)
    test.assertEqual((lo, hi), (F(expected), F(expected)))


def _midpoint(value):
    if isinstance(value, list):
        return [_midpoint(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


def _torch_formula(inp):
    dtype = torch.float16 if inp["input_dtype"] == "f16" else torch.bfloat16
    x = torch.tensor([[float(F(item)) for item in row] for row in inp["x"]], dtype=dtype)
    weights = torch.tensor(
        [[[[float(F(item)) for item in row] for row in lora] for lora in weight_slice]
         for weight_slice in inp["weights"]], dtype=dtype)
    before = torch.tensor(
        [[[float(F(item)) for item in row] for row in output_slice]
         for output_slice in inp["out_before"]], dtype=torch.float32)
    x_saved, weights_saved, before_saved = x.clone(), weights.clone(), before.clone()
    if all(index == -1 for index in inp["mapping"]):
        output = before.clone()
    else:
        output = torch.zeros_like(before)
        for token, lora in enumerate(inp["mapping"]):
            if lora >= 0:
                output[:, token] = torch.einsum(
                    "srk,k->sr", weights[:, lora].float(), x[token].float()
                ) * float(F(inp["scaling"]))
    return output, (x, x_saved), (weights, weights_saved), (before, before_saved)


class IndependentLoRAShrink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("LORA").adapter

    def test_reference_matches_independent_fraction_element_loop(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                reference = self.adapter.reference(PROFILE, inp).fields["y"]
                _assert_exact(self, reference, _independent(inp))
                self.assertEqual(inp, before)

    def test_public_pytorch_formula_matches_every_output_coordinate(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                output, *unchanged = _torch_formula(inp)
                reference = torch.tensor(
                    _midpoint(self.adapter.reference(PROFILE, inp).fields["y"]), dtype=torch.float32)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                for value, saved in unchanged:
                    self.assertTrue(torch.equal(value, saved))

    def test_complete_output_checker_requires_dtype_shape_and_every_coordinate(self):
        inp = CASES[0]
        expected = _midpoint(self.adapter.reference(PROFILE, inp).fields["y"])
        candidate = {"y": {"t": "f32", "shape": [2, 4, 3], "v": expected}}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, candidate))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"y": {**candidate["y"], "t": "f16"}}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {"y": {**candidate["y"], "shape": [8, 3]}}))
        wrong = copy.deepcopy(candidate)
        wrong["y"]["v"][1][3][2] += 1
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong))

    def test_active_and_all_inactive_calls_have_different_output_state_rules(self):
        active, inactive = CASES[0], CASES[-1]
        active_y = _midpoint(self.adapter.reference(PROFILE, active).fields["y"])
        self.assertEqual(active_y[0][0], [0.0, 0.0, 0.0])
        self.assertNotEqual(active_y[0][0], [float(F(item)) for item in active["out_before"][0][0]])
        inactive_y = _midpoint(self.adapter.reference(PROFILE, inactive).fields["y"])
        self.assertEqual(inactive_y, [[[float(F(item)) for item in row] for row in output_slice]
                                      for output_slice in inactive["out_before"]])

    def test_values_outside_the_reviewed_release_subset_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, input_dtype="f32")).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, output_dtype="f16")).outcome, "unsupported")
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, mapping=[0, 1, 2, 3])).outcome, "invalid_input")
        bad = copy.deepcopy(base)
        bad["x"][0][0] = "1/3"
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, bad).outcome, "invalid_input")


if __name__ == "__main__":
    unittest.main()
