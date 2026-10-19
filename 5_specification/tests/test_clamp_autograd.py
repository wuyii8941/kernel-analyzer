"""Independent checks for the PyTorch 2.10 public scalar-bound clamp profile."""
from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


PROFILE = "torch210_clamp_scalar_autograd"
DTYPES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def record(tensor):
    return {"t": DTYPES[tensor.dtype], "shape": list(tensor.shape), "v": tensor.tolist()}


def query(x, upstream, minimum, maximum, **changes):
    result = {"op": "torch.clamp", "input": record(x), "min": minimum,
              "max": maximum, "upstream": record(upstream)}
    result.update(changes)
    return result


def exact_nested(array):
    array = np.asarray(array, dtype=object)
    if array.ndim == 0:
        return Fraction(array.item())
    return [exact_nested(child) for child in array]


def independent_reference(x, upstream, minimum, maximum):
    values = np.asarray(x.tolist(), dtype=object)
    derivatives = np.asarray(upstream.tolist(), dtype=object)
    out = np.empty(values.shape, dtype=object)
    grad = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        value = Fraction(values[index])
        clipped = value
        if minimum is not None:
            clipped = max(clipped, Fraction(minimum))
        if maximum is not None:
            clipped = min(clipped, Fraction(maximum))
        active = (minimum is None or value >= Fraction(minimum)) and \
                 (maximum is None or value <= Fraction(maximum))
        out[index] = clipped
        grad[index] = Fraction(derivatives[index]) if active else Fraction(0)
    return {"out": exact_nested(out), "grad": exact_nested(grad)}


class ClampAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("ARITH").adapter

    def test_inclusive_boundaries_hand_witness(self):
        x = torch.tensor([-2., -1., 0., 1., 2.])
        upstream = torch.tensor([1., 2., 3., 4., 5.])
        fields = self.adapter.reference(PROFILE, query(x, upstream, -1, 1)).fields
        self.assertEqual(fields, {"out": [-1, -1, 0, 1, 1],
                                  "grad": [0, 2, 3, 4, 0]})

    def test_independent_coordinates_and_pytorch_210_autograd(self):
        cases = [
            ((2, 4), -1, 1, [-2, -1, 0, 1, 2, 3, -1, 1]),
            ((2, 3), 0, None, [-1, 0, 1, 2, -2, 0]),
            ((3, 2), None, 0, [-1, 0, 1, 2, -2, 0]),
            ((2, 3), 1, -1, [-2, -1, 0, 1, 2, 3]),
            ((2, 2), 0, 0, [-1, 0, 1, 0]),
            ((2, 2, 2), Fraction(-1, 2), Fraction(1, 2),
             [-1, Fraction(-1, 2), 0, Fraction(1, 2), 1,
              Fraction(-1, 2), Fraction(1, 2), 2]),
        ]
        for dtype, (shape, minimum, maximum, raw_values) in itertools.product(DTYPES, cases):
            with self.subTest(dtype=dtype, shape=shape, minimum=minimum, maximum=maximum):
                x = torch.tensor([float(value) for value in raw_values], dtype=dtype).reshape(shape)
                x.requires_grad_()
                upstream = ((torch.arange(x.numel()).reshape(shape) % 7) - 3).to(dtype)
                torch_minimum = None if minimum is None else float(minimum)
                torch_maximum = None if maximum is None else float(maximum)
                out = torch.clamp(x, min=torch_minimum, max=torch_maximum)
                (dx,) = torch.autograd.grad(out, x, upstream)

                expected = independent_reference(x.detach(), upstream, minimum, maximum)
                actual = self.adapter.reference(
                    PROFILE, query(x.detach(), upstream, minimum, maximum)).fields
                self.assertEqual(actual, expected)
                to_float = np.vectorize(float, otypes=[float])
                self.assertEqual(out.detach().tolist(),
                                 to_float(np.asarray(expected["out"], dtype=object)).tolist())
                self.assertEqual(dx.tolist(),
                                 to_float(np.asarray(expected["grad"], dtype=object)).tolist())

    def test_profile_rejects_other_overloads_and_incomplete_inputs(self):
        x = torch.tensor([-1., 0., 1.])
        upstream = torch.ones(3)
        valid = query(x, upstream, 0, 1)
        cases = [
            ({**valid, "op": "aten.clamp"}, Unsupported),
            ({**valid, "min": None, "max": None}, Invalid),
            ({**valid, "min": record(torch.tensor(0.))}, Unsupported),
            ({**valid, "input": {"t": "f32", "v": [-1., 0., 1.]}}, Invalid),
            ({**valid, "upstream": record(torch.ones(1))}, Invalid),
            ({**valid, "upstream": record(torch.ones(3, dtype=torch.float64))}, Invalid),
            ({**valid, "min": True}, Invalid),
            ({**valid, "extra": 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_forward_and_gradient(self):
        x = torch.tensor([-2., -1., 0., 1., 2.])
        upstream = torch.tensor([1., 2., 3., 4., 5.])
        q = query(x, upstream, -1, 1)
        out = torch.clamp(x, -1, 1)
        grad = torch.tensor([0., 2., 3., 4., 0.])
        valid = {"out": record(out), "grad": record(grad)}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {"out": record(out)}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "grad": record(torch.zeros_like(grad))}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "out": {**valid["out"], "t": "f64"}}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "grad": record(torch.zeros(1))}))


if __name__ == "__main__":
    unittest.main()
