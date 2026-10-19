"""Independent checks for PyTorch 2.10 Tensor/Tensor maximum and minimum."""
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


PROFILE = "torch210_extrema_tensor_autograd"
DTYPES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def record(tensor):
    return {"t": DTYPES[tensor.dtype], "shape": list(tensor.shape), "v": tensor.tolist()}


def query(op, x, other, upstream, **changes):
    result = {"op": "torch." + op, "input": record(x), "other": record(other),
              "upstream": record(upstream)}
    result.update(changes)
    return result


def exact_nested(array):
    array = np.asarray(array, dtype=object)
    if array.ndim == 0:
        return Fraction(array.item())
    return [exact_nested(child) for child in array]


def broadcast_index(output_index, input_shape, output_shape):
    leading = len(output_shape) - len(input_shape)
    return tuple(0 if length == 1 else output_index[leading + axis]
                 for axis, length in enumerate(input_shape))


def independent_reference(op, x, other, upstream):
    output_shape = torch.broadcast_shapes(x.shape, other.shape)
    xv = np.broadcast_to(np.asarray(x.tolist(), dtype=object), output_shape)
    yv = np.broadcast_to(np.asarray(other.tolist(), dtype=object), output_shape)
    uv = np.asarray(upstream.tolist(), dtype=object)
    out = np.empty(output_shape, dtype=object)
    grad_input = np.empty(x.shape, dtype=object)
    grad_other = np.empty(other.shape, dtype=object)
    grad_input.fill(Fraction(0))
    grad_other.fill(Fraction(0))
    for output_index in np.ndindex(output_shape):
        a, b, g = Fraction(xv[output_index]), Fraction(yv[output_index]), Fraction(uv[output_index])
        input_index = broadcast_index(output_index, x.shape, output_shape)
        other_index = broadcast_index(output_index, other.shape, output_shape)
        if a == b:
            out[output_index] = a
            grad_input[input_index] += g / 2
            grad_other[other_index] += g / 2
        else:
            input_selected = a > b if op == "maximum" else a < b
            out[output_index] = a if input_selected else b
            if input_selected:
                grad_input[input_index] += g
            else:
                grad_other[other_index] += g
    return {"out": exact_nested(out), "grad_input": exact_nested(grad_input),
            "grad_other": exact_nested(grad_other)}


class ExtremaAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("ARITH").adapter

    def test_broadcast_and_equal_split_hand_witness(self):
        x = torch.tensor([[1.], [2.]])
        other = torch.tensor([[1., 0., 2.]])
        upstream = torch.tensor([[2., 4., 6.], [8., 10., 12.]])
        fields = self.adapter.reference(PROFILE, query("maximum", x, other, upstream)).fields
        self.assertEqual(fields, {"out": [[1, 1, 2], [2, 2, 2]],
                                  "grad_input": [[5], [24]],
                                  "grad_other": [[1, 0, 12]]})

    def test_independent_coordinates_and_pytorch_210_autograd(self):
        shapes_and_values = [
            (((2, 3), [-1, 0, 1, 2, 0, -1]),
             ((2, 3), [-1, 1, 0, 2, -1, -1])),
            (((1, 3), [-1, 0, 1]), ((2, 1), [-1, 1])),
            (((2, 1, 4), [-1, 0, 1, 2, 0, 1, 2, 3]),
             ((1, 3, 1), [-1, 1, 3])),
        ]
        for dtype, op, ((x_shape, x_values), (other_shape, other_values)) in itertools.product(
                DTYPES, ("maximum", "minimum"), shapes_and_values):
            with self.subTest(dtype=dtype, op=op, x_shape=x_shape, other_shape=other_shape):
                x = torch.tensor(x_values, dtype=dtype).reshape(x_shape).requires_grad_()
                other = torch.tensor(other_values, dtype=dtype).reshape(other_shape).requires_grad_()
                output_shape = torch.broadcast_shapes(x.shape, other.shape)
                upstream = ((torch.arange(np.prod(output_shape)).reshape(output_shape) % 7) + 1).to(dtype)
                out = getattr(torch, op)(x, other)
                dx, dother = torch.autograd.grad(out, (x, other), upstream)

                expected = independent_reference(op, x.detach(), other.detach(), upstream)
                actual = self.adapter.reference(
                    PROFILE, query(op, x.detach(), other.detach(), upstream)).fields
                self.assertEqual(actual, expected)
                to_float = np.vectorize(float, otypes=[float])
                self.assertEqual(out.detach().tolist(),
                                 to_float(np.asarray(expected["out"], dtype=object)).tolist())
                self.assertEqual(dx.tolist(),
                                 to_float(np.asarray(expected["grad_input"], dtype=object)).tolist())
                self.assertEqual(dother.tolist(),
                                 to_float(np.asarray(expected["grad_other"], dtype=object)).tolist())

    def test_profile_rejects_other_overloads_and_incomplete_inputs(self):
        x = torch.tensor([1., 2.])
        other = torch.tensor([2., 1.])
        upstream = torch.ones(2)
        valid = query("maximum", x, other, upstream)
        cases = [
            ({**valid, "op": "aten.maximum"}, Unsupported),
            ({**valid, "other": 1}, Unsupported),
            ({**valid, "input": {"t": "f32", "v": [1., 2.]}}, Invalid),
            ({**valid, "upstream": record(torch.ones(1))}, Invalid),
            ({**valid, "upstream": record(torch.ones(2, dtype=torch.float64))}, Invalid),
            ({**valid, "other": record(other.double())}, Unsupported),
            ({**valid, "extra": 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_both_gradients(self):
        x = torch.tensor([[1.], [2.]])
        other = torch.tensor([[1., 0., 2.]])
        upstream = torch.tensor([[2., 4., 6.], [8., 10., 12.]])
        q = query("maximum", x, other, upstream)
        valid = {"out": record(torch.tensor([[1., 1., 2.], [2., 2., 2.]])),
                 "grad_input": record(torch.tensor([[5.], [24.]])),
                 "grad_other": record(torch.tensor([[1., 0., 12.]]))}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {name: value for name, value in valid.items() if name != "grad_other"}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "grad_input": record(torch.zeros_like(x))}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "out": {**valid["out"], "t": "f64"}}))


if __name__ == "__main__":
    unittest.main()
