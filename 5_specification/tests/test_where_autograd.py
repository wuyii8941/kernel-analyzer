"""Independent checks for the PyTorch 2.10 public Tensor/Tensor where profile."""
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


PROFILE = "torch210_where_tensor_autograd"
DTYPES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def record(tensor):
    dtype = "b8" if tensor.dtype == torch.bool else DTYPES[tensor.dtype]
    return {"t": dtype, "shape": list(tensor.shape), "v": tensor.tolist()}


def query(condition, x, other, upstream, **changes):
    result = {"op": "torch.where", "condition": record(condition), "input": record(x),
              "other": record(other), "upstream": record(upstream)}
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


def independent_reference(condition, x, other, upstream):
    output_shape = torch.broadcast_shapes(condition.shape, x.shape, other.shape)
    c = np.broadcast_to(np.asarray(condition.tolist(), dtype=bool), output_shape)
    xv = np.broadcast_to(np.asarray(x.tolist(), dtype=object), output_shape)
    yv = np.broadcast_to(np.asarray(other.tolist(), dtype=object), output_shape)
    uv = np.asarray(upstream.tolist(), dtype=object)
    out = np.where(c, xv, yv)
    grad_input = np.empty(x.shape, dtype=object)
    grad_other = np.empty(other.shape, dtype=object)
    grad_input.fill(Fraction(0))
    grad_other.fill(Fraction(0))
    for output_index in np.ndindex(output_shape):
        if c[output_index]:
            grad_input[broadcast_index(output_index, x.shape, output_shape)] += Fraction(uv[output_index])
        else:
            grad_other[broadcast_index(output_index, other.shape, output_shape)] += Fraction(uv[output_index])
    return {"out": exact_nested(out), "grad_input": exact_nested(grad_input),
            "grad_other": exact_nested(grad_other)}


class WhereAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("ARITH").adapter

    def test_two_broadcast_data_gradients_hand_witness(self):
        condition = torch.tensor([[True, False, True], [False, True, False]])
        x = torch.tensor([[1., 2., 3.]])
        other = torch.tensor([[10.], [20.]])
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        fields = self.adapter.reference(PROFILE, query(condition, x, other, upstream)).fields
        self.assertEqual(fields, {"out": [[1, 10, 3], [20, 2, 20]],
                                  "grad_input": [[1, 5, 3]],
                                  "grad_other": [[2], [10]]})

    def test_independent_coordinates_and_pytorch_210_autograd(self):
        shapes = [
            ((2, 3), (1, 3), (2, 1)),
            ((1, 3), (2, 3), (2, 1)),
            ((2, 3), (3,), (2, 1)),
            ((2, 3, 4), (2, 1, 4), (1, 3, 1)),
        ]
        for dtype, (condition_shape, input_shape, other_shape) in itertools.product(DTYPES, shapes):
            with self.subTest(dtype=dtype, input_shape=input_shape, other_shape=other_shape):
                x_count = int(np.prod(input_shape))
                y_count = int(np.prod(other_shape))
                x = ((torch.arange(x_count).reshape(input_shape) - 3) / 4).to(dtype).requires_grad_()
                other = ((torch.arange(y_count).reshape(other_shape) + 2) / 4).to(dtype).requires_grad_()
                condition_count = int(np.prod(condition_shape))
                condition = torch.arange(condition_count).reshape(condition_shape) % 3 != 1
                out = torch.where(condition, x, other)
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 7) - 3).to(dtype)
                dx, dy = torch.autograd.grad(out, (x, other), upstream)

                expected = independent_reference(condition, x.detach(), other.detach(), upstream)
                actual = self.adapter.reference(
                    PROFILE, query(condition, x.detach(), other.detach(), upstream)).fields
                self.assertEqual(actual, expected)
                to_float = np.vectorize(float, otypes=[float])
                self.assertEqual(out.detach().tolist(),
                                 to_float(np.asarray(expected["out"], dtype=object)).tolist())
                self.assertEqual(dx.tolist(),
                                 to_float(np.asarray(expected["grad_input"], dtype=object)).tolist())
                self.assertEqual(dy.tolist(),
                                 to_float(np.asarray(expected["grad_other"], dtype=object)).tolist())

    def test_profile_rejects_other_overloads_and_incomplete_inputs(self):
        condition = torch.tensor([True, False])
        x = torch.tensor([1., 2.])
        other = torch.tensor([3., 4.])
        upstream = torch.ones(2)
        valid = query(condition, x, other, upstream)
        cases = [
            ({**valid, "op": "torch.where.out"}, Unsupported),
            ({**valid, "input": 1}, Unsupported),
            ({**valid, "condition": {"t": "b8", "v": [True, False]}}, Invalid),
            ({**valid, "upstream": record(torch.ones(1))}, Invalid),
            ({**valid, "upstream": record(torch.ones(2, dtype=torch.float64))}, Invalid),
            ({**valid, "other": record(other.double())}, Unsupported),
            ({**valid, "extra": 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_both_gradients(self):
        condition = torch.tensor([[True, False, True], [False, True, False]])
        x = torch.tensor([[1., 2., 3.]])
        other = torch.tensor([[10.], [20.]])
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        q = query(condition, x, other, upstream)
        valid = {"out": record(torch.where(condition, x, other)),
                 "grad_input": record(torch.tensor([[1., 5., 3.]])),
                 "grad_other": record(torch.tensor([[2.], [10.]]))}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {name: value for name, value in valid.items() if name != "grad_other"}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "grad_input": record(torch.zeros_like(x))}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "out": {**valid["out"], "t": "f64"}}))


if __name__ == "__main__":
    unittest.main()
