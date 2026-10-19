"""Independent checks for the PyTorch 2.10 public constant-pad profile."""
from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


PROFILE = "torch210_functional_constant_pad_autograd"
DTYPES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def record(tensor):
    return {"t": DTYPES[tensor.dtype], "shape": list(tensor.shape), "v": tensor.tolist()}


def query(x, pad, value, upstream, **changes):
    result = {
        "op": "torch.nn.functional.pad",
        "input": record(x),
        "pad": list(pad),
        "mode": "constant",
        "value": value,
        "upstream": record(upstream),
    }
    result.update(changes)
    return result


def exact_nested(array):
    array = np.asarray(array, dtype=object)
    if array.ndim == 0:
        return Fraction(array.item())
    return [exact_nested(child) for child in array]


def independent_reference(x, pad, value, upstream):
    """Coordinate construction independent of kernel_spec.aten.constant_pad_nd."""
    shape = tuple(x.shape)
    if len(pad) % 2 or len(pad) > 2 * len(shape):
        raise ValueError("padding must contain pairs for existing dimensions")
    left = [0] * len(shape)
    right = [0] * len(shape)
    for pair in range(len(pad) // 2):
        axis = len(shape) - 1 - pair
        left[axis], right[axis] = pad[2 * pair:2 * pair + 2]
        if shape[axis] + min(left[axis], 0) + min(right[axis], 0) < 0:
            raise ValueError("crop exceeds input")

    output_shape = tuple(length + lo + hi for length, lo, hi in zip(shape, left, right))
    out = np.empty(output_shape, dtype=object)
    out.fill(Fraction(0 if value is None else value))
    grad = np.empty(shape, dtype=object)
    grad.fill(Fraction(0))
    x_values = np.asarray(x.tolist(), dtype=object)
    upstream_values = np.asarray(upstream.tolist(), dtype=object)
    for input_index in np.ndindex(shape):
        retained = all(
            input_index[axis] >= max(-left[axis], 0)
            and input_index[axis] < shape[axis] - max(-right[axis], 0)
            for axis in range(len(shape))
        )
        if retained:
            output_index = tuple(input_index[axis] + left[axis] for axis in range(len(shape)))
            out[output_index] = Fraction(x_values[input_index])
            grad[input_index] = Fraction(upstream_values[output_index])
    return {"out": exact_nested(out), "grad": exact_nested(grad)}


class ConstantPadAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("LAYOUT").adapter

    def test_mixed_pad_hand_witness(self):
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        pad = (1, -1, -1, 1)
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        fields = self.adapter.reference(PROFILE, query(x, pad, "1/2", upstream)).fields
        self.assertEqual(fields, {
            "out": [[Fraction(1, 2), 4, 5], [Fraction(1, 2)] * 3],
            "grad": [[0, 0, 0], [2, 3, 0]],
        })

    def test_independent_coordinates_and_pytorch_210_autograd(self):
        cases = [
            ((3,), (2, 1), None),
            ((4,), (-1, 2), Fraction(-1, 2)),
            ((2, 3), (1, -1, -1, 2), Fraction(1, 4)),
            ((2, 3, 2), (0, 1, -1, 0), Fraction(-1, 4)),
            ((4,), (-2, -2), Fraction(1, 2)),
        ]
        for dtype, (shape, pad, value) in itertools.product(DTYPES, cases):
            with self.subTest(dtype=dtype, shape=shape, pad=pad, value=value):
                count = int(np.prod(shape))
                x = ((torch.arange(count).reshape(shape) - 3) / 4).to(dtype).requires_grad_()
                torch_value = None if value is None else float(value)
                out = F.pad(x, pad, mode="constant", value=torch_value)
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 7) - 3).to(dtype)
                dx, = torch.autograd.grad(out, x, upstream)

                expected = independent_reference(x.detach(), pad, value, upstream)
                encoded_value = None if value is None else str(value)
                actual = self.adapter.reference(
                    PROFILE, query(x.detach(), pad, encoded_value, upstream)).fields
                self.assertEqual(actual, expected)
                to_float = np.vectorize(float, otypes=[float])
                self.assertEqual(out.detach().tolist(),
                                 to_float(np.asarray(expected["out"], dtype=object)).tolist())
                self.assertEqual(dx.tolist(),
                                 to_float(np.asarray(expected["grad"], dtype=object)).tolist())

    def test_profile_rejects_unselected_or_incomplete_calls(self):
        x = torch.tensor([1., 2., 3.])
        upstream = torch.ones(5)
        valid = query(x, (1, 1), 0, upstream)
        cases = [
            ({**valid, "op": "aten.constant_pad_nd"}, Unsupported),
            ({**valid, "mode": "reflect"}, Unsupported),
            ({**valid, "input": {"t": "f32", "v": [1, 2, 3]}}, Invalid),
            ({**valid, "upstream": record(torch.ones(4))}, Invalid),
            ({**valid, "upstream": record(torch.ones(5, dtype=torch.float64))}, Invalid),
            ({**valid, "pad": [1, True]}, Invalid),
            ({**valid, "pad": [-4, 5], "upstream": record(torch.ones(4))}, Invalid),
            ({**valid, "extra": 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_forward_and_gradient(self):
        x = torch.tensor([1., 2., 3.])
        pad = (2, 1)
        upstream = torch.arange(6, dtype=torch.float32)
        q = query(x, pad, None, upstream)
        valid = {
            "out": record(F.pad(x, pad)),
            "grad": record(torch.tensor([2., 3., 4.])),
        }
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {"out": valid["out"]}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "grad": record(torch.zeros_like(x))}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, "out": {**valid["out"], "t": "f64"}}))


if __name__ == "__main__":
    unittest.main()
