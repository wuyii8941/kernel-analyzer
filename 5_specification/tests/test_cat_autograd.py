"""Independent checks for the PyTorch 2.10 public Tensor-list cat profile."""
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


PROFILE = "torch210_cat_autograd"
DTYPES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def record(tensor):
    return {"t": DTYPES[tensor.dtype], "shape": list(tensor.shape), "v": tensor.tolist()}


def query(tensors, dim, upstream, **changes):
    result = {
        "op": "torch.cat",
        "tensors": [record(tensor) for tensor in tensors],
        "dim": dim,
        "upstream": record(upstream),
    }
    result.update(changes)
    return result


def exact_nested(value):
    value = np.asarray(value, dtype=object)
    if value.ndim == 0:
        return Fraction(value.item())
    return [exact_nested(child) for child in value]


def independent_reference(tensors, dim, upstream):
    """Construct the coordinate copy and its transpose without LAYOUT helpers."""
    active = [tensor for tensor in tensors if tuple(tensor.shape) != (0,)]
    if not active:
        return {"out": [], "grads": [[] for _ in tensors]}
    rank = active[0].ndim
    axis = dim + rank if dim < 0 else dim
    if axis < 0 or axis >= rank:
        raise ValueError("invalid cat dimension")
    arrays = [np.asarray(tensor.tolist(), dtype=object) for tensor in active]
    out = np.concatenate(arrays, axis=axis)
    upstream_values = np.asarray(upstream.tolist(), dtype=object)
    grads, offset = [], 0
    for tensor in tensors:
        if tuple(tensor.shape) == (0,):
            grads.append([])
            continue
        index = [slice(None)] * rank
        index[axis] = slice(offset, offset + tensor.shape[axis])
        grads.append(exact_nested(upstream_values[tuple(index)]))
        offset += tensor.shape[axis]
    return {"out": exact_nested(out), "grads": grads}


class CatAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("LAYOUT").adapter

    def test_hand_witness_preserves_order_and_offsets(self):
        tensors = [torch.tensor([[1.], [2.]]),
                   torch.tensor([[3., 4.], [5., 6.]]),
                   torch.tensor([[7.], [8.]])]
        upstream = torch.tensor([[10., 11., 12., 13.], [14., 15., 16., 17.]])
        fields = self.adapter.reference(PROFILE, query(tensors, -1, upstream)).fields
        self.assertEqual(fields, {
            "out": [[1, 3, 4, 7], [2, 5, 6, 8]],
            "grads": [[[10], [14]], [[11, 12], [15, 16]], [[13], [17]]],
        })

    def test_independent_coordinates_and_pytorch_210_autograd(self):
        cases = [
            ([(2,), (3,)], 0),
            ([(2, 1), (2, 2), (2, 3)], -1),
            ([(1, 2, 2), (2, 2, 2)], 0),
            ([(2, 3, 1), (2, 3, 2)], 2),
            ([(0,), (2, 3), (2, 0)], 1),
            ([(0,), (0,)], 5),
        ]
        for dtype, (shapes, dim) in itertools.product(DTYPES, cases):
            with self.subTest(dtype=dtype, shapes=shapes, dim=dim):
                tensors, cursor = [], 0
                for shape in shapes:
                    count = int(np.prod(shape))
                    tensor = ((torch.arange(count).reshape(shape) + cursor - 4) / 4).to(dtype)
                    tensors.append(tensor.requires_grad_())
                    cursor += count
                out = torch.cat(tensors, dim)
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 11) - 5).to(dtype)
                grads = torch.autograd.grad(out, tensors, upstream)

                expected = independent_reference([item.detach() for item in tensors], dim, upstream)
                actual = self.adapter.reference(
                    PROFILE, query([item.detach() for item in tensors], dim, upstream)).fields
                self.assertEqual(actual, expected)
                self.assertEqual(out.detach().tolist(),
                                 torch.tensor(expected["out"], dtype=dtype).tolist())
                for grad, expected_grad in zip(grads, expected["grads"]):
                    expected_tensor = torch.tensor(expected_grad, dtype=dtype).reshape(grad.shape)
                    self.assertTrue(torch.equal(grad, expected_tensor))

    def test_profile_rejects_other_overloads_and_incomplete_calls(self):
        tensors = [torch.tensor([[1.], [2.]]), torch.tensor([[3., 4.], [5., 6.]])]
        upstream = torch.ones(2, 3)
        valid = query(tensors, 1, upstream)
        cases = [
            ({**valid, "op": "aten.cat"}, Unsupported),
            ({**valid, "tensors": []}, Invalid),
            ({**valid, "tensors": [{"t": "f32", "shape": [], "v": 1}],
              "upstream": {"t": "f32", "shape": [], "v": 1}}, Invalid),
            ({**valid, "tensors": [record(tensors[0]), record(tensors[1].double())]}, Unsupported),
            ({**valid, "upstream": record(torch.ones(2, 2))}, Invalid),
            ({**valid, "tensors": [{"t": "f32", "v": [[1], [2]]}, record(tensors[1])]}, Invalid),
            ({**valid, "dim": True}, Invalid),
            ({**valid, "extra": 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_every_ordered_gradient(self):
        tensors = [torch.tensor([[1.], [2.]]), torch.tensor([[3., 4.], [5., 6.]])]
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        q = query(tensors, 1, upstream)
        valid = {
            "out": record(torch.cat(tensors, 1)),
            "grads": [record(upstream[:, :1]), record(upstream[:, 1:])],
        }
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {"out": valid["out"]}))
        self.assertFalse(self.adapter.check_output(PROFILE, q,
                                                   {**valid, "grads": valid["grads"][:-1]}))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {
            **valid, "grads": [valid["grads"][1], valid["grads"][0]],
        }))
        wrong = record(upstream[:, 1:].clone())
        wrong["v"][0][0] = 0
        self.assertFalse(self.adapter.check_output(PROFILE, q, {
            **valid, "grads": [valid["grads"][0], wrong],
        }))


if __name__ == "__main__":
    unittest.main()
