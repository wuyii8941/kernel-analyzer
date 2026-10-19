"""Independent formula and public PyTorch checks for the reviewed dense embedding profile."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


PROFILE = "torch210_dense_real"
DTYPES = (
    (torch.float16, "f16", 4e-3),
    (torch.bfloat16, "bf16", 3e-2),
    (torch.float32, "f32", 2e-6),
    (torch.float64, "f64", 1e-12),
)


def _task(weight, indices, upstream, dtype="f64", padding_idx=None, frequency=False):
    return {
        "weight": weight, "indices": indices, "upstream": upstream,
        "dtype": dtype, "index_dtype": "i64", "padding_idx": padding_idx,
        "max_norm": None, "norm_type": 2, "scale_grad_by_freq": frequency, "sparse": False,
    }


def _independent(inp):
    weight = np.asarray(inp["weight"], dtype=object)
    indices = np.asarray(inp["indices"], dtype=np.int64)
    upstream = np.asarray(inp["upstream"], dtype=object)
    padding = inp["padding_idx"]
    if padding is not None and padding < 0:
        padding += weight.shape[0]
    out = weight[indices]
    dweight = np.zeros(weight.shape, dtype=object)
    counts = {int(row): int(np.count_nonzero(indices == row)) for row in np.unique(indices)}
    for position in np.ndindex(indices.shape):
        row = int(indices[position])
        if row == padding:
            continue
        scale = counts[row] if inp["scale_grad_by_freq"] else 1
        dweight[row] = dweight[row] + upstream[position] / scale
    return out.tolist(), dweight.tolist()


CASES = (
    _task([[1, 2], [3, 4], [5, 6]], [2, 0, 2], [[1, 1], [2, 3], [10, 20]]),
    _task([[7, 8], [1, 2]], [0, 1, 0], [[1, 1], [2, 2], [3, 3]], padding_idx=0),
    _task([[7, 8], [1, 2]], [1, 0], [[1, 2], [3, 4]], padding_idx=-1),
    _task([[1, 1], [2, 2]], [1, 1, 0], [[2, 4], [6, 8], [1, 1]], frequency=True),
    _task([[1, 2], [3, 4], [5, 6]], [[2, 0], [1, 2]],
          [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], frequency=True),
)


class IndependentEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("EMB").adapter

    def test_reference_equals_independent_coordinate_formula(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                expected_out, expected_gradient = _independent(inp)
                actual = self.adapter.reference(PROFILE, inp).fields
                self.assertEqual(actual["out"], expected_out)
                self.assertEqual(actual["dweight"], expected_gradient)
                self.assertEqual(inp, before)

    def test_twenty_public_calls_follow_the_same_task(self):
        checked = 0
        for torch_dtype, dtype_name, tolerance in DTYPES:
            for index, template in enumerate(CASES):
                with self.subTest(dtype=str(torch_dtype), case=index):
                    weight = torch.tensor(template["weight"], dtype=torch_dtype, requires_grad=True)
                    indices = torch.tensor(template["indices"], dtype=torch.int64)
                    upstream = torch.tensor(template["upstream"], dtype=torch_dtype)
                    weight_before = weight.detach().clone()
                    indices_before = indices.clone()
                    upstream_before = upstream.clone()
                    out = torch_f.embedding(
                        indices, weight, padding_idx=template["padding_idx"], max_norm=None, norm_type=2,
                        scale_grad_by_freq=template["scale_grad_by_freq"], sparse=False,
                    )
                    out.backward(upstream)
                    inp = _task(
                        weight_before.tolist(), indices.tolist(), upstream.tolist(), dtype=dtype_name,
                        padding_idx=template["padding_idx"], frequency=template["scale_grad_by_freq"],
                    )
                    reference = self.adapter.reference(PROFILE, inp).fields
                    expected_out = torch.tensor(reference["out"], dtype=torch.float64)
                    expected_gradient = torch.tensor(reference["dweight"], dtype=torch.float64)
                    torch.testing.assert_close(out.detach().double(), expected_out, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(weight.grad.double(), expected_gradient, rtol=tolerance, atol=tolerance)
                    self.assertTrue(torch.equal(weight.detach(), weight_before))
                    self.assertTrue(torch.equal(indices, indices_before))
                    self.assertTrue(torch.equal(upstream, upstream_before))
                    checked += 1
        self.assertEqual(checked, 20)

    def test_complete_output_checker_enforces_fields_dtype_and_shape(self):
        inp = _task([[1, 2], [3, 4]], [1, 0], [[2, 3], [4, 5]], dtype="f32")
        exact = {
            "out": {"t": "f32", "shape": [2, 2], "v": [[3, 4], [1, 2]]},
            "dweight": {"t": "f32", "shape": [2, 2], "v": [[4, 5], [2, 3]]},
        }
        self.assertTrue(self.adapter.check_output(PROFILE, inp, exact))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": exact["out"]}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**exact, "extra": []}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**exact, "out": dict(exact["out"], t="f64")}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, inp, {**exact, "dweight": dict(exact["dweight"], shape=[4])}))

    def test_unselected_public_tasks_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, max_norm=1)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, sparse=True)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, op="embedding_bag")).outcome,
                         "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, index_dtype="i32")).outcome,
                         "unsupported")


if __name__ == "__main__":
    unittest.main()
