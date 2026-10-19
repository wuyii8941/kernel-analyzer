"""Independent coordinates and PyTorch autograd checks for reviewed Llama RoPE application."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


PROFILE = "hf4573_llama_tables_real"
DTYPES = (
    (torch.float16, "f16", 5e-3),
    (torch.bfloat16, "bf16", 4e-2),
    (torch.float32, "f32", 3e-6),
    (torch.float64, "f64", 1e-12),
)


def _task(q, k, cos, sin, upstream_q, upstream_k, dtype="f64"):
    return {
        "q": q, "k": k, "cos": cos, "sin": sin,
        "upstream_q": upstream_q, "upstream_k": upstream_k,
        "dtype": dtype, "position_ids": None, "unsqueeze_dim": 1,
    }


CASES = (
    _task([[[[1, 2, 3, 4]]]], [[[[5, 6, 7, 8]]]], [[[1, 2, 3, 4]]], [[[5, 6, 7, 8]]],
          [[[[1, 2, 3, 4]]]], [[[[4, 3, 2, 1]]]]),
    _task([[[[1, 2, 3, 4]]]], [[[[5, 6, 7, 8]]]], [[[0, 0, 0, 0]]], [[[1, 1, 1, 1]]],
          [[[[1, 2, 3, 4]]]], [[[[4, 3, 2, 1]]]]),
    _task([[[[1, 2]], [[3, 4]]]], [[[[5, 6]]]], [[[1, 1]]], [[[0, 0]]],
          [[[[7, 8]], [[9, 10]]]], [[[[11, 12]]]]),
    _task([[[[1, 2]]], [[[3, 4]]]], [[[[5, 6]]], [[[7, 8]]]],
          [[[1, 1]], [[0, 0]]], [[[0, 0]], [[1, 1]]],
          [[[[5, 6]]], [[[7, 8]]]], [[[[9, 10]]], [[[11, 12]]]]),
    _task([[[[1, 0], [2, 0]]]], [[[[0, 1], [0, 2]]]],
          [[[1, 1], [0, 0]]], [[[0, 0], [1, 1]]],
          [[[[1, 1], [2, 3]]]], [[[[4, 5], [6, 7]]]]),
)


def _coordinates(inp):
    q = np.asarray(inp["q"], dtype=object)
    k = np.asarray(inp["k"], dtype=object)
    cos = np.asarray(inp["cos"], dtype=object)
    sin = np.asarray(inp["sin"], dtype=object)

    def apply(x, upstream):
        upstream = np.asarray(upstream, dtype=object)
        out = np.zeros(x.shape, dtype=object)
        grad = np.zeros(x.shape, dtype=object)
        half = x.shape[-1] // 2
        for b, h, t in np.ndindex(x.shape[:-1]):
            for p in range(half):
                s = p + half
                out[b, h, t, p] = x[b, h, t, p] * cos[b, t, p] - x[b, h, t, s] * sin[b, t, p]
                out[b, h, t, s] = x[b, h, t, s] * cos[b, t, s] + x[b, h, t, p] * sin[b, t, s]
                grad[b, h, t, p] = (upstream[b, h, t, p] * cos[b, t, p]
                                     + upstream[b, h, t, s] * sin[b, t, s])
                grad[b, h, t, s] = (-upstream[b, h, t, p] * sin[b, t, p]
                                     + upstream[b, h, t, s] * cos[b, t, s])
        return out.tolist(), grad.tolist()

    q_out, dq = apply(q, inp["upstream_q"])
    k_out, dk = apply(k, inp["upstream_k"])
    return {"q_out": q_out, "k_out": k_out, "dq": dq, "dk": dk}


def _torch_apply(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class IndependentRope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("ROPE").adapter

    def test_reference_equals_independent_coordinate_formula(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                self.assertEqual(self.adapter.reference(PROFILE, inp).fields, _coordinates(inp))
                self.assertEqual(inp, before)

    def test_twenty_pytorch_formula_and_autograd_calls_follow_the_same_task(self):
        checked = 0
        for torch_dtype, dtype_name, tolerance in DTYPES:
            for index, template in enumerate(CASES):
                with self.subTest(dtype=str(torch_dtype), case=index):
                    q = torch.tensor(template["q"], dtype=torch_dtype, requires_grad=True)
                    k = torch.tensor(template["k"], dtype=torch_dtype, requires_grad=True)
                    cos = torch.tensor(template["cos"], dtype=torch_dtype)
                    sin = torch.tensor(template["sin"], dtype=torch_dtype)
                    uq = torch.tensor(template["upstream_q"], dtype=torch_dtype)
                    uk = torch.tensor(template["upstream_k"], dtype=torch_dtype)
                    q_before, k_before, cos_before, sin_before = (value.detach().clone() for value in (q, k, cos, sin))
                    q_out, k_out = _torch_apply(q, k, cos, sin)
                    torch.autograd.backward((q_out, k_out), (uq, uk))
                    inp = _task(q_before.tolist(), k_before.tolist(), cos.tolist(), sin.tolist(), uq.tolist(), uk.tolist(),
                                dtype=dtype_name)
                    expected = self.adapter.reference(PROFILE, inp).fields
                    for actual, name in ((q_out, "q_out"), (k_out, "k_out"), (q.grad, "dq"), (k.grad, "dk")):
                        target = torch.from_numpy(np.asarray(expected[name], dtype=np.float64))
                        torch.testing.assert_close(actual.detach().double(), target,
                                                   rtol=tolerance, atol=tolerance)
                    self.assertTrue(torch.equal(q.detach(), q_before))
                    self.assertTrue(torch.equal(k.detach(), k_before))
                    self.assertTrue(torch.equal(cos, cos_before))
                    self.assertTrue(torch.equal(sin, sin_before))
                    checked += 1
        self.assertEqual(checked, 20)

    def test_complete_output_checker_enforces_fields_dtype_and_shape(self):
        inp = CASES[1]
        expected = self.adapter.reference(PROFILE, inp).fields
        records = {name: {"t": "f64", "shape": list(np.asarray(value).shape), "v": value}
                   for name, value in expected.items()}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, records))
        missing = dict(records); missing.pop("dk")
        self.assertFalse(self.adapter.check_output(PROFILE, inp, missing))
        wrong_dtype = dict(records); wrong_dtype["dq"] = dict(records["dq"], t="f32")
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_dtype))
        wrong_shape = dict(records); wrong_shape["k_out"] = dict(records["k_out"], shape=[4])
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_shape))

    def test_other_rope_tasks_are_refused(self):
        base = CASES[0]
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, unsqueeze_dim=2)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, position_ids=[[0]])).outcome,
                         "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, rotary_dim=4)).outcome, "unsupported")


if __name__ == "__main__":
    unittest.main()
