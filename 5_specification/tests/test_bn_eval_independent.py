"""Independent coordinate and PyTorch 2.10 checks for BatchNorm evaluation."""
from fractions import Fraction
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import ends


DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}
SHAPES = ((4, 3), (2, 3, 5), (2, 3, 4, 5))


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return float(value) if isinstance(value, Fraction) else value


def independent(data, mean, variance, weight, bias, eps):
    out = np.empty(data.shape, dtype=np.float64)
    for index in np.ndindex(data.shape):
        channel = index[1]
        out[index] = ((data[index] - mean[channel]) /
                      np.sqrt(variance[channel] + eps) * weight[channel] + bias[channel])
    return out


class BatchNormEvalIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("BN").adapter

    def test_generated_cases_match_coordinate_loop_and_pytorch(self):
        rng = random.Random(20261019)
        mean = np.asarray([0, 1, -2], dtype=np.float64)
        variance = np.asarray([0, 3, 15], dtype=np.float64)
        weight = np.asarray([1, 2, -1], dtype=np.float64)
        bias = np.asarray([0, -1, 2], dtype=np.float64)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for shape in SHAPES:
                values = [rng.randrange(-8, 9) for _ in range(int(np.prod(shape)))]
                data = np.asarray(values, dtype=np.float64).reshape(shape)
                tensors = [torch.tensor(item.tolist(), dtype=torch_dtype)
                           for item in (data, mean, variance, weight, bias)]
                records = [{"t": dtype_name, "v": tensor.tolist(), "shape": list(tensor.shape)}
                           for tensor in tensors]
                expected = independent(data, mean, variance, weight, bias, 1)
                inp = {"op": "torch.nn.functional.batch_norm",
                       "args": records + [False, 0, 1]}
                result = self.adapter.reference("finite_eval_values", inp).fields["out"]
                self.assertEqual(np.asarray(plain(result)).tolist(), expected.tolist())
                actual = F.batch_norm(*tensors, training=False, momentum=0, eps=1)
                expected_tensor = torch.tensor(expected.tolist(), dtype=torch_dtype)
                self.assertTrue(torch.equal(actual, expected_tensor), (dtype_name, shape, actual, expected_tensor))
                observed = {"out": {"t": dtype_name, "v": actual.tolist(), "shape": list(actual.shape)}}
                self.assertTrue(self.adapter.check_output("finite_eval_values", inp, observed))
                checked += 1
        self.assertEqual(checked, 12)

    def test_nonsquare_radicands_enclose_independent_high_precision_values(self):
        inp = {"op": "torch.nn.functional.batch_norm", "args": [
            {"t": "f32", "v": [[-3, 2, 7], [5, -1, 4]]},
            {"t": "f32", "v": [1, -2, 3]}, {"t": "f32", "v": [1, 2, 5]},
            {"t": "f32", "v": [2, -1, 3]}, {"t": "f32", "v": [0, 4, -2]},
            False, 0, "1/8"]}
        result = self.adapter.reference("finite_eval_values", inp).fields["out"]
        mpmath.mp.dps = 100
        x = [[-3, 2, 7], [5, -1, 4]]
        mean, variance, weight, bias = [1, -2, 3], [1, 2, 5], [2, -1, 3], [0, 4, -2]
        saw_nonpoint = False
        for n in range(2):
            for c in range(3):
                lo, hi = ends(result[n][c])
                exact = ((mpmath.mpf(x[n][c]) - mean[c]) /
                         mpmath.sqrt(mpmath.mpf(variance[c]) + mpmath.mpf(1) / 8) *
                         weight[c] + bias[c])
                lower = mpmath.mpf(lo.numerator) / lo.denominator
                upper = mpmath.mpf(hi.numerator) / hi.denominator
                self.assertLessEqual(lower, exact)
                self.assertLessEqual(exact, upper)
                saw_nonpoint |= lo != hi
        self.assertTrue(saw_nonpoint)


if __name__ == "__main__":
    unittest.main()
