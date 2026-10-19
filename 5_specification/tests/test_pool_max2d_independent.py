"""Independent coordinate and PyTorch 2.10 checks for reviewed max_pool2d values."""
from fractions import Fraction
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry


DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}
CONFIGS = [((2, 2), (2, 2), (1, 1)), ((2, 3), (1, 2), (1, 1)), ((2, 2), (1, 1), (2, 2))]


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return float(value) if isinstance(value, Fraction) else value


def independent(data, kernel, stride, dilation):
    h, w = data.shape[-2:]
    oh = (h - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    ow = (w - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    out = np.empty(data.shape[:-2] + (oh, ow), dtype=np.float64)
    for lead in np.ndindex(data.shape[:-2]):
        for i in range(oh):
            for j in range(ow):
                values = [data[lead + (i * stride[0] + m * dilation[0],
                                       j * stride[1] + n * dilation[1])]
                          for m in range(kernel[0]) for n in range(kernel[1])]
                out[lead + (i, j)] = max(values)
    return out


class PoolMax2dIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("POOL").adapter

    def test_generated_cases_match_coordinate_loop_and_pytorch(self):
        rng = random.Random(20261019)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for shape in ((3, 7, 8), (2, 3, 7, 8)):
                values = [rng.randrange(-20, 21) for _ in range(int(np.prod(shape)))]
                data = np.asarray(values, dtype=np.float64).reshape(shape)
                tensor = torch.tensor(data.tolist(), dtype=torch_dtype)
                record = {"t": dtype_name, "v": tensor.tolist(), "shape": list(shape)}
                for kernel, stride, dilation in CONFIGS:
                    expected = independent(data, kernel, stride, dilation)
                    inp = {"op": "torch.nn.functional.max_pool2d", "args": [record, list(kernel)],
                           "kwargs": {"stride": list(stride), "dilation": list(dilation),
                                      "padding": 0, "ceil_mode": False, "return_indices": False}}
                    result = self.adapter.reference("finite_max2d_values", inp).fields["out"]
                    self.assertEqual(np.asarray(plain(result)).tolist(), expected.tolist())
                    actual = F.max_pool2d(tensor, kernel, stride, 0, dilation, False, False)
                    self.assertTrue(torch.equal(actual, torch.tensor(expected.tolist(), dtype=torch_dtype)))
                    observed = {"out": {"t": dtype_name, "v": actual.tolist(), "shape": list(actual.shape)}}
                    self.assertTrue(self.adapter.check_output("finite_max2d_values", inp, observed))
                    checked += 1
        self.assertEqual(checked, 24)


if __name__ == "__main__":
    unittest.main()
