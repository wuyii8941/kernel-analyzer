"""Independent coordinate and PyTorch 2.10 checks for finite clip_grad_value_."""
from fractions import Fraction
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry


DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}
SHAPES = ((7,), (3, 5), (2, 3, 4), (2, 2, 3, 4))
BOUNDS = (0.75, 1.0, 1.5, 2.0)
FOREACH = (False, True, None, False)


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return float(value) if isinstance(value, Fraction) else value


def independent(values, bound):
    return np.asarray([min(max(value, -bound), bound) for value in values.flat], dtype=np.float64).reshape(values.shape)


class ClipValueIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("CLIP").adapter

    def test_generated_cases_match_coordinate_clamp_and_pytorch(self):
        rng = random.Random(20261019)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for shape, bound, foreach in zip(SHAPES, BOUNDS, FOREACH):
                values = np.asarray([rng.randrange(-16, 17) / 4 for _ in range(int(np.prod(shape)))],
                                    dtype=np.float64).reshape(shape)
                expected = independent(values, bound)
                grad = torch.tensor(values.tolist(), dtype=torch_dtype)
                parameter = torch.nn.Parameter(torch.zeros_like(grad))
                parameter.grad = grad.clone()
                torch.nn.utils.clip_grad_value_([parameter], bound, foreach=foreach)
                expected_tensor = torch.tensor(expected.tolist(), dtype=torch_dtype)
                self.assertTrue(torch.equal(parameter.grad, expected_tensor), (dtype_name, shape, foreach))

                inp = {"op": "torch.nn.utils.clip_grad_value_", "args": [
                    {"t": dtype_name, "v": grad.tolist(), "shape": list(shape)}, bound],
                    "kwargs": {"foreach": foreach}}
                result = self.adapter.reference("finite_value_single_gradient", inp).fields["out"]
                self.assertEqual(np.asarray(plain(result)).tolist(), expected.tolist())
                observed = {"out": {"t": dtype_name, "v": parameter.grad.tolist(), "shape": list(shape)}}
                self.assertTrue(self.adapter.check_output("finite_value_single_gradient", inp, observed))
                checked += 1
        self.assertEqual(checked, 16)


if __name__ == "__main__":
    unittest.main()
