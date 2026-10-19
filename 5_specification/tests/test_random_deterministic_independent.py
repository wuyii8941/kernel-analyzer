"""Independent coordinate and PyTorch 2.10 checks for deterministic dropout scopes."""
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
SHAPES = ((17,), (3, 5), (2, 3, 4), (2, 2, 3, 4))


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return float(value) if isinstance(value, Fraction) else value


class RandomDeterministicIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RANDOM").adapter

    def test_generated_backward_matches_coordinate_formula_and_pytorch(self):
        rng = random.Random(20261019)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for shape in SHAPES:
                size = int(np.prod(shape))
                values = np.asarray([rng.randrange(-8, 9) for _ in range(size)]).reshape(shape)
                mask = np.asarray([rng.randrange(2) == 1 for _ in range(size)]).reshape(shape)
                grad = torch.tensor(values.tolist(), dtype=torch_dtype)
                keep = torch.tensor(mask.tolist(), dtype=torch.bool)
                expected = np.where(mask, values * 1.25, 0)
                inp = {"op": "aten.native_dropout_backward", "args": [
                    {"t": dtype_name, "v": grad.tolist(), "shape": list(shape)},
                    {"t": "b8", "v": keep.tolist(), "shape": list(shape)}, "5/4"]}
                result = self.adapter.reference("finite_backward", inp).fields["out"]
                self.assertEqual(np.asarray(plain(result)).tolist(), expected.tolist())
                actual = torch.ops.aten.native_dropout_backward.default(grad, keep, 1.25)
                expected_tensor = torch.tensor(expected.tolist(), dtype=torch_dtype)
                self.assertTrue(torch.equal(actual, expected_tensor), (dtype_name, shape))
                observed = {"out": {"t": dtype_name, "v": actual.tolist(), "shape": list(shape)}}
                self.assertTrue(self.adapter.check_output("finite_backward", inp, observed))
                checked += 1
        self.assertEqual(checked, 16)

    def test_forward_boundaries_match_pytorch(self):
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            tensor = torch.tensor([[1, -2, 3], [4, 0, -5]], dtype=torch_dtype)
            record = {"t": dtype_name, "v": tensor.tolist(), "shape": [2, 3]}
            for p, training in ((0.375, False), (0, True), (1, True)):
                inp = {"op": "torch.nn.functional.dropout", "args": [record, p, training, False]}
                result = self.adapter.reference("deterministic_forward", inp).fields["out"]
                actual = F.dropout(tensor, p=p, training=training, inplace=False)
                self.assertEqual(np.asarray(plain(result)).tolist(), actual.tolist())
                observed = {"out": {"t": dtype_name, "v": actual.tolist(), "shape": [2, 3]}}
                self.assertTrue(self.adapter.check_output("deterministic_forward", inp, observed))
                checked += 1
        self.assertEqual(checked, 12)


if __name__ == "__main__":
    unittest.main()
