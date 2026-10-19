"""Independent coordinate and PyTorch 2.10 checks for finite out-of-place index_add."""
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
CASES = (
    ((7,), 0, [2, 0, 2, 6], "1/2"),
    ((4, 5), 1, [3, 0, 3], 2),
    ((2, 4, 3), 1, [3, 0, 3], "1/2"),
    ((2, 3, 4, 2), 2, [3, 0, 3], 2),
)


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return float(value) if isinstance(value, Fraction) else value


def independent(self_values, axis, indices, source, alpha):
    out = self_values.copy()
    for source_index in np.ndindex(source.shape):
        target_index = list(source_index)
        target_index[axis] = indices[source_index[axis]]
        out[tuple(target_index)] += alpha * source[source_index]
    return out


class IndexAddIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("INDEXREDUCE").adapter

    def test_generated_cases_match_coordinate_loop_and_pytorch(self):
        rng = random.Random(20261019)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for shape, axis, indices, alpha_record in CASES:
                source_shape = list(shape)
                source_shape[axis] = len(indices)
                self_values = np.asarray([rng.randrange(-4, 5) for _ in range(int(np.prod(shape)))],
                                         dtype=np.float64).reshape(shape)
                source = np.asarray([rng.randrange(-4, 5) for _ in range(int(np.prod(source_shape)))],
                                    dtype=np.float64).reshape(source_shape)
                alpha = 0.5 if alpha_record == "1/2" else float(alpha_record)
                expected = independent(self_values, axis, indices, source, alpha)
                self_tensor = torch.tensor(self_values.tolist(), dtype=torch_dtype)
                source_tensor = torch.tensor(source.tolist(), dtype=torch_dtype)
                index_tensor = torch.tensor(indices, dtype=torch.int64)
                inp = {"op": "torch.index_add", "args": [
                    {"t": dtype_name, "v": self_tensor.tolist(), "shape": list(shape)}, axis,
                    {"t": "i64", "v": indices, "shape": [len(indices)]},
                    {"t": dtype_name, "v": source_tensor.tolist(), "shape": source_shape}],
                    "kwargs": {"alpha": alpha_record}}
                result = self.adapter.reference("finite_index_add_values", inp).fields["out"]
                self.assertEqual(np.asarray(plain(result)).tolist(), expected.tolist())
                actual = torch.index_add(self_tensor, axis, index_tensor, source_tensor, alpha=alpha)
                expected_tensor = torch.tensor(expected.tolist(), dtype=torch_dtype)
                self.assertTrue(torch.equal(actual, expected_tensor), (dtype_name, shape, axis))
                observed = {"out": {"t": dtype_name, "v": actual.tolist(), "shape": list(shape)}}
                self.assertTrue(self.adapter.check_output("finite_index_add_values", inp, observed))
                checked += 1
        self.assertEqual(checked, 16)


if __name__ == "__main__":
    unittest.main()
