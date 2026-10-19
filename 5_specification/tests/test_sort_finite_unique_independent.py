"""Independent NumPy construction and PyTorch 2.10 checks for reviewed SORT."""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry


DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


class SortFiniteUniqueIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SORT").adapter

    def test_generated_unique_rows_match_numpy_and_torch(self):
        rng = random.Random(20261019)
        checked = 0
        for dtype_name, torch_dtype in DTYPES.items():
            for dim in (0, 1):
                for descending in (False, True):
                    for _ in range(5):
                        rows = [rng.sample(range(-30, 31), 7) for _ in range(3)]
                        data = np.asarray(rows, dtype=np.float64)
                        # Sampling each original row is enough for dim=1. For dim=0,
                        # make every column unique as well by deterministic offsets.
                        if dim == 0:
                            data = np.asarray([[100 * r + c for c in rng.sample(range(-20, 21), 7)]
                                               for r in range(3)], dtype=np.float64)
                        tensor = torch.tensor(data.tolist(), dtype=torch_dtype)
                        record = {"t": dtype_name, "v": tensor.tolist(), "shape": list(tensor.shape)}
                        order = np.argsort(data, axis=dim, kind="stable")
                        if descending:
                            order = np.flip(order, axis=dim)
                        values = np.take_along_axis(data, order, axis=dim)

                        sort_inp = {"op": "aten.sort", "args": [record, dim, descending]}
                        sort_out = self.adapter.reference("finite_unique", sort_inp).fields["out"]
                        self.assertEqual(np.asarray(plain(sort_out[1])).tolist(), order.tolist())
                        self.assertEqual(np.asarray(plain(sort_out[0])).tolist(), values.tolist())
                        tv, ti = torch.sort(tensor, dim=dim, descending=descending)
                        self.assertTrue(torch.equal(tv, torch.tensor(values.tolist(), dtype=torch_dtype)))
                        self.assertEqual(ti.tolist(), order.tolist())

                        argsort_inp = {"op": "aten.argsort", "args": [record, dim, descending]}
                        argsort_out = self.adapter.reference("finite_unique", argsort_inp).fields["out"]
                        self.assertEqual(np.asarray(plain(argsort_out)).tolist(), order.tolist())
                        self.assertEqual(torch.argsort(tensor, dim=dim, descending=descending).tolist(), order.tolist())

                        k = 2
                        top_order = np.flip(np.argsort(data, axis=dim, kind="stable"), axis=dim)
                        take = np.arange(k)
                        top_order = np.take(top_order, take, axis=dim)
                        top_values = np.take_along_axis(data, top_order, axis=dim)
                        top_inp = {"op": "aten.topk", "args": [record, k, dim, True, True]}
                        top_out = self.adapter.reference("finite_unique", top_inp).fields["out"]
                        self.assertEqual(np.asarray(plain(top_out[1])).tolist(), top_order.tolist())
                        self.assertEqual(np.asarray(plain(top_out[0])).tolist(), top_values.tolist())
                        pv, pi = torch.topk(tensor, k, dim=dim, largest=True, sorted=True)
                        self.assertTrue(torch.equal(pv, torch.tensor(top_values.tolist(), dtype=torch_dtype)))
                        self.assertEqual(pi.tolist(), top_order.tolist())
                        checked += 3
        self.assertEqual(checked, 240)


if __name__ == "__main__":
    unittest.main()
