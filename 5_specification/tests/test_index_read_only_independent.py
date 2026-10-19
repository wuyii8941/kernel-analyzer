import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "INDEX.py"
    spec = importlib.util.spec_from_file_location("reviewed_index_independent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value):
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return Fraction(value)


def record(value, dtype):
    return {"t": dtype, "v": np.asarray(value).tolist()}


def manual_gather(source, dim, index):
    out = np.empty(index.shape, dtype=object)
    for coordinate in np.ndindex(index.shape):
        selected = list(coordinate)
        selected[dim] = int(index[coordinate])
        out[coordinate] = source[tuple(selected)]
    return out


def manual_index_select(source, dim, index):
    shape = list(source.shape)
    shape[dim] = len(index)
    out = np.empty(shape, dtype=object)
    for coordinate in np.ndindex(tuple(shape)):
        selected = list(coordinate)
        selected[dim] = int(index[coordinate[dim]])
        out[coordinate] = source[tuple(selected)]
    return out


class IndexReadOnlyIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def result(self, op, args, kwargs=None):
        inp = {"op": op, "args": args}
        if kwargs:
            inp["kwargs"] = kwargs
        return normalize(self.adapter.reference("real", inp).fields["out"])

    def test_coordinate_formulas_match_independent_numpy_loops(self):
        rng = np.random.default_rng(1203)
        for trial in range(40):
            source = rng.integers(-20, 21, size=(3, 4, 5)).astype(object)
            dim = trial % 3
            index_shape = [2, 3, 4]
            index = rng.integers(0, source.shape[dim], size=index_shape, dtype=np.int64)
            expected = manual_gather(source, dim, index)
            actual = self.result("aten.gather", [record(source, "f64"), dim, record(index, "i64")])
            self.assertEqual(actual, normalize(expected.tolist()))

            selected = rng.integers(0, source.shape[dim], size=6, dtype=np.int64)
            expected = manual_index_select(source, dim, selected)
            actual = self.result("aten.index_select",
                                 [record(source, "f64"), dim, record(selected, "i64")])
            self.assertEqual(actual, normalize(expected.tolist()))

            weights = rng.integers(-20, 21, size=(7, 5)).astype(object)
            indices = rng.integers(0, 7, size=(2, 3), dtype=np.int64)
            expected = np.asarray([[weights[int(i)] for i in row] for row in indices], dtype=object)
            actual = self.result("aten.embedding", [record(weights, "f64"), record(indices, "i64")])
            self.assertEqual(actual, normalize(expected.tolist()))

    def test_public_pytorch_210_calls_match_every_reviewed_dtype(self):
        dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16,
                     "f32": torch.float32, "f64": torch.float64}
        for name, dtype in dtype_map.items():
            source = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).to(dtype)
            gather_index = torch.tensor([[[2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0]],
                                         [[1, 0, 2, 1], [0, 2, 1, 0], [2, 1, 0, 2]]], dtype=torch.int64)
            selected = torch.tensor([2, 0, 2], dtype=torch.int64)
            embedding_index = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
            cases = (
                ("aten.gather", [record(source.tolist(), name), 0, record(gather_index.tolist(), "i64")],
                 torch.gather(source, 0, gather_index)),
                ("aten.index_select", [record(source.tolist(), name), 1, record(selected.tolist(), "i64")],
                 torch.index_select(source, 1, selected)),
                ("aten.embedding", [record(source[0].tolist(), name), record(embedding_index.tolist(), "i64")],
                 F.embedding(embedding_index, source[0])),
            )
            for op, args, actual in cases:
                with self.subTest(dtype=name, op=op):
                    expected = torch.tensor(self.result(op, args), dtype=dtype)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                    self.assertTrue(torch.equal(actual, expected), (name, op, actual, expected))


if __name__ == "__main__":
    unittest.main()
