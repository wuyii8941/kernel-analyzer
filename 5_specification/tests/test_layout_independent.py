import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "LAYOUT.py"
    spec = importlib.util.spec_from_file_location("reviewed_layout_independent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def q(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def nested(value):
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return q(array.item())
    return [nested(item) for item in array]


def record(value, dtype="f64", shape=None):
    out = {"t": dtype, "v": nested(value)}
    if shape is not None:
        out["shape"] = list(shape)
    return out


def normalize(value):
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return q(value)


class LayoutIndependentEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def result(self, op, args, kwargs=None):
        call = {"op": op, "args": args}
        if kwargs:
            call["kwargs"] = kwargs
        return normalize(self.adapter.reference("real", call).fields["out"])

    def test_axis_shape_and_range_operations_match_separate_numpy_indexing(self):
        x = np.arange(24, dtype=object).reshape(2, 3, 4) - 8
        xr = record(x)
        cases = (
            ("aten.reshape", [xr, [4, 6]], np.reshape(x, (4, 6))),
            ("aten.permute", [xr, [2, 0, 1]], np.transpose(x, (2, 0, 1))),
            ("aten.transpose", [xr, 0, 2], np.swapaxes(x, 0, 2)),
            ("aten.unsqueeze", [xr, 1], np.expand_dims(x, 1)),
            ("aten.squeeze", [record(np.expand_dims(x, 0)), [0]], x),
            ("aten.select", [xr, 1, -1], x[:, -1, :]),
            ("aten.slice", [xr, 2, -3, 100, 2], x[:, :, -3:100:2]),
            ("aten.narrow", [xr, -1, -3, 2], x[:, :, 1:3]),
        )
        for op, args, expected in cases:
            with self.subTest(op=op):
                self.assertEqual(self.result(op, args), nested(expected))

    def test_join_split_padding_scatter_and_triangles_match_independent_arrays(self):
        a = np.asarray([[1, 2], [3, 4]], dtype=object)
        b = np.asarray([[5, 6]], dtype=object)
        self.assertEqual(self.result("aten.cat", [[record(a), record(b)], 0]), nested(np.concatenate([a, b], 0)))
        self.assertEqual(self.result("aten.stack", [[record(a), record(a + 10)], 1]),
                         nested(np.stack([a, a + 10], 1)))
        self.assertEqual(self.result("aten.split", [record(np.arange(7, dtype=object)), 3]),
                         [nested(np.arange(7, dtype=object)[start:start + 3]) for start in (0, 3, 6)])

        padded = np.pad(np.asarray([[1, 2]], dtype=object), ((0, 1), (1, 0)), constant_values=Fraction(1, 2))
        self.assertEqual(self.result("aten.constant_pad_nd", [record([[1, 2]]), [1, 0, 0, 1], "1/2"]),
                         nested(padded))
        cropped = np.pad(np.asarray([1, 2, 3], dtype=object)[1:], (0, 1), constant_values=0)
        self.assertEqual(self.result("aten.constant_pad_nd", [record([1, 2, 3]), [-1, 1]]), nested(cropped))

        base = np.zeros((3, 4), dtype=object)
        selected = base.copy(); selected[1] = np.asarray([5, 6, 7, 8], dtype=object)
        self.assertEqual(self.result("aten.select_scatter", [record(base), record([5, 6, 7, 8]), 0, 1]),
                         nested(selected))
        sliced = base.copy(); sliced[:, 1:4:2] = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=object)
        self.assertEqual(self.result("aten.slice_scatter", [record(base), record([[1, 2], [3, 4], [5, 6]]), 1, 1, 4, 2]),
                         nested(sliced))

        matrix = np.arange(18, dtype=object).reshape(2, 3, 3) - 4
        self.assertEqual(self.result("aten.triu", [record(matrix), 1]), nested(np.triu(matrix, 1)))
        self.assertEqual(self.result("aten.tril", [record(matrix), -1]), nested(np.tril(matrix, -1)))

    def test_public_torch_calls_match_for_all_reviewed_dtypes(self):
        dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16,
                     "f32": torch.float32, "f64": torch.float64}
        for name, dtype in dtype_map.items():
            x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(dtype)
            source = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(dtype)
            calls = (
                ("aten.reshape", [record(x.tolist(), name), [4, 6]], None, x.reshape(4, 6)),
                ("aten.permute", [record(x.tolist(), name), [2, 0, 1]], None, x.permute(2, 0, 1)),
                ("aten.expand", [record(x[:, :1, :].tolist(), name), [2, 3, 4]], None,
                 x[:, :1, :].expand(2, 3, 4)),
                ("aten.slice", [record(x.tolist(), name), 2, 1, 4, 2], None, x[:, :, 1:4:2]),
                ("aten.select_scatter", [record(x.tolist(), name), record(source.tolist(), name), 1, 1], None,
                 torch.select_scatter(x, source, 1, 1)),
                ("aten.constant_pad_nd", [record(x.tolist(), name), [1, 0], 0], None, F.pad(x, [1, 0])),
                ("aten.triu", [record(x.tolist(), name), 1], None, torch.triu(x, 1)),
            )
            for op, args, kwargs, actual in calls:
                with self.subTest(dtype=name, op=op):
                    exact = self.result(op, args, kwargs)
                    expected = torch.tensor(exact, dtype=dtype)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                    self.assertTrue(torch.equal(actual, expected), (name, op, actual, expected))


if __name__ == "__main__":
    unittest.main()
