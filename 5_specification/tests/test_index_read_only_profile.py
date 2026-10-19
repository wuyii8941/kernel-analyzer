import importlib.util
import unittest
from pathlib import Path

from kernel_spec.num import Invalid, Unsupported


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "INDEX.py"
    spec = importlib.util.spec_from_file_location("reviewed_index_domain_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndexReadOnlyProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, args, kwargs=None):
        inp = {"op": op, "args": args}
        if kwargs is not None:
            inp["kwargs"] = kwargs
        return self.adapter.reference("real", inp)

    def test_selected_operations_support_all_four_physical_floating_dtypes(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            source = {"t": dtype, "v": [[1, 2], [3, 4], [5, 6]]}
            with self.subTest(dtype=dtype, op="gather"):
                out = self.call("aten.gather", [source, 0, {"t": "i64", "v": [[2, 1], [0, 2]]}])
                self.assertEqual(out.fields["out"], [[5, 4], [1, 6]])
            with self.subTest(dtype=dtype, op="index_select"):
                out = self.call("aten.index_select", [source, 0, {"t": "i64", "v": [2, 0, 2]}])
                self.assertEqual(out.fields["out"], [[5, 6], [1, 2], [5, 6]])
            with self.subTest(dtype=dtype, op="embedding"):
                out = self.call("aten.embedding", [source, {"t": "i64", "v": [[2, 0]]}])
                self.assertEqual(out.fields["out"], [[[5, 6], [1, 2]]])

    def test_source_values_must_be_finite_physical_values(self):
        index = {"t": "i64", "v": [0]}
        for record in ({"t": "i64", "v": [1]}, {"t": "b8", "v": [True]},
                       {"t": "complex64", "v": [1]}, {"t": "f32", "v": ["inf"]}):
            with self.subTest(record=record), self.assertRaises(Unsupported):
                self.call("aten.index_select", [record, 0, index])
        for record in ({"t": "f16", "v": ["1/3"]}, {"t": "f32", "v": [-0.0]}):
            with self.subTest(record=record), self.assertRaises(Invalid):
                self.call("aten.index_select", [record, 0, index])

    def test_indices_options_shapes_and_ranges_are_strict(self):
        source = {"t": "f32", "v": [[1, 2], [3, 4]]}
        with self.assertRaises(Unsupported):
            self.call("aten.gather", [source, 1, {"t": "i32", "v": [[0], [1]]}])
        with self.assertRaises(Invalid):
            self.call("aten.gather", [source, 1, {"t": "i64", "v": [0, 1]}])
        with self.assertRaises(Invalid):
            self.call("aten.gather", [source, 1, {"t": "i64", "v": [[-1], [0]]}])
        with self.assertRaises(Invalid):
            self.call("aten.index_select", [source, 0, {"t": "i64", "v": [[0]]}])
        with self.assertRaises(Invalid):
            self.call("aten.index_select", [source, 2, {"t": "i64", "v": [0]}])
        with self.assertRaises(Unsupported):
            self.call("aten.gather", [source, 1, {"t": "i64", "v": [[0], [1]]}], {"sparse_grad": True})
        with self.assertRaises(Unsupported):
            self.call("aten.embedding", [source, {"t": "i64", "v": [0]}], {"padding_idx": 0})

    def test_public_keyword_names_match_positional_calls(self):
        source = {"t": "f32", "v": [[1, 2], [3, 4]]}
        for op, index in (("gather", {"t": "i64", "v": [[1], [0]]}),
                          ("index_select", {"t": "i64", "v": [1, 0, 1]})):
            expected = self.call("aten." + op, [source, 1, index]).fields
            for namespace, input_name in (("aten.", "self"), ("torch.", "input")):
                with self.subTest(op=namespace + op):
                    result = self.call(namespace + op, [], {input_name: source, "dim": 1, "index": index})
                    self.assertEqual(result.fields, expected)
                    self.assertEqual(self.call(namespace + op, [source], {"dim": 1, "index": index}).fields,
                                     expected)
                    observed = {"out": {"t": "f32", "shape": [2, 1] if op == "gather" else [2, 3],
                                        "v": expected["out"]}}
                    query = {"op": namespace + op, "kwargs": {input_name: source, "dim": 1, "index": index}}
                    self.assertTrue(self.adapter.check_output("real", query, observed))

    def test_keyword_aliases_do_not_hide_duplicates_or_invalid_indices(self):
        source = {"t": "f32", "v": [[1, 2], [3, 4]]}
        index = {"t": "i64", "v": [[1], [0]]}
        for args, kwargs in (
            ([source, 1, index], {"index": index}),
            ([source], {"dim": 1, "idx": index, "index": index}),
            ([], {"a": source, "self": source, "dim": 1, "index": index}),
            ([source], {"dim": 1, "index": {"t": "i64", "v": [[2], [0]]}}),
        ):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(Invalid):
                self.call("aten.gather", args, kwargs)
        with self.assertRaises(Unsupported):
            self.call("aten.gather", [source], {"dim": 1, "index": {"t": "i32", "v": [[1], [0]]}})
        with self.assertRaises(Unsupported):
            self.call("aten.gather", [source], {"dim": 1, "index": index, "out": source})
        with self.assertRaises(Unsupported):
            self.call("aten.scatter", [source, 0, {"t": "i64", "v": [[0], [1]]}, source])

    def test_complete_output_dtype_shape_and_values_are_required(self):
        call = {"op": "aten.index_select",
                "args": [{"t": "f32", "v": [[1, 2], [3, 4]]}, 0, {"t": "i64", "v": [1, 0]}]}
        correct = {"out": {"t": "f32", "shape": [2, 2], "v": [[3, 4], [1, 2]]}}
        self.assertTrue(self.adapter.check_output("real", call, correct))
        bad = (
            {"out": {"t": "f64", "shape": [2, 2], "v": correct["out"]["v"]}},
            {"out": {"t": "f32", "shape": [4], "v": [3, 4, 1, 2]}},
            {"out": {"t": "f32", "shape": [2, 2], "v": [[1, 2], [3, 4]]}},
            {"out": {"t": "f32", "v": correct["out"]["v"]}},
            {"out": correct["out"], "extra": 1},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("real", call, observed))


if __name__ == "__main__":
    unittest.main()
