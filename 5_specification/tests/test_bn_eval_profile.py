"""Admission boundaries and complete outputs for reviewed BatchNorm evaluation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class BatchNormEvalProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("BN").adapter

    @staticmethod
    def tensor(dtype, values, shape):
        return {"t": dtype, "v": values, "shape": shape}

    def test_complete_values_dtype_and_shape_are_required(self):
        inp = {"op": "torch.nn.functional.batch_norm", "args": [
            {"t": "f32", "v": [[1, 2], [5, 8]]},
            {"t": "f32", "v": [1, 2]}, {"t": "f32", "v": [3, 8]},
            {"t": "f32", "v": [2, -1]}, {"t": "f32", "v": [0, 5]},
            False, "1/10", 1]}
        good = {"out": self.tensor("f32", [[0, 5], [4, 3]], [2, 2])}
        self.assertTrue(self.adapter.check_output("finite_eval_values", inp, good))
        for bad in (
            {"out": [[0, 5], [4, 3]]},
            {"out": self.tensor("f16", [[0, 5], [4, 3]], [2, 2])},
            {"out": self.tensor("f32", [[0, 5], [4, 4]], [2, 2])},
            {"out": self.tensor("f32", [[0, 5], [4, 3]], [1, 2, 2])},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("finite_eval_values", inp, bad))

    def test_public_and_aten_functional_forms_agree(self):
        tensors = [
            {"t": "f32", "v": [[1, 2], [5, 8]]},
            {"t": "f32", "v": [1, 2]}, {"t": "f32", "v": [3, 8]},
            {"t": "f32", "v": [2, -1]}, {"t": "f32", "v": [0, 5]},
        ]
        public = {"op": "torch.nn.functional.batch_norm",
                  "args": [tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], False, 0, 1]}
        aten = {"op": "aten.batch_norm",
                "args": [tensors[0], tensors[3], tensors[4], tensors[1], tensors[2], False, 0, 1, False]}
        self.assertEqual(self.adapter.reference("finite_eval_values", public).fields,
                         self.adapter.reference("finite_eval_values", aten).fields)

    def test_unselected_semantics_and_malformed_inputs_are_refused(self):
        base = [{"t": "f32", "v": [[1, 2]]}, {"t": "f32", "v": [0, 0]},
                {"t": "f32", "v": [1, 1]}, None, None, False, "1/10", 1]
        cases = [
            ({"op": "torch.nn.functional.batch_norm", "args": base[:5] + [True, "1/10", 1]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [base[0], None] + base[2:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": base[:-1] + [0]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [base[0], base[1],
              {"t": "f32", "v": [1, -1]}] + base[3:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [{"t": "i64", "v": [[1, 2]]}] + base[1:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [{"t": "f32", "v": [1, 2]}] + base[1:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [base[0], {"t": "f32", "v": [0]}] + base[2:]}, Invalid),
            ({"op": "torch.nn.functional.batch_norm", "args": [base[0], {"t": "f64", "v": [0, 0]}] + base[2:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [{"t": "f32", "v": [[1, "nan"]]}] + base[1:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [{"t": "f32", "v": [[1, "-0.0"]]}] + base[1:]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": [{"t": "f32", "v": [[1, "1/3"]]}] + base[1:]}, Invalid),
            ({"op": "torch.nn.functional.batch_norm", "args": base[:5] + [0, "1/10", 1]}, Invalid),
            ({"op": "torch.nn.functional.batch_norm", "args": base[:-1] + [float("nan")]}, Unsupported),
            ({"op": "torch.nn.functional.batch_norm", "args": base[:6] + [float("inf"), 1]}, Unsupported),
        ]
        for inp, error in cases:
            with self.subTest(inp=inp), self.assertRaises(error):
                self.adapter.reference("finite_eval_values", inp)


if __name__ == "__main__":
    unittest.main()
