import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec.num import Invalid
import region_reference as R


class RegionReference(unittest.TestCase):
    def fixture(self):
        task = {"inputs": {}, "nodes": [{"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 2]}], "outputs": ["y"]}
        inputs = {"x": {"values": np.array([1., 2.]), "dtype": "f32"}}
        outputs = [{"position": 1, "node": "y", "values": np.array([2., 4.]), "dtype": "f32"}]
        return task, inputs, outputs

    def test_complete_outputs_and_difference_do_not_certify_semantics(self):
        args = self.fixture()
        report = R.compare_task(*args, max_elements=10, max_evaluations=100)
        self.assertTrue(report["complete"])
        self.assertEqual(report["outputs"][0]["comparison"]["relations"], {"exact_equal": 2})
        self.assertEqual(report["semantic_verdict"], "not_established")
        self.assertEqual(report["reference_identity"], "captured_compiler_graph")
        self.assertIn("no independent f", report["scope"])
        args[2][0]["values"][1] = 5
        report = R.compare_task(*args, max_elements=10, max_evaluations=100)
        self.assertTrue(report["complete"])
        self.assertEqual(report["outputs"][0]["comparison"]["relations"]["outside_reference"], 1)

    def test_partial_reference_preserves_known_difference_without_certifying(self):
        def get(index):
            if index==(0,):raise R.NotEstablished('unresolved first coordinate')
            return 2
        ref=R.L.Lazy((2,),'f32',get)
        with patch.object(R.L,'evaluate_lazy',return_value={'y':ref}):
            report=R.compare_task(*self.fixture(),max_elements=10,max_evaluations=100)
        self.assertFalse(report['complete'])
        self.assertEqual(report['outcome'],'not_established')
        self.assertEqual(report['semantic_verdict'],'not_established')
        comparison=report['outputs'][0]['comparison']
        self.assertEqual(comparison['relations'],{'outside_reference':1})
        self.assertEqual(comparison['unestablished_elements'],1)

    def test_output_and_scalar_budgets_preserve_incompletion(self):
        for elements, evaluations in ((1, 100), (10, 1)):
            report = R.compare_task(*self.fixture(), max_elements=elements, max_evaluations=evaluations)
            self.assertFalse(report["complete"])
            self.assertEqual(report["outcome"], "not_established")
        self.assertTrue(R.compare_task(*self.fixture(), max_elements=10, max_evaluations=100)["complete"])

    def test_dtype_shape_and_missing_output_are_not_ignored(self):
        for field, value in (("dtype", "f64"), ("values", np.array([[2., 4.]]))):
            args = self.fixture(); args[2][0][field] = value
            report = R.compare_task(*args, max_elements=10, max_evaluations=100)
            self.assertFalse(report["complete"])
        task, inputs, outputs = self.fixture()
        with self.assertRaises(Invalid):
            R.compare_task(task, inputs, [], max_elements=10, max_evaluations=100)

    def test_repeated_return_positions_are_compared_individually(self):
        task, inputs, outputs = self.fixture()
        outputs.append(dict(outputs[0], position=2, values=np.array([2., 9.])))
        report = R.compare_task(task, inputs, outputs, max_elements=10, max_evaluations=100)
        self.assertEqual(len(report["outputs"]), 2)
        self.assertEqual(report["outputs"][1]["comparison"]["relations"]["outside_reference"], 1)


if __name__ == "__main__":
    unittest.main()
