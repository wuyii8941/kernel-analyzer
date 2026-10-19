"""Essential workload composition keeps exact values/enclosures across operations."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import run_essential as E
from kernel_spec import graph, expr
from kernel_spec.num import Result, ends


class EssentialComposition(unittest.TestCase):
    def test_softmax_backward_encloses_composed_formula(self):
        evaluated = []
        def query(cid, profile, inputs, **kwargs):
            self.assertEqual(cid, "GRAPH")
            result = graph.evaluate(inputs["graph"])
            evaluated.append(result)
            return Result("value", {"out": result})
        with patch.object(E.registry, "query", side_effect=query):
            statuses = E.fam_reductions({"op": "softmax", "dims": "single"},
                                       {"x": np.array([[0., 1.]]), "g": np.array([[1., 0.]])})
        self.assertEqual(statuses["dx"][0], "value")
        self.assertEqual(len(evaluated), 1)
        dx = evaluated[0][1][0]
        for actual, expected in zip(dx, ("exp(1)/(1+exp(1))**2", "-exp(1)/(1+exp(1))**2")):
            lo, hi = ends(actual)
            el, eh = ends(expr.parse(expected))
            self.assertLessEqual(lo, el)
            self.assertGreaterEqual(hi, eh)

    def test_batched_bias_does_not_use_numpy_matmul_intermediate(self):
        outputs = []
        def query(cid, profile, inputs, **kwargs):
            self.assertEqual(cid, "GRAPH")
            outputs.append(graph.evaluate(inputs["graph"]))
            return Result("value", {"out": outputs[-1]})
        tensors = {"a": np.array([[[2.**53, 1., -2.**53]]]), "b": np.ones((1, 3, 1)),
                   "bias": np.array([0.]), "g": np.ones((1, 1, 1))}
        with patch.object(E, "q", return_value=("value", "")), patch.object(E.registry, "query", side_effect=query):
            E.fam_matmul({}, tensors)
        self.assertEqual(outputs, [[[[[1]]]]])

    def test_failed_composition_cannot_report_backward_success(self):
        with patch.object(E.registry, "query", return_value=Result("not_established", note="graph not enabled")):
            statuses = E.fam_reductions({"op": "log_softmax", "dims": "single"},
                                       {"x": np.array([[0., 1.]]), "g": np.array([[1., 0.]])})
        self.assertEqual(statuses["out"][0], "not_established")
        self.assertEqual(statuses["dx"][0], "not_established")


if __name__ == "__main__":
    unittest.main()
