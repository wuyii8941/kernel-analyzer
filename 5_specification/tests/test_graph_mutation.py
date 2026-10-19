"""A value-only graph must not silently erase in-place state transitions."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import graph, lazy
from kernel_spec.num import Unsupported


class GraphMutation(unittest.TestCase):
    def fixture(self):
        return {"inputs": {"x": {"t": "f32", "v": [0]}},
                "nodes": [{"name": "alias", "op": "aten.alias", "args": [{"ref": "x"}]},
                          {"name": "written", "op": "aten.index_put_", "args": [
                              {"ref": "x"}, [{"t": "i64", "v": [0]}], {"t": "f32", "v": [9]}]}],
                "outputs": ["x", "alias", "written"]}

    def test_mutation_cannot_be_evaluated_as_a_pure_node(self):
        for outputs in (["x"], ["alias"], ["written"], ["x", "alias", "written"]):
            g = dict(self.fixture(), outputs=outputs)
            with self.subTest(outputs=outputs):
                with self.assertRaises(Unsupported):
                    graph.evaluate(g)
                with self.assertRaises(Unsupported):
                    lazy.evaluate_lazy(g, {"x": lazy.from_array(np.array([0.]), "f32")})

    def test_functional_index_put_keeps_old_and_new_values_distinct(self):
        g = copy.deepcopy(self.fixture())
        g["nodes"][1]["op"] = "aten.index_put"
        self.assertEqual(graph.evaluate(g), [[0], [0], [9]])

    def test_static_graph_coverage_excludes_unmodeled_mutation(self):
        ops = graph.covered_ops()
        self.assertNotIn("aten.index_put_", ops)
        self.assertIn("aten.index_put", ops)


if __name__ == "__main__":
    unittest.main()
