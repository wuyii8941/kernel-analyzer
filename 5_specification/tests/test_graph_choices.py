"""Graph results preserve dependence between outputs of one sorting operation."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import graph as G
from kernel_spec.num import NotEstablished


class GraphChoices(unittest.TestCase):
    def graph(self):
        return {"inputs": {"x": {"t": "f32", "v": [2, 2]}},
                "nodes": [{"name": "i", "op": "aten.argsort", "args": [{"ref": "x"}]}], "outputs": ["i"]}

    def test_joint_permutations_exclude_duplicate_indices(self):
        graph = self.graph()
        self.assertEqual(G.evaluate(graph), frozenset({((0, 1),), ((1, 0),)}))

    def test_repeated_use_is_the_same_choice(self):
        graph = self.graph()
        graph["nodes"].append({"name": "zero", "op": "aten.sub", "args": [{"ref": "i"}, {"ref": "i"}]})
        graph["outputs"] = ["zero"]
        self.assertEqual(G.evaluate(graph), [[0, 0]])

    def test_permutation_sum_is_invariant(self):
        graph = self.graph()
        graph["nodes"].append({"name": "sum", "op": "aten.sum", "args": [{"ref": "i"}]})
        graph["outputs"] = ["sum"]
        self.assertEqual(G.evaluate(graph), [1])

    def test_separate_nodes_can_choose_differently(self):
        graph = self.graph()
        graph["nodes"] += [{"name": "j", "op": "aten.argsort", "args": [{"ref": "x"}]},
                           {"name": "diff", "op": "aten.sub", "args": [{"ref": "i"}, {"ref": "j"}]}]
        graph["outputs"] = ["diff"]
        self.assertEqual(G.evaluate(graph), frozenset({((0, 0),), ((1, -1),), ((-1, 1),)}))

    def test_budget_does_not_return_a_partial_set(self):
        graph = self.graph()
        graph["inputs"]["x"]["v"] = [2] * 4
        original = copy.deepcopy(graph)
        with self.assertRaises(NotEstablished):
            G.evaluate(graph, max_choices=10)
        self.assertEqual(graph, original)


if __name__ == "__main__":
    unittest.main()
