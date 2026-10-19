import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, graph as G, lazy as L
from kernel_spec.num import Unsupported


class LocalUpdate(unittest.TestCase):
    def fixture(self):
        return {"inputs": {}, "nodes": [
            {"name": "initial", "op": "aten.full", "args": [[3], 2], "kwargs": {"dtype": "f64"}},
            {"name": "updated", "op": "aten.index_put_", "args": [
                {"ref": "initial"}, [{"t": "i64", "v": [1, 1]}], {"t": "f64", "v": [3, 4]}, True]},
            {"name": "twice", "op": "aten.add", "args": [{"ref": "updated"}, {"ref": "updated"}]}],
            "outputs": ["updated", "twice"]}

    def test_private_allocation_matches_independent_mutating_execution(self):
        import torch
        original = self.fixture()
        before = copy.deepcopy(original)
        expected = torch.full((3,), 2., dtype=torch.float64)
        expected.index_put_((torch.tensor([1, 1]),), torch.tensor([3., 4.], dtype=torch.float64), accumulate=True)
        normalized, evidence = G.functional_value_graph(original)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(normalized["nodes"][1]["op"], "aten.index_put")
        self.assertEqual(original, before)
        self.assertEqual(G.evaluate(original), [expected.tolist(), (2 * expected).tolist()])
        env = L.evaluate_lazy(original, {})
        for name, target in (("updated", expected), ("twice", 2 * expected)):
            self.assertEqual([A.plain(env[name].get((i,))) for i in range(3)], target.tolist())

    def test_other_uses_aliases_and_returned_prestate_require_explicit_state_semantics(self):
        variants = []
        g = self.fixture(); g["outputs"].append("initial"); variants.append(g)
        g = self.fixture(); g["nodes"].insert(1, {"name": "alias", "op": "aten.alias", "args": [{"ref": "initial"}]}); variants.append(g)
        g = self.fixture(); g["nodes"][1]["args"][2] = {"ref": "initial"}; variants.append(g)
        g = self.fixture(); g["nodes"][0]["kwargs"]["out"] = {"t": "f64", "v": [0, 0, 0]}; variants.append(g)
        g = self.fixture(); g["inputs"]["initial"] = {"t": "f64", "v": [2, 2, 2]}; g["nodes"].pop(0); variants.append(g)
        for g in variants:
            with self.subTest(graph=g), self.assertRaises(Unsupported):
                G.functional_value_graph(g)


if __name__ == "__main__":
    unittest.main()
