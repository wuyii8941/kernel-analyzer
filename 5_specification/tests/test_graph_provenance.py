import json
import copy
import operator
import sys
import unittest
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from graph_provenance import snapshot_graph
import dynamic_three_way as D


class GraphProvenance(unittest.TestCase):
    def test_omitted_tuple_selection_is_restored_from_definition(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        pair = graph.call_function(torch.ops.aten.var_mean.correction, (x, [0]), {"correction": 0})
        mean = graph.call_function(operator.getitem, (pair, 1))
        graph.output(mean)
        x.meta["val"] = torch.empty(2)
        mean.meta["val"] = torch.empty(())
        task = snapshot_graph(torch.fx.GraphModule({}, graph))
        frag = [{"name": mean.name, "kind": "placeholder", "meta": 'Tensor "f32[]"'}]
        params = {"in": {"role": "input", "buffer": "arg", "origin_node": "x", "origins": ["x"]}}
        full, restored = D.complete_fragment(frag, task, params, {"in": "arg"})
        self.assertEqual([n["name"] for n in full], [x.name, pair.name, mean.name])
        self.assertEqual(full[-1]["args"], [{"ref": pair.name}, 1])
        self.assertEqual(set(restored), {pair.name, mean.name})
        self.assertEqual(frag[0]["kind"], "placeholder")
        reference_graph = D.fx_text.to_graph(full)
        reference_graph["outputs"] = [mean.name]
        env = D.lazy.evaluate_lazy(reference_graph, {"x": D.lazy.from_array(np.array([1., 5.]), "f32")})
        comparison = D.full_reference(env[mean.name], np.array(3.), 1)
        self.assertTrue(comparison["all_exact_equal"])
        self.assertEqual(D.full_reference(env[mean.name], np.array(4.), 1)["relations"],
                         {"outside_reference": 1})
        # An explicitly supplied intermediate is a boundary, not recomputed.
        params["in"]["origin_node"] = mean.name
        params["in"]["origins"] = [mean.name]
        bounded, restored = D.complete_fragment(frag, task, params, {"in": "arg"})
        self.assertEqual(bounded, frag)
        self.assertEqual(restored, [])
        bad = copy.deepcopy(task)
        bad["nodes"].append(bad["nodes"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            D.complete_fragment(frag, bad, {}, {})
        bad = copy.deepcopy(task)
        bad["nodes"][2]["inputs"] = []
        with self.assertRaisesRegex(ValueError, "dependencies disagree"):
            D.complete_fragment(frag, bad, {}, {})
        bad_fragment = copy.deepcopy(full)
        bad_fragment[-1]["args"][-1] = 0
        with self.assertRaisesRegex(ValueError, "fragment and complete task graph disagree"):
            D.complete_fragment(bad_fragment, task, {}, {})

    def test_complete_graph_keeps_tuple_selection_and_output_order(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        pair = graph.call_function(torch.max, (x, 1))
        index = graph.call_function(operator.getitem, (pair, 1))
        value = graph.call_function(operator.getitem, (pair, 0))
        graph.output((index, value))
        x.meta["val"] = torch.empty(2, 3).t()
        pair.meta["val"] = (torch.empty(3), torch.empty(3, dtype=torch.int64))
        module = torch.fx.GraphModule({}, graph)
        # Recording must not execute the task or require tensor data.
        module.forward = lambda *args: self.fail("task was executed")
        saved = json.loads(json.dumps(snapshot_graph(module)))
        nodes = saved["nodes"]
        self.assertEqual(len(nodes), 5)
        self.assertEqual(nodes[0]["value"]["stride"], ["1", "3"])
        self.assertEqual(nodes[1]["value"]["kind"], "tuple")
        self.assertEqual(nodes[1]["value"]["items"][1]["dtype"], "torch.int64")
        self.assertEqual(nodes[2]["inputs"], [pair.name])
        self.assertIn(", 1)", nodes[2]["expression"])
        self.assertEqual(nodes[-1]["inputs"], [index.name, value.name])
        self.assertNotIn("value", nodes[2])
        self.assertEqual(saved["semantic_verdict"], "not_established")


if __name__ == "__main__":
    unittest.main()
