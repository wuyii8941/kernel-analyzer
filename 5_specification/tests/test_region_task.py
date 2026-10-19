import copy
import operator
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from graph_provenance import snapshot_graph
from region_task import build_task_region
import dynamic_three_way as D


class RegionTask(unittest.TestCase):
    def setUp(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        pair = graph.call_function(torch.ops.aten.var_mean.correction, (x, [0]), {"correction": 0})
        var = graph.call_function(operator.getitem, (pair, 0))
        mean = graph.call_function(operator.getitem, (pair, 1))
        graph.output((var, mean))
        x.meta["val"] = torch.empty(2)
        var.meta["val"] = mean.meta["val"] = torch.empty(())
        self.task = snapshot_graph(torch.fx.GraphModule({}, graph))
        self.outputs = [var.name, mean.name]
        self.arg = SimpleNamespace(dtype="float32", shape=(2,), before=np.array([1., 5.]), window=None)

    def test_region_keeps_all_outputs_and_exact_tuple_selections(self):
        graph = build_task_region(self.task, {"x": self.arg}, self.outputs)
        env = D.lazy.evaluate_lazy(graph, {"x": D.lazy.from_array(self.arg.before, "f32")})
        for name, expected in zip(self.outputs, (4., 3.)):
            self.assertTrue(D.full_reference(env[name], np.array(expected), 1)["all_exact_equal"])

    def test_missing_wrong_or_incomplete_input_is_refused(self):
        for inputs in ({}, {"unknown": self.arg}):
            with self.assertRaises(ValueError):
                build_task_region(self.task, inputs, self.outputs)
        for key, value in (("dtype", "int32"), ("shape", (1,)), ("before", None), ("window", [0])):
            arg = copy.copy(self.arg)
            setattr(arg, key, value)
            with self.assertRaises(ValueError):
                build_task_region(self.task, {"x": arg}, self.outputs)


if __name__ == "__main__":
    unittest.main()
