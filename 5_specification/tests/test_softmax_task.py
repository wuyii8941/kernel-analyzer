import copy
import json
import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from kernel_spec.num import NotEstablished
from softmax_task import require_region_formula
from softmax_reference import evaluate_region


class SoftmaxTask(unittest.TestCase):
    def setUp(self):
        # Actual complete graph extracted from the captured OPT training region.
        self.graph = json.loads((Path(__file__).parent / "data/softmax_region_graph.json").read_text())

    def check(self, graph):
        return require_region_formula(graph, logits_input="mm", labels_input="constant_pad_nd", logits_shape=(128, 50272))

    def test_actual_graph_and_renamed_nodes_use_the_same_formulas(self):
        result = self.check(self.graph)
        self.assertEqual(result, {"mean_loss": "div", "maxima": "getitem", "log_mass": "log", "count": "convert_element_type_3"})
        names = {n["name"]: f"renamed_{i}" for i, n in enumerate(self.graph["nodes"])}
        def rename(x):
            if isinstance(x, list):
                return [rename(v) for v in x]
            if isinstance(x, dict):
                return {k: names.get(v, v) if k in ("ref", "name") else rename(v) for k, v in x.items()}
            return x
        changed = rename(self.graph)
        changed["outputs"] = [names[n] for n in reversed(self.graph["outputs"])]
        self.assertEqual(self.check(changed), {k: names[v] for k, v in result.items()})

    def test_semantic_changes_do_not_get_the_original_reference(self):
        changes = [("ne", 1, -1), ("slice_1", 2, 0), ("prepare_softmax_online_default", 1, 0),
                   ("getitem", 1, 1), ("div", 1, 128), ("gather", 1, 0)]
        for name, index, replacement in changes:
            graph = copy.deepcopy(self.graph)
            next(n for n in graph["nodes"] if n["name"] == name)["args"][index] = replacement
            with self.subTest(name=name), self.assertRaises(NotEstablished):
                self.check(graph)

    def test_missing_output_or_extra_work_is_not_ignored(self):
        graph = copy.deepcopy(self.graph)
        graph["outputs"].pop()
        with self.assertRaises(NotEstablished):
            self.check(graph)

        graph = copy.deepcopy(self.graph)
        graph["nodes"].append({"name": "extra", "op": "aten.neg", "args": [{"ref": "mm"}], "kwargs": {}})
        with self.assertRaisesRegex(NotEstablished, "additional nodes"):
            self.check(graph)

    def test_guarded_evaluation_enforces_the_input_cast(self):
        graph = copy.deepcopy(self.graph)
        for n in graph["nodes"]:
            if n["name"] == "view_1":
                n["args"][1] = [1, 2, 3]
            if n["name"] == "view_2":
                n["args"][1] = [-1, 3]
        inputs = {"mm": np.array([[2., 2., 2.], [4., 4., 4.]]), "constant_pad_nd": np.array([[9, 0, -100]])}
        result = evaluate_region(graph, inputs, logits_input="mm", labels_input="constant_pad_nd")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["maxima"], [(2, 2), (4, 4)])
        inputs["mm"][0, 0] = np.nextafter(2., 3.)
        with self.assertRaisesRegex(NotEstablished, "conversion is not exact"):
            evaluate_region(graph, inputs, logits_input="mm", labels_input="constant_pad_nd")


if __name__ == "__main__":
    unittest.main()
