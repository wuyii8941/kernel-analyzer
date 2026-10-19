import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import graph, lazy
from kernel_spec.num import Invalid, Unsupported


class GraphStructure(unittest.TestCase):
    def fixture(self, nodes, outputs):
        return {"inputs": {"x": {"t": "f32", "v": [2]}}, "nodes": nodes, "outputs": outputs}

    def both_reject(self, value, error=Invalid):
        for engine in (lambda: graph.evaluate(value),
                       lambda: lazy.evaluate_lazy(value, {"x": lazy.from_array(np.array([2.]), "f32")})):
            with self.assertRaises(error):
                engine()

    def test_input_and_node_names_cannot_be_overwritten(self):
        node = {"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 3]}
        self.both_reject(self.fixture([dict(node, name="x")], ["x"]))
        self.both_reject(self.fixture([node, dict(node, args=[{"ref": "y"}, 2])], ["y"]))

    def test_unknown_and_forward_references_are_invalid_even_in_unused_nodes(self):
        for ref in ("missing", "later"):
            nodes = [{"name": "y", "op": "aten.mul", "args": [{"ref": ref}, 3]},
                     {"name": "later", "op": "aten.mul", "args": [{"ref": "x"}, 2]}]
            self.both_reject(self.fixture(nodes, ["x"]))
        self.both_reject(self.fixture([], ["missing"]))

    def test_external_lazy_inputs_and_repeated_outputs_remain_valid(self):
        value = self.fixture([{"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 3]}], ["y", "y"])
        self.assertEqual(graph.evaluate(value), [[6], [6]])
        value.pop("inputs")
        env = lazy.evaluate_lazy(value, {"x": lazy.from_array(np.array([2.]), "f32")})
        self.assertEqual(env["y"].get((0,)).v, 6)

    def test_malformed_containers_are_not_silently_interpreted(self):
        self.both_reject(self.fixture([], "x"))
        self.both_reject(self.fixture({}, ["x"]))
        for field, value in (("name", []), ("args", {}), ("kwargs", [])):
            node = {"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 3]}
            node[field] = value
            self.both_reject(self.fixture([node], ["x"]))

    def test_reference_metadata_is_not_ignored(self):
        for reference, error in (({"ref": "x", "offset": 1}, Unsupported),
                                 ({"ref": "x", "index": 0.5}, Invalid),
                                 ({"ref": "x", "index": "0"}, Invalid)):
            self.both_reject(self.fixture([], [reference]), error)
            node = {"name": "y", "op": "aten.mul", "args": [reference, 2]}
            self.both_reject(self.fixture([node], ["x"]), error)

    def test_getitem_cannot_ignore_extra_arguments(self):
        pair = {"name": "pair", "op": "aten.var_mean", "args": [{"ref": "x"}, [0]],
                "kwargs": {"correction": 0}}
        for args, kwargs in (([{"ref": "pair"}, 0, 1], {}),
                             ([{"ref": "pair"}, 0], {"unexpected": 1}), ([{"ref": "pair"}], {})):
            item = {"name": "y", "op": "operator.getitem", "args": args, "kwargs": kwargs}
            self.both_reject(self.fixture([pair, item], ["y"]))

    def test_unimplemented_graph_and_node_fields_are_not_discarded(self):
        value = self.fixture([], ["x"])
        value["post_state"] = ["x"]
        self.both_reject(value, Unsupported)
        node = {"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 2], "out": "x"}
        self.both_reject(self.fixture([node], ["y"]), Unsupported)


if __name__ == "__main__":
    unittest.main()
