import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from region_outputs import bind_outputs


class RegionOutputs(unittest.TestCase):
    def test_absent_return_is_not_dropped_or_counted_as_storage(self):
        absent = {"position": 1, "constant": None}
        result = bind_outputs(self.graph, self.returns + [absent], [(1, "out", self.arg)], {1, 2})
        self.assertEqual(result["constant_returns"], [absent])
        self.assertEqual(len(result["outputs"]), 1)
        self.assertEqual(result["unreturned_written_storages"], [2])
        for bad in ({"position": 1, "constant": 0}, {"position": 1, "constant": None, "buffer": "out"}):
            with self.assertRaises(ValueError):
                bind_outputs(self.graph, [bad], [], set())

    def setUp(self):
        self.arg = SimpleNamespace(storage_id=1, storage_ptr=100, data_ptr=100, storage_nbytes=8,
                                   dtype="float32", element_size=4, shape=(2,), stride=(1,), window=None,
                                   after=np.zeros(8, dtype=np.uint8))
        self.graph = {"nodes": [{"name": "mean", "value": {"kind": "tensor", "dtype": "torch.float32", "shape": ["2"]}}]}
        self.returns = [{"position": 0, "node": "mean", "buffer": "out", "view": None}]

    def test_final_storage_from_alias_and_duplicate_positions_are_kept(self):
        alias = copy.deepcopy(self.arg)
        alias.after[0] = 7
        result = bind_outputs(self.graph, self.returns * 2, [(1, "out", self.arg), (2, "alias", alias)], {1, 2})
        self.assertEqual(len(result["outputs"]), 2)
        self.assertEqual(result["outputs"][0]["arg"].after[0], 7)
        self.assertEqual(result["unreturned_written_storages"], [2])
        self.assertEqual(result["binding_status"], "not_established")

    def test_passthrough_and_unobserved_returns_are_not_computed_outputs(self):
        result = bind_outputs(self.graph, self.returns, [(1, "out", self.arg)], set())
        self.assertEqual(result["outputs"], [])
        self.assertEqual(result["passthrough_returns"], self.returns)
        result = bind_outputs(self.graph, self.returns, [], {1})
        self.assertEqual(result["unobserved_returns"], self.returns)
        self.assertEqual(result["unreturned_written_storages"], [1])

    def test_wrong_type_shape_bounds_and_reused_name_are_rejected(self):
        for attr, value in (("dtype", "int32"), ("shape", (1,)), ("storage_nbytes", 4), ("window", [0])):
            arg = copy.deepcopy(self.arg)
            setattr(arg, attr, value)
            with self.assertRaises(ValueError):
                bind_outputs(self.graph, self.returns, [(1, "out", arg)], {1})
        other = copy.copy(self.arg)
        other.storage_id = 2
        with self.assertRaises(ValueError):
            bind_outputs(self.graph, self.returns, [(1, "out", self.arg), (2, "out", other)], {1, 2})
        self.graph["nodes"][0]["value"]["shape"] = [2.9]
        with self.assertRaises(ValueError):
            bind_outputs(self.graph, self.returns, [(1, "out", self.arg)], {1})


if __name__ == "__main__":
    unittest.main()
