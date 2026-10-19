import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from region_inputs import bind_inputs


class RegionInputs(unittest.TestCase):
    def setUp(self):
        arg = SimpleNamespace(name="p", storage_id=1, dtype="float32", shape=(2,), stride=(1,),
                              element_size=4, storage_nbytes=8, data_ptr=100, storage_ptr=100,
                              before=b"bytes", window=None)
        self.row = {"launch": 2, "buffer": "buf", "arg": arg, "write_only": False,
                    "provenance": {"buffer": "buf", "role": "input", "origin_node": "x", "origins": ["x"]}}

    def test_source_origin_selects_input_and_earliest_prestate(self):
        later = copy.deepcopy(self.row)
        later["launch"] = 3
        inputs, descriptions = bind_inputs([self.row, later], set())
        self.assertIs(inputs["x"], self.row["arg"])
        self.assertEqual(descriptions["x"]["launch"], 2)

    def test_shapes_do_not_replace_missing_or_conflicting_origins(self):
        for key, value in (("origin_node", None), ("origins", []), ("buffer", "other"), ("role", "output")):
            row = copy.deepcopy(self.row)
            row["provenance"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind_inputs([row], set())
        other = copy.deepcopy(self.row)
        other["arg"].storage_id = 2
        with self.assertRaises(ValueError):
            bind_inputs([self.row, other], set())

    def test_read_modify_write_and_same_launch_read_alias_require_prestate(self):
        with self.assertRaises(ValueError):
            bind_inputs([self.row], {1})
        write = copy.deepcopy(self.row)
        write["write_only"] = True
        with self.assertRaises(ValueError):
            bind_inputs([write, self.row], {1})
        external = copy.deepcopy(self.row)
        external["arg"].storage_id = 2
        later = copy.deepcopy(self.row)
        later["launch"] = 3
        inputs, _ = bind_inputs([write, external, later], {1})
        self.assertEqual(inputs["x"].storage_id, 2)

    def test_input_view_must_fit_the_captured_storage(self):
        for key, value in (("before", None), ("window", [0]), ("stride", (2,)), ("data_ptr", 101)):
            row = copy.deepcopy(self.row)
            setattr(row["arg"], key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind_inputs([row], set())


if __name__ == "__main__":
    unittest.main()
