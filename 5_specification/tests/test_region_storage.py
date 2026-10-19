import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from region_storage import check_sequence as check_captured_sequence


def check_sequence(launches):
    return check_captured_sequence(launches, same_recorder_run=True, storages_retained=True)


class RegionStorage(unittest.TestCase):
    def setUp(self):
        arg = SimpleNamespace(kind="tensor", name="out", storage_id=7, storage_ptr=100,
                              storage_nbytes=3, window=None,
                              before=np.array([0, 0, 0], dtype=np.uint8),
                              after=np.array([1, 2, 3], dtype=np.uint8))
        self.first = SimpleNamespace(index=12, executed=True, implicit_recorded=True, implicit=[], args=[arg])
        self.second = copy.deepcopy(self.first)
        self.second.index = 13
        self.second.args[0].name = "in"
        self.second.args[0].before = arg.after.copy()

    def test_complete_bytes_link_even_if_parameter_names_differ(self):
        result = check_sequence([self.first, self.second])
        self.assertEqual(len(result["links"]), 1)
        self.assertEqual(result["links"][0]["bytes"], 3)
        self.assertEqual(result["semantic_verdict"], "not_established")

    def test_reused_address_does_not_establish_storage_identity(self):
        self.second.args[0].storage_id = 8
        self.assertEqual(check_sequence([self.first, self.second])["links"], [])

    def test_changes_missing_launches_and_incomplete_captures_are_rejected(self):
        mutations = [lambda x: setattr(x, "index", 14), lambda x: setattr(x, "executed", None),
                     lambda x: setattr(x.args[0], "window", [0]),
                     lambda x: setattr(x.args[0], "storage_id", None),
                     lambda x: setattr(x.args[0], "storage_nbytes", 2),
                     lambda x: x.args[0].before.__setitem__(2, 4)]
        for mutate in mutations:
            item = copy.deepcopy(self.second)
            mutate(item)
            self.assertEqual(check_sequence([self.first, item])["status"], "not_established")

    def test_conflicting_aliases_are_rejected(self):
        alias = copy.deepcopy(self.second.args[0])
        alias.after[0] = 9
        self.second.args.append(alias)
        self.assertEqual(check_sequence([self.first, self.second])["status"], "not_established")

    def test_unestablished_lifetimes_or_run_are_rejected(self):
        self.assertEqual(check_captured_sequence([self.first, self.second])["status"], "not_established")
        self.assertEqual(check_captured_sequence([self.first, self.second], same_recorder_run=True)["status"],
                         "not_established")


if __name__ == "__main__":
    unittest.main()
