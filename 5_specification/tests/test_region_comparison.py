import copy
import sys
import unittest
from unittest.mock import patch
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / "corpus"), str(SPEC.parent / "2_tool/src")]
from kernel_spec.num import NotEstablished
from region_comparison import compare_outputs, validate_sequence, evaluate_and_compare


class RegionComparison(unittest.TestCase):
    def setUp(self):
        self.outputs, self.memory = [], {}
        values = {"mean_loss": [2], "maxima": [1, 2], "log_mass": [3, 4], "count": [2]}
        self.reference = {"output_nodes": {r: r for r in values}, "mean_loss": (2, 2),
                          "maxima": [(1, 1), (2, 2)], "log_mass": [(Fraction(11, 4), Fraction(13, 4)),
                                                                 (Fraction(15, 4), Fraction(17, 4))], "count": 2}
        for pointer, (role, points) in enumerate(values.items(), 100):
            shape = () if len(points) == 1 else (2,)
            arg = NS(storage_ptr=pointer, data_ptr=pointer + 4, storage_nbytes=20, element_size=4,
                     shape=shape, stride=() if not shape else (2,), dtype="float32")
            lo = np.zeros(5)
            for index, value in zip((1, 3), points):
                lo[index] = value
            self.memory[pointer] = NS(kind="f", index=None, dtype="float32", lo=lo, hi=lo.copy(),
                                      st=np.zeros(5), cond=np.zeros(5, dtype=bool), written=np.ones(5, dtype=bool),
                                      premise=None, iset=None)
            self.outputs.append({"arg": arg, "return": {"node": role, "position": pointer - 100}})

    def test_every_output_coordinate_uses_its_storage_view(self):
        result = compare_outputs(self.memory, self.outputs, self.reference)
        self.assertEqual(sum(r["elements"] for r in result), 6)
        self.assertEqual(result[1]["rows"][1]["g_lower"], 2)
        self.assertEqual(result[2]["rows"][0]["relation"], "zero_enclosed")
        self.assertEqual(result[0]["rows"][0]["relation"], "exact_equal")
        self.memory[101].lo[3] = self.memory[101].hi[3] = 3
        self.assertEqual(compare_outputs(self.memory, self.outputs, self.reference)[1]["rows"][1]["relation"],
                         "zero_excluded")

    def test_omissions_and_qualified_coordinates_are_refused(self):
        with self.assertRaises(NotEstablished):
            compare_outputs(self.memory, self.outputs[:-1], self.reference)
        for field, value in (("written", False), ("cond", True), ("st", 1)):
            memory = copy.deepcopy(self.memory)
            getattr(memory[101], field)[3] = value
            with self.subTest(field=field), self.assertRaises(NotEstablished):
                compare_outputs(memory, self.outputs, self.reference)
        memory = copy.deepcopy(self.memory)
        memory[101].premise = np.ones(5, dtype=bool)
        with self.assertRaises(NotEstablished):
            compare_outputs(memory, self.outputs, self.reference)

    def test_full_grid_and_unqualified_composition_are_required(self):
        launches = [NS(index=7, grid=(2, 1, 1))]
        ref = NS(mode="numerical_difference", programs=[(0, 0, 0), (1, 0, 0)], aborted={}, reasons={}, notes=[])
        result = NS(launches=[ref], external_writes=[], carry_overs=[])
        self.assertEqual(validate_sequence(launches, result)[0]["programs"], 2)
        for programs in ([(0, 0, 0)], [(0, 0, 0)] * 2, [(0, 0, 0), (2, 0, 0)]):
            bad = copy.deepcopy(result)
            bad.launches[0].programs = programs
            with self.subTest(programs=programs), self.assertRaises(NotEstablished):
                validate_sequence(launches, bad)
        for field, value in (("aborted", {(1, 0, 0): "bad"}), ("notes", ["incomplete"]),
                             ("reasons", {"unknown": 1}), ("mode", "rounding_check")):
            bad = copy.deepcopy(result)
            setattr(bad.launches[0], field, value)
            with self.subTest(field=field), self.assertRaises(NotEstablished):
                validate_sequence(launches, bad)
        result.external_writes = [{"launch": 0}]
        with self.assertRaises(NotEstablished):
            validate_sequence(launches, result)
        result.external_writes = []
        result.carry_overs = [{"evidence": "bytes alone"}]
        with self.assertRaises(NotEstablished):
            validate_sequence(launches, result)

    def test_program_budget_refuses_before_evaluation(self):
        with patch("kernel_analyzer.reference_eval.ttir_eval.evaluate_sequence") as run:
            with self.assertRaises(NotEstablished):
                evaluate_and_compare([NS(grid=(2, 1, 1))], [], {}, max_programs=1)
            run.assert_not_called()

    def test_inverted_interval_is_an_implementation_failure(self):
        self.memory[101].lo[3] = 5
        with self.assertRaises(RuntimeError):
            compare_outputs(self.memory, self.outputs, self.reference)


if __name__ == "__main__":
    unittest.main()
