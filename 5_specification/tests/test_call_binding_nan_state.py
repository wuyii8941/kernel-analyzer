"""An unchanged tensor argument holding NaN must not be reported as changed (state check compares bit values).

Path A deviation D-A4: the before/after records are separate lists whose NaN floats are distinct objects, and a
Python list comparison treats NaN as unequal to itself, so `before != after` declared an untouched input changed.
"""
import copy
import sys
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / "corpus")]
import call_binding as B
import test_call_binding as T


class NanState(unittest.TestCase):
    def fixture(self, after_x):
        plan, observation = T.CallBinding.exp_fixture(None)
        observation["tensor_arguments"]["x"] = {"before": T.tensor("f32", [float("nan"), 2.0], [2]),
                                                "after": T.tensor("f32", after_x, [2])}
        return plan, observation

    def reasons(self, report):
        return " ".join(str(v) for v in (report.get("reason"), report.get("binding_review_reasons"),
                                         report.get("binding_problems")))

    def test_unchanged_nan_argument_is_unchanged(self):
        report = B.inspect(*self.fixture([float("nan"), 2.0]))
        self.assertNotIn("changed but the plan declares it unchanged", self.reasons(report))
        # the state check passes; the NaN input then meets the task's own input domain, not a state refusal
        self.assertNotEqual(report["reference_status"], "not_evaluated")

    def test_changed_argument_is_still_reported(self):
        report = B.inspect(*self.fixture([float("nan"), 3.0]))
        self.assertIn("changed but the plan declares it unchanged", self.reasons(report))

    def test_nan_replaced_by_number_is_reported(self):
        report = B.inspect(*self.fixture([1.0, 2.0]))
        self.assertIn("changed but the plan declares it unchanged", self.reasons(report))


if __name__ == "__main__":
    unittest.main()
