"""Selective state updates must not invent an order for concurrent rows."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


class StateMapping(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load("SSU").adapter
        self.inp = dict(
            state=[[[10]], [[20]]], x=[[1], [2]], dt=[[1], [1]], A=[[0]],
            B=[[1], [1]], C=[[1], [1]], D=[0], z=None, dt_bias=[0],
            dt_softplus=False, state_batch_indices=[1, 0], dtype="f32",
        )

    def test_duplicate_live_slots_have_no_declared_serial_order(self):
        inp = dict(self.inp, A=[[-1]], state_batch_indices=[0, 0])
        original = copy.deepcopy(inp)
        result = evaluate(self.adapter.reference, "mamba232_real", inp)
        self.assertEqual(result.outcome, "not_established")
        self.assertIn("concurrent", result.note)
        self.assertIsNone(result.fields)
        self.assertEqual(inp, original)

    def test_distinct_slots_use_their_own_prestate(self):
        original = copy.deepcopy(self.inp)
        result = self.adapter.reference("mamba232_real", self.inp)
        self.assertEqual(result.fields, {"out": {"y": [[21], [12]], "state": [[[12]], [[21]]]}})
        self.assertEqual(self.inp, original)

    def test_repeated_padding_is_not_a_state_collision(self):
        result = self.adapter.reference("mamba232_real", dict(self.inp, state_batch_indices=[-1, -1]))
        self.assertEqual(result.fields, {"out": {"y": [[0], [0]], "state": self.inp["state"]}})


if __name__ == "__main__":
    unittest.main()
