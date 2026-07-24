from __future__ import annotations

import unittest

from scripts.phase15_attribution_ladder import propagation_exponent
from scripts.phase15_measure_hf import module_filter, transformer_layer_index


class Phase15PropagationTest(unittest.TestCase):
    def test_linear_growth_has_unit_log_log_slope(self) -> None:
        rows = [{"diff_l2": float(index)} for index in range(1, 9)]
        self.assertAlmostEqual(propagation_exponent(rows), 1.0, places=6)

    def test_insufficient_nonzero_points(self) -> None:
        self.assertIsNone(propagation_exponent([{"diff_l2": 0.0}, {"diff_l2": 1.0}]))

    def test_explicit_layer_depth_controls_slope(self) -> None:
        rows = [
            {"layer_index": 0, "diff_l2": 1.0},
            {"layer_index": 1, "diff_l2": 2.0},
            {"layer_index": 3, "diff_l2": 4.0},
        ]
        self.assertAlmostEqual(propagation_exponent(rows), 1.0, places=6)

    def test_only_full_transformer_blocks_are_recorded(self) -> None:
        self.assertEqual(transformer_layer_index("model.layers.17"), 17)
        self.assertTrue(module_filter("model.layers.17", object()))
        self.assertFalse(module_filter("model.layers.17.self_attn", object()))


if __name__ == "__main__":
    unittest.main()
