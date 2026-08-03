import gzip
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LigerTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with gzip.open(ROOT / "results/final/trajectory.json.gz", "rt", encoding="utf-8") as handle:
            cls.result = json.load(handle)

    def test_complete_causal_chain(self) -> None:
        results = self.result["results"]
        campaign = results["campaign"]
        self.assertEqual(
            campaign["verdict"],
            "COMPLETE_LIGER_ACCUMULATOR_REPAIR_LIVE_WEIGHT_CAUSAL_CHAIN",
        )
        self.assertTrue(all(campaign["gates"].values()))
        self.assertEqual(campaign["denominators"]["steps"], 32)
        self.assertEqual(campaign["aggregate"]["positive_frozen_carrier_projections"], 64)
        self.assertEqual(campaign["aggregate"]["carrier_projection_denominator"], 64)

    def test_every_step_is_bound_and_controlled(self) -> None:
        steps = self.result["results"]["steps"]
        self.assertEqual(len(steps["default"]), 32)
        self.assertEqual(len(steps["repair"]), 32)
        self.assertEqual(len(steps["pairs"]), 32)
        for index in range(32):
            for arm in ("default", "repair"):
                worker = steps[arm][index]
                self.assertEqual(worker["step"]["step_index"], index)
                self.assertTrue(all(worker["gates"].values()))
                self.assertEqual(
                    worker["same_weight_control"]["gradient_contrast"]["nonzero_parameter_names"],
                    ["model.embed_tokens.weight"],
                )
                self.assertGreater(
                    worker["same_weight_control"]["gradient_contrast"]["frozen_carrier_projection"],
                    0.0,
                )
            self.assertEqual(steps["pairs"][index]["step"]["step_index"], index)
            self.assertTrue(all(steps["pairs"][index]["gates"].values()))


if __name__ == "__main__":
    unittest.main()
