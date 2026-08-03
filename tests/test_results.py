import unittest
import json
from pathlib import Path

from scripts.check import main


class FinalResultsTest(unittest.TestCase):
    def test_package(self) -> None:
        main()

    def test_next_round_controls_are_separated(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        flash = json.loads((root / "flash_control.json").read_text())
        atlas = json.loads((root / "implementation_atlas.json").read_text())
        self.assertEqual(flash["kind"], "PAPER_REFERENCE_REPRODUCTION")
        self.assertIn("real_sdpa", flash)
        self.assertEqual(atlas["denominator"], len(atlas["rows"]))
        self.assertGreater(atlas["exact_replay_count"], 0)
        self.assertGreater(atlas["changed_count"], 0)


if __name__ == "__main__":
    unittest.main()
