from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.phase2_bounds import file_sha256, validate_measurement_coverage


class Phase2CoverageTest(unittest.TestCase):
    def test_requires_mapping_for_every_nonzero_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            measurements = Path(tmp) / "measurements.jsonl"
            measurements.write_text(
                '{"level":"L1","final_logprob_delta":0.01}\n'
                '{"level":"L2","final_logprob_delta":0.0}\n'
                '{"level":"L6","max_logprob_delta":0.02}\n',
                encoding="utf-8",
            )
            payload = {
                "coverage": {
                    "measurements_sha256": file_sha256(measurements),
                    "measured_levels": ["L1", "L2", "L6"],
                    "level_sources": {"L1": ["attention"]},
                }
            }

            ok, details = validate_measurement_coverage(payload, {"attention"}, measurements)

        self.assertFalse(ok)
        self.assertIn("L6", " ".join(details["failures"]))

    def test_accepts_exact_hash_and_complete_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            measurements = Path(tmp) / "measurements.jsonl"
            measurements.write_text(
                '{"level":"L1","final_logprob_delta":0.01}\n'
                '{"level":"L2","final_logprob_delta":0.0}\n',
                encoding="utf-8",
            )
            payload = {
                "coverage": {
                    "measurements_sha256": file_sha256(measurements),
                    "measured_levels": ["L1", "L2"],
                    "level_sources": {"L1": ["attention"]},
                }
            }

            ok, details = validate_measurement_coverage(payload, {"attention"}, measurements)

        self.assertTrue(ok)
        self.assertEqual(details["failures"], [])

    def test_rejects_stale_measurement_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            measurements = Path(tmp) / "measurements.jsonl"
            measurements.write_text('{"level":"L1","final_logprob_delta":0.01}\n', encoding="utf-8")
            payload = {
                "coverage": {
                    "measurements_sha256": "stale",
                    "measured_levels": ["L1"],
                    "level_sources": {"L1": ["attention"]},
                }
            }

            ok, details = validate_measurement_coverage(payload, {"attention"}, measurements)

        self.assertFalse(ok)
        self.assertIn("hash mismatch", details["failures"][0])


if __name__ == "__main__":
    unittest.main()
