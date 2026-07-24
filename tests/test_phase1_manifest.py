from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.write_phase1_manifest import summarize_pair


class Phase1ManifestTest(unittest.TestCase):
    def test_small_fixture_is_not_claim_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.jsonl"
            path.write_text(
                '{"case_id":"a","logprob_delta":0.01,"delta_self_ref":0.0,"delta_self_alt":0.0}\n'
                '{"case_id":"b","logprob_delta":0.02,"delta_self_ref":0.0,"delta_self_alt":0.0}\n',
                encoding="utf-8",
            )

            summary = summarize_pair("pair", "claim", path)

        self.assertTrue(summary["self_gate"])
        self.assertFalse(summary["scale_gate"])
        self.assertFalse(summary["weights_gate"])
        self.assertFalse(summary["determinism_gate"])
        self.assertFalse(summary["pair_gate"])


if __name__ == "__main__":
    unittest.main()
