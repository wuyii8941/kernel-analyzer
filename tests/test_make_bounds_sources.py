from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_bounds_sources import load_rows, source_from_measurement


class MakeBoundsSourcesTest(unittest.TestCase):
    def test_load_rows_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"level":"L6"}\n{"level":"L1"}\n', encoding="utf-8")
            rows = load_rows(path)
        self.assertEqual([row["level"] for row in rows], ["L6", "L1"])

    def test_source_from_measurement_logsoftmax_vocab_reduction(self) -> None:
        source = source_from_measurement(
            {
                "level": "L4",
                "variable": "log_softmax precision",
                "mechanism": "rounding_precision",
                "path_ref": "bf16",
                "path_alt": "bf16",
                "final_logprob_delta": 0.01,
            },
            0,
        )
        self.assertEqual(source["reduction_length"], 128000)
        self.assertEqual(source["materialization_count_delta"], 1)

    def test_default_selection_keeps_all_rows(self) -> None:
        rows = [
            {"level": "L6", "final_logprob_delta": 0.1},
            {"level": "L1", "final_logprob_delta": 0.2},
            {"level": "L4", "final_logprob_delta": 0.3},
        ]
        sorted_rows = sorted(rows, key=lambda row: abs(float(row.get("final_logprob_delta", 0.0))), reverse=True)
        self.assertEqual(len(sorted_rows), 3)
        self.assertEqual([row["level"] for row in sorted_rows], ["L4", "L1", "L6"])


if __name__ == "__main__":
    unittest.main()
