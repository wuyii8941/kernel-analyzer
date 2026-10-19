"""Corpus query reports retain evidence levels and cannot hide process failures."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import run_corpus as C
from kernel_spec.num import Result


class CorpusResults(unittest.TestCase):
    def run_report(self, factories):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            with patch.object(C, "calls", return_value=factories), contextlib.redirect_stdout(io.StringIO()):
                code = C.evaluate_calls([f"gems.{op}.float32" for op in factories], path)
            return code, [json.loads(line) for line in path.read_text().splitlines()]

    def test_reference_success_is_not_binding_success(self):
        code, rows = self.run_report({"test": lambda: Result("value", {"out": 3},
                                                            labels={"contract": "TEST", "draft_contract": True})})
        self.assertEqual(code, 0)
        self.assertTrue(rows[0]["reference_labels"]["draft_contract"])
        self.assertEqual(rows[0]["reference_status"], "value")
        self.assertEqual(rows[0]["binding_status"], "not_established")
        self.assertEqual(rows[0]["semantic_verdict"], "not_established")

    def test_errors_fail_run_and_remaining_calls_are_recorded(self):
        def broken():
            raise RuntimeError("injected failure")
        code, rows = self.run_report({"first": broken, "second": lambda: Result("implementation_error", note="bad reference"),
                                     "third": lambda: Result("value", {})})
        self.assertEqual(code, 1)
        self.assertEqual([row["outcome"] for row in rows], ["implementation_error", "implementation_error", "value"])

    def test_existing_records_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text("original\n")
            with patch.object(C, "calls") as calls, self.assertRaises(FileExistsError):
                C.evaluate_calls(["gems.test.float32"], path)
            calls.assert_not_called()
            self.assertEqual(path.read_text(), "original\n")

    def test_refusals_remain_separate_outcomes(self):
        states = ("invalid_input", "domain_undefined", "unsupported", "not_established")
        code, rows = self.run_report({state: lambda s=state: Result(s) for state in states})
        self.assertEqual(code, 0)  # the reporting process succeeded, not the queries
        self.assertEqual([row["reference_status"] for row in rows], list(states))

    def test_malformed_query_result_fails(self):
        for result in (("value", ""), Result("made_up")):
            code, rows = self.run_report({"test": lambda: result})
            self.assertEqual(code, 1)
            self.assertEqual(rows[0]["outcome"], "implementation_error")


if __name__ == "__main__":
    unittest.main()
