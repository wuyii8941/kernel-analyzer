"""Essential-run results must preserve failures, evidence labels and old files."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import run_essential as E


class Generator:
    def conditions(self):
        return [{"id": "case"}]

    def inputs(self, condition, seed, mode):
        if seed == 9:
            raise RuntimeError("injected generator failure")
        return {}


class EssentialResults(unittest.TestCase):
    def test_failed_seed_does_not_erase_following_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            counts = E.run_family("test", Generator(), lambda c, t: {
                "out": ("value", "", {"draft_contract": True})}, [9, 0], path)
            rows = [json.loads(s) for s in path.read_text().splitlines()]
        self.assertEqual(counts["implementation_error"], 1)
        self.assertEqual([r["seed"] for r in rows], [9, 0])
        self.assertTrue(rows[1]["reference_labels"]["out"]["draft_contract"])
        self.assertEqual(rows[1]["binding_status"], "not_established")

    def test_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("original")
            with self.assertRaises(FileExistsError):
                E.run_family("test", Generator(), lambda c, t: {}, [0], path)
            self.assertEqual(path.read_text(), "original")

    def test_empty_workload_cannot_pass(self):
        generator = SimpleNamespace(conditions=lambda: [])
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "no query records"):
            E.run_family("test", generator, lambda c, t: {}, [0], Path(directory)/"records")

    def test_empty_or_malformed_results_are_not_success(self):
        for result in ({}, {"out": ("value", "")}, {"out": ("unknown", "", {})}):
            with tempfile.TemporaryDirectory() as directory:
                counts = E.run_family("test", Generator(), lambda c, t: result, [0], Path(directory)/"records")
            self.assertEqual(dict(counts), {"implementation_error": 1})

    def test_error_priority_is_independent_of_output_order(self):
        with tempfile.TemporaryDirectory() as directory:
            counts = E.run_family("test", Generator(), lambda c, t: {
                "out": ("unsupported", "", {}), "dx": ("implementation_error", "bad", {})},
                                  [0], Path(directory)/"records")
        self.assertEqual(dict(counts), {"implementation_error": 1})

    def test_main_returns_failure_and_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)/"new"
            argv = ["run_essential.py", "--out-dir", str(dest), "--family", "test", "--seeds", "9"]
            with patch.object(sys, "argv", argv), patch.object(E, "load_basic", return_value=SimpleNamespace(Test=Generator)), \
                 patch.object(E, "FAMILIES", {"test": ("Test", lambda c, t: {})}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(E.main(), 1)
                with self.assertRaises(FileExistsError):
                    E.main()


if __name__ == "__main__":
    unittest.main()
