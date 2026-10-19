import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import check_region as C


class RegionCaptureEntry(unittest.TestCase):
    def test_every_requested_launch_must_be_present_once(self):
        manifest = {"saved": [{"launch_index": i, "dir": f"unit{i}", "kernel": f"k{i}"} for i in (4, 5)]}
        self.assertEqual(len(C.select_launches(manifest, 4, 5)), 2)
        for first, last in ((3, 5), (4, 6), (5, 4), (True, 5)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                C.select_launches(manifest, first, last)
        manifest["saved"].append(copy.deepcopy(manifest["saved"][0]))
        with self.assertRaises(ValueError):
            C.select_launches(manifest, 4, 5)

    def test_call_stack_must_match_kernel_file_and_line(self):
        source = Path("wrapper.py").resolve()
        launch = SimpleNamespace(index=4, kernel_name="k", call_stack=[{"file": str(source), "line": 8}])
        region = {"calls": [{"kernel": "k", "line": 8}]}
        C.check_call_locations([launch], source, region)
        for key, value in (("file", "other.py"), ("line", 9), ("line", 8.0)):
            bad = copy.deepcopy(launch)
            bad.call_stack[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.check_call_locations([bad], source, region)
        with self.assertRaises(ValueError):
            C.check_call_locations([], source, region)

    def test_missing_recorder_evidence_is_not_reconstructed_from_old_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"saved": [
                {"launch_index": 0, "dir": "unit", "kernel": "k"}]}))
            report = C.inspect(root, root / "absent_wrapper.py", 0, 0)
            self.assertEqual(report["failed_stage"], "capture_manifest")
            self.assertIn("recorder", report["reason"])
            self.assertEqual(report["semantic_verdict"], "not_established")
            self.assertEqual(report["reference_status"], "not_evaluated")

    def test_cli_records_refusal_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"saved": []}))
            out = root / "report.json"
            args = ["--capture-dir", str(root), "--output-code", str(root / "wrapper.py"),
                    "--first", "0", "--last", "1", "--out", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main(args), 2)
            saved = out.read_text()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                C.main(args)
            self.assertEqual(out.read_text(), saved)


if __name__ == "__main__":
    unittest.main()
