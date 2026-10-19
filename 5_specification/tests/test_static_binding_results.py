"""Static matching must preserve frozen records and cannot certify task binding."""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import bind_inductor as B


class StaticBindingResults(unittest.TestCase):
    def test_actual_call_location_can_select_one_source_candidate(self):
        with tempfile.TemporaryDirectory(dir=B.REPO / ".cache/tmp") as directory:
            root = Path(directory)
            records = []
            for name in ("first", "second"):
                path = root / f"{name}.py"
                path.write_text("def call():\n    kernel.run(x, y)\n    another.run(x, y)\n")
                records.append(dict(file=path.name, kernel="kernel", launcher="kernel"))
            frame = dict(file=str(root / "second.py"), line=2, function="call")
            with patch.object(B, "REPO", root):
                selected = B.candidates_at_call_site(records, [frame])
                self.assertEqual(selected, [records[1]])
                for frames in (None, [], [dict(frame, line=3)], [dict(frame, line=True)],
                               [dict(frame, file=str(root / "absent.py"))]):
                    self.assertEqual(B.candidates_at_call_site(records, frames), [])
                both = [frame, dict(frame, file=str(root / "first.py"))]
                self.assertEqual(B.candidates_at_call_site(records, both), records)
            self.assertEqual(len(records), 2)  # original candidate evidence is retained

    def test_writer_refuses_existing_result(self):
        with tempfile.TemporaryDirectory(dir=B.REPO / ".cache/tmp") as directory:
            path = Path(directory) / "frozen.json.gz"
            path.write_bytes(b"original record")
            with self.assertRaises(FileExistsError):
                B.write_results(path, [])
            self.assertEqual(path.read_bytes(), b"original record")

    def test_empty_inventory_is_not_success(self):
        with tempfile.TemporaryDirectory(dir=B.REPO / ".cache/tmp") as directory:
            root = Path(directory)
            units = root / "1_experiments/three_way/results/units.jsonl"
            units.parent.mkdir(parents=True)
            units.write_text("")
            with patch.object(B, "REPO", root), patch.object(B, "index_output_code") as index:
                with self.assertRaises(SystemExit) as raised:
                    B.main(["--out", str(root / "new.json.gz")])
                self.assertNotEqual(raised.exception.code, 0)
                index.assert_not_called()
            self.assertFalse((root / "new.json.gz").exists())

    def test_missing_match_is_retained_without_semantic_claim(self):
        with tempfile.TemporaryDirectory(dir=B.REPO / ".cache/tmp") as directory:
            root = Path(directory)
            units = root / "1_experiments/three_way/results/units.jsonl"
            units.parent.mkdir(parents=True)
            units.write_text(json.dumps(dict(unit="units/model/config/kernel", kernel="kernel", verify="bit_identical"))+"\n")
            output = root / "new.json.gz"
            with patch.object(B, "REPO", root), patch.object(B, "index_output_code", return_value=({}, {})):
                self.assertNotEqual(B.main(["--out", str(output)]), 0)
            rows = json.loads(gzip.open(output, "rt").read())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_status"], "not_matched")
            self.assertEqual(rows[0]["binding_status"], "not_established")
            self.assertEqual(rows[0]["reference_status"], "not_evaluated")
            self.assertEqual(rows[0]["semantic_verdict"], "not_established")
            with patch.object(B, "index_output_code") as index:
                with self.assertRaises(SystemExit):
                    B.main(["--out", str(output)])
                index.assert_not_called()

    def test_multiple_source_matches_are_not_reduced_to_first_task(self):
        for template in (False, True):
            with tempfile.TemporaryDirectory(dir=B.REPO / ".cache/tmp") as directory:
                root = Path(directory)
                kernel = "triton_tem_fixture" if template else "kernel"
                base = f"def {kernel}():\n    pass\n"
                captured = base + ("num_warps = 8\n" if template else "")
                source = root / "kernel.py"
                source.write_text(captured)
                units = root / "1_experiments/three_way/results/units.jsonl"
                units.parent.mkdir(parents=True)
                units.write_text(json.dumps(dict(unit="units/model/config/kernel", kernel=kernel))+"\n")
                compiled = root / ".cache/three_way/units/model/config/kernel/compiled"
                compiled.mkdir(parents=True)
                (compiled / "captured.source").write_text(f'#loc = loc("{source}":1:1)')
                for number, operation in enumerate(("add", "mul"), 1):
                    output_code = root / f".cache/inductor/c{number}/output.py"
                    output_code.parent.mkdir(parents=True)
                    body = base + (f"num_warps = {number}\n" if template else "")
                    output_code.write_text(
                        "# Topologically Sorted Source Nodes: [node]\n"
                        f"# Original ATen: [aten.{operation}]\n# Graph fragment:\n"
                        '#   %x : Tensor "f32[1]" = PlaceHolder[target=x]\n'
                        f'#   %y : Tensor "f32[1]" = call_function[target=torch.ops.aten.{operation}.Tensor](args = (%x, %x), kwargs = {{}})\n'
                        + f"compiled = async_compile.triton('{kernel}', " + "'"*3 + "\n" + body + "'"*3 + ", device_str='cuda')\n"
                        + "def call():\n    compiled.run(x)\n")
                destination = root / "new.json.gz"
                with patch.object(B, "REPO", root):
                    self.assertNotEqual(B.main(["--out", str(destination)]), 0)
                rows = json.loads(gzip.open(destination, "rt").read())
                self.assertEqual(rows[0]["candidate_status"], "ambiguous")
                self.assertIsNone(rows[0]["output_code"])
                self.assertEqual(len(rows[0]["candidates"]), 2)
                self.assertEqual(rows[0]["binding_status"], "not_established")
                self.assertEqual(rows[0]["semantic_verdict"], "not_established")
                # A captured call location can distinguish these two source
                # candidates; retain both originals and do not certify binding.
                (compiled.parent / "launch.json").write_text(json.dumps({"call_stack": [
                    dict(file=str(output_code), line=len(output_code.read_text().splitlines()), function="call")]}))
                called = root / "called.json.gz"
                with patch.object(B, "REPO", root):
                    self.assertEqual(B.main(["--out", str(called)]), 0)
                chosen = json.loads(gzip.open(called, "rt").read())[0]
                self.assertEqual(len(chosen["source_candidates"]), 2)
                self.assertEqual(len(chosen["candidates"]), 1)
                self.assertEqual(chosen["candidate_selection"], "recorded Python launcher call")
                self.assertEqual(chosen["binding_status"], "not_established")
                # With one origin left, candidate matching succeeds but still cannot certify a task.
                output_code.unlink()
                unique = root / "unique.json.gz"
                with patch.object(B, "REPO", root):
                    self.assertEqual(B.main(["--out", str(unique)]), 0)
                record = json.loads(gzip.open(unique, "rt").read())[0]
                self.assertEqual(record["candidate_status"], "matched")
                self.assertEqual(record["binding_status"], "not_established")
                self.assertEqual(record["reference_status"], "not_evaluated")
                self.assertEqual(record["semantic_verdict"], "not_established")
