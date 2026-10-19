"""Diagnostic comparison must not erase discrete differences or failed queries."""
import random
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import dynamic_three_way as D


class DynamicComparison(unittest.TestCase):
    def test_complete_comparison_checks_every_coordinate_and_exact_integers(self):
        actual = np.arange(40, dtype=np.int64)
        ref = SimpleNamespace(shape=(40,), get=lambda idx: int(actual[idx]) + (1 if idx == (17,) else 0))
        result = D.full_reference(ref, actual, 40)
        self.assertTrue(result["complete"])
        self.assertEqual(result["elements"], 40)
        self.assertEqual(result["relations"], {"exact_equal": 39, "outside_reference": 1})
        ref = SimpleNamespace(shape=(1,), get=lambda _: np.int64(2**53+1))
        self.assertFalse(D.full_reference(ref, np.array([float(2**53)]), 1)["all_exact_equal"])

    def test_complete_comparison_never_substitutes_sampling_for_budget_or_failure(self):
        ref = SimpleNamespace(shape=(2,), get=lambda _: self.fail("over-budget reference evaluated"))
        result = D.full_reference(ref, np.zeros(2), 1)
        self.assertEqual(result["outcome"], "not_established")
        self.assertEqual(result["elements"], 0)
        for cls, outcome in ((D.NotEstablished, "not_established"), (D.Undefined, "domain_undefined")):
            result = D.full_reference(cls("witness"), np.zeros(1), 1)
            self.assertFalse(result["complete"])
            self.assertEqual(result["outcome"], outcome)

    def test_unknown_coordinate_keeps_later_difference_visible(self):
        def get(index):
            if index == (0,):
                raise D.NotEstablished("unresolved coordinate")
            return 2
        result=D.full_reference(SimpleNamespace(shape=(3,),get=get),np.array([0,2,9]),3)
        self.assertFalse(result['complete'])
        self.assertEqual(result['outcome'],'not_established')
        self.assertEqual(result['elements'],2)
        self.assertEqual(result['relations'],{'exact_equal':1,'outside_reference':1})
        self.assertEqual(result['unestablished_elements'],1)
        self.assertEqual(result['unestablished_examples'][0]['index'],[0])

    def test_unknown_examples_are_bounded_and_do_not_claim_equality(self):
        ref=SimpleNamespace(shape=(20,),get=lambda _:D.A.AllowedSet([0,1]))
        result=D.full_reference(ref,np.zeros(20),20)
        self.assertEqual(result['unestablished_elements'],20)
        self.assertLessEqual(len(result['unestablished_examples']),3)
        self.assertFalse(result['complete'])
        self.assertFalse(result['all_exact_equal'])

    def test_budget_exhaustion_still_stops_remaining_coordinates(self):
        calls=[]
        ref=D.lazy.Lazy((100,),'i64',lambda index:calls.append(index) or 1)
        with D.lazy.evaluation_budget(2) as budget:
            result=D.full_reference(ref,np.ones(100,dtype=int),100)
        self.assertEqual(calls,[(0,),(1,)])
        self.assertEqual(budget.used,2)
        self.assertEqual(result['elements'],2)
        self.assertFalse(result['complete'])
        self.assertIn('budget exhausted',result['note'])

    def test_enclosure_compatibility_is_not_exact_equality(self):
        from kernel_spec.num import from_interval
        ref = SimpleNamespace(shape=(1,), get=lambda _: D.R(from_interval(0, 2)))
        result = D.full_reference(ref, np.array([1]), 1)
        self.assertEqual(result["relations"], {"within_enclosure": 1})
        self.assertFalse(result["all_exact_equal"])
        self.assertEqual(D.reference_status({"out": {"node": "n", "eager": result}}), "complete")

    def test_provenance_is_selected_by_its_recorded_generated_graph(self):
        with tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
            root = Path(directory)
            compiled = root / "capture/model/config/unit/compiled"
            compiled.mkdir(parents=True)
            source = root / "kernel.py"
            source.write_text("def kernel():\n    return 1\n")
            (compiled / "kernel.source").write_text(f'#loc = loc("{source}":1:0)')
            origins = root / "origins/model"
            origins.mkdir(parents=True)
            records = [dict(kernel="kernel", kernel_path=str(source), output_code=str(root / (name+".py")),
                            params={"x": {"origin_node": name}}) for name in ("first", "second")]
            (origins / "config.jsonl").write_text("\n".join(map(json.dumps, records)))
            rec = dict(unit="model/config/unit", kernel="kernel", output_code=str(root / "second.py"))
            actual, _ = D.provenance(rec, root / "capture", root / "origins")
            self.assertEqual(actual, [records[1]])
            rec["output_code"] = str(root / "unrecorded.py")
            actual, reason = D.provenance(rec, root / "capture", root / "origins")
            self.assertIsNone(actual)
            self.assertIn("selected output code", reason)

    def test_provenance_cache_relocation_requires_identical_source(self):
        with tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
            root = Path(directory)
            capture = root / "capture/model/config/unit/compiled"
            capture.mkdir(parents=True)
            old, new = root / "old.py", root / "new.py"
            old.write_text("def kernel():\n    return 1\n")
            new.write_text(old.read_text())
            (capture / "kernel.source").write_text(f'#loc = loc("{old}":1:0)')
            origins = root / "origins/model"
            origins.mkdir(parents=True)
            records = [dict(kernel="kernel", kernel_path=str(new), params={"x": {"origin_node": name}})
                       for name in ("first", "second")]
            (origins / "config.jsonl").write_text("\n".join(map(json.dumps, records)))
            rec = dict(unit="model/config/unit", kernel="kernel")
            actual, reason = D.provenance(rec, root / "capture", root / "origins")
            self.assertEqual(actual, records)  # retain all call origins, never choose the first
            self.assertIn("identical kernel source", reason)
            new.write_text("def kernel():\n    return 2\n")
            self.assertIsNone(D.provenance(rec, root / "capture", root / "origins")[0])

    def test_unverified_captures_require_explicit_binding_only_selection(self):
        records = [dict(unit="fresh", kernel="k", verify="not_run", status="candidate",
                        candidate_status="not_matched", candidates=[], output_code=None)]
        for flags, succeeds in (((), False), (("--include-unverified",), False),
                                (("--include-unverified", "--binding-only"), True)):
            with tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
                target = Path(directory) / "result.jsonl"
                argv = ["dynamic_three_way", "--bindings", "fresh.json.gz", "--out", str(target), "--n", "1", *flags]
                with patch.object(sys, "argv", argv), patch.object(D.gzip, "open") as source:
                    source.return_value.read.return_value = json.dumps(records).encode()
                    if succeeds:
                        self.assertEqual(D.main(), 0)
                        row = json.loads(target.read_text())
                        self.assertEqual(row["verify"], "not_run")
                        self.assertTrue(row["include_unverified"])
                        self.assertEqual(row["reference_status"], "not_evaluated")
                        self.assertEqual(row["semantic_verdict"], "not_established")
                    else:
                        with self.assertRaises(SystemExit):
                            D.main()
                        self.assertFalse(target.exists())

    def run_report(self, directory, *args, result=None):
        target = Path(directory) / "report.jsonl"
        record = dict(unit="unit", kernel="kernel", verify="bit_identical", output_code="code", status="draft")
        with patch.object(sys, "argv", ["dynamic_three_way", "--bindings", "candidate_records.json.gz", "--out", str(target), *args]), \
                patch.object(D.gzip, "open") as source, patch.object(D, "run_unit", return_value=result):
            source.return_value.read.return_value = json.dumps([record]).encode()
            code = D.main()
        return code, target

    def test_empty_or_unmatched_selection_fails_without_report(self):
        for args in (("--n", "0"), ("--n", "-1"), ("--only",), ("--only", "unknown"),
                     ("--only", "unit", "unknown")):
            with self.subTest(args=args), tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
                with self.assertRaises(SystemExit) as raised:
                    self.run_report(directory, *args)
                self.assertNotEqual(raised.exception.code, 0)
                self.assertFalse((Path(directory) / "report.jsonl").exists())

    def test_refusal_explicitly_preserves_unknown_semantics(self):
        with tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
            code, target = self.run_report(directory, "--n", "1", result={"status": "missing origin"})
            self.assertEqual(code, 0)  # a recorded refusal is not an implementation crash
            row = json.loads(target.read_text())
            self.assertEqual(row["semantic_verdict"], "not_established")
            self.assertEqual(row["reference_status"], "not_evaluated")

    def test_missing_and_ambiguous_candidates_remain_in_selection(self):
        records = [dict(unit=status, kernel="kernel", verify="bit_identical", output_code=None,
                        status="candidate only", candidate_status=status, candidates=[])
                   for status in ("not_matched", "ambiguous")]
        with tempfile.TemporaryDirectory(dir=D.REPO / ".cache/tmp") as directory:
            target = Path(directory) / "report.jsonl"
            with patch.object(sys, "argv", ["dynamic_three_way", "--bindings", "new.json.gz", "--n", "2", "--out", str(target)]), \
                    patch.object(D.gzip, "open") as source, \
                    patch.object(D, "OutputCode", side_effect=AssertionError("unresolved origin was evaluated")):
                source.return_value.read.return_value = json.dumps(records).encode()
                self.assertEqual(D.main(), 0)
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual({row["unit"] for row in rows}, {"not_matched", "ambiguous"})
            self.assertTrue(all(row["binding_status"] == "not_established" for row in rows))
            self.assertTrue(all(row["reference_status"] == "not_evaluated" for row in rows))

    def test_legacy_or_inconsistent_origin_is_not_used(self):
        base = dict(unit="model/config/kernel", kernel="kernel", output_code="code.py")
        records = [base, dict(base, candidate_status="matched", candidates=[]),
                   dict(base, candidate_status="matched", candidates=[{"file": "different.py"}])]
        with patch.object(D, "OutputCode", side_effect=AssertionError("invalid origin was evaluated")):
            for record in records:
                result = D.run_unit(record, random.Random(0))
                self.assertEqual(result["binding_status"], "not_established")

    def test_large_integer_difference_survives(self):
        ref = SimpleNamespace(shape=(1,), get=lambda _: D.R(2**53 + 1))
        result = D.sample_reference(ref, np.array([2**53], dtype=np.int64), "i64", random.Random(0))
        self.assertGreater(result["max_rel"], 0)
        self.assertEqual(D.relerr(D.R(2**53 + 1), 2**53 + 1, "i64"), 0)

    def test_typed_failure_at_node_and_element(self):
        for cls, outcome in ((D.NotEstablished, "not_established"), (D.Undefined, "domain_undefined"),
                             (D.Unsupported, "unsupported"), (D.Invalid, "invalid_input")):
            def fail(_):
                raise cls("witness")
            with self.subTest(outcome=outcome):
                for ref in (cls("witness"), SimpleNamespace(shape=(1,), get=fail)):
                    result = D.sample_reference(ref, np.array([0.]), "f64", random.Random(0))
                    self.assertEqual(result["outcome"], outcome)
                    outputs = {"out": {"node": "x", "eager": {"outcome": "value"}, "compiled": result}}
                    self.assertEqual(D.reference_status(outputs), outcome)

    def test_unexpected_exception_is_not_a_mathematical_outcome(self):
        with self.assertRaisesRegex(RuntimeError, "broken reference"):
            D.sample_reference(RuntimeError("broken reference"), np.array([0.]), "f64", random.Random(0))

    def test_no_output_does_not_count_as_sampled(self):
        self.assertEqual(D.reference_status({}), "not_evaluated")


if __name__ == "__main__":
    unittest.main()
