"""Failure injections A-I of PLAN_v3 V1 and the schema checks of the 2026-10-15 review, against the single checker
(tools/check_contracts.py) and the single runtime entry (kernel_spec.registry). Every injection works on a copy in a
temporary directory; no tracked file is written."""
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import check_contracts  # noqa: E402
import run_checks  # noqa: E402
from kernel_spec import compare as C  # noqa: E402
from kernel_spec import expr, registry, schema  # noqa: E402
from kernel_spec.num import Result, to_iv, hull  # noqa: E402

sys.path.insert(0, str(ROOT))
SOURCES = json.loads((ROOT / "SOURCES.json").read_text())
CE = yaml.safe_load((ROOT / "contracts/CE.yaml").read_text())


PILOTS = ("CE", "ADAMW", "RANKMEAN")


class Sandbox:
    """A copy of the three pilot contracts, adapters/ and SOURCES.json that a test may corrupt (the other contracts are
    checked by tools/check_contracts.py itself; they are left out so that work in progress cannot affect these tests)."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="spec_checker_", dir=None))
        (self.dir / "contracts").mkdir()
        for p in registry.contract_paths():
            if p.stem in PILOTS:
                shutil.copy(p, self.dir / "contracts" / p.name)
        shutil.copytree(ROOT / "adapters", self.dir / "adapters", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "SOURCES.json", self.dir / "SOURCES.json")

    def edit_yaml(self, cid, fn):
        p = self.dir / "contracts" / f"{cid}.yaml"
        d = yaml.safe_load(p.read_text())
        fn(d)
        p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))

    def edit_text(self, rel, old, new):
        p = self.dir / rel
        s = p.read_text()
        assert old in s, f"{old!r} not in {rel}"
        p.write_text(s.replace(old, new, 1))

    def check(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = check_contracts.main(["--contracts", str(self.dir / "contracts"), "--adapters",
                                       str(self.dir / "adapters"), "--sources", str(self.dir / "SOURCES.json"), *extra])
        self.output = out.getvalue()
        return rc

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Injections(unittest.TestCase):
    def setUp(self):
        self.s = Sandbox()

    def tearDown(self):
        self.s.close()

    def test_control_passes(self):
        self.assertEqual(self.s.check(), 0, self.s.output)

    def test_duplicate_contract_fields_rejected_by_checker_and_runtime(self):
        p = self.s.dir / "contracts" / "CE.yaml"
        original = p.read_text()
        for extra in ("\nid: CE\n", "\nextra: {guarantee: exact, guarantee: rigorous_enclosure}\n"):
            with self.subTest(extra=extra):
                p.write_text(original + extra)
                self.assertEqual(self.s.check(), 1, self.s.output)
                with self.assertRaises(yaml.YAMLError):
                    registry.load("CE", self.s.dir / "contracts", self.s.dir / "adapters")

    def test_A_wrong_expected_value(self):
        self.s.edit_yaml("CE", lambda d: d["examples"][0]["expected"]["fields"].__setitem__("loss", "999"))
        self.assertEqual(self.s.check(), 1)
        self.assertIn("CE.E1: contradiction", self.s.output)

    def test_B_unknown_source_id(self):
        self.s.edit_yaml("CE", lambda d: d["clauses"][0].__setitem__("basis", ["PCE.DOES_NOT_EXIST"]))
        self.assertEqual(self.s.check(), 1)
        self.assertIn("unresolved basis", self.s.output)

    def test_D_inverted_interval(self):
        self.s.edit_text("adapters/RANKMEAN.py", 'return Result("value", {"loss": v24.value(tp.cross_rank_mean(S, n))})',
                         'return Result("value", {"loss": v24.value(type("X", (), {"lo": Fraction(2), "hi": Fraction(1)})())})')
        self.assertEqual(self.s.check(), 1)
        self.assertIn("inverted", self.s.output)

    def test_E_missing_output_field(self):
        self.s.edit_text("adapters/CE.py", 'return Result("value", {"loss": v24.tree(z["loss"]), "gradient": grad})',
                         'return Result("value", {"loss": v24.tree(z["loss"])})')
        self.assertEqual(self.s.check(), 1)

    def test_E_missing_state_field(self):
        self.s.edit_text("adapters/ADAMW.py", '"v": v24.tree(state.v), "t": state.t})', '"v": v24.tree(state.v)})')
        self.assertEqual(self.s.check(), 1)

    def test_E_missing_output_row(self):
        self.s.edit_text("adapters/CE.py", 'return Result("value", {"loss": v24.tree(z["loss"]), "gradient": grad})',
                         'return Result("value", {"loss": v24.tree(z["loss"]), "gradient": grad[:-1] if len(grad) > 1 '
                         'else grad})')
        self.assertEqual(self.s.check(), 1)

    def _with_reading(self, expected_loss):
        def add(d):
            d["schema_version"] = "1.2"
            d["readings"] = {"R_B": {"meaning": "diagnostic reading R_B", "basis": ["PCE.SMOOTH"]}}
            e = next(e for e in d["examples"] if e["id"] == "CE.E5")
            fields = dict(e["expected"]["fields"], loss=expected_loss)
            # R_B: every class term weighted by the target-class weight w_y = 1, so a = [3/4, 1/4] and grad = p - a
            fields["gradient"] = [["1/(1+exp(1))-3/4", "exp(1)/(1+exp(1))-1/4"]]
            e["readings"] = {"R_B": {"outcome": "value", "fields": fields}}
        self.s.edit_yaml("CE", add)

    def test_F_reading_control(self):
        self._with_reading("log(1+exp(1))-1/4")
        self.assertEqual(self.s.check(), 0, self.s.output)

    def test_F_wrong_reading_expected(self):
        self._with_reading("777")
        self.assertEqual(self.s.check(), 1)
        self.assertIn("CE.E5[R_B]: contradiction", self.s.output)

    def test_H_mutant_killed_only_by_exception(self):
        self.s.edit_text("adapters/CE.py", '"CE.M1": _m("mean_by_N"),', '"CE.M1": lambda p, i: 1 / 0,')
        self.assertEqual(self.s.check(), 1)
        self.assertIn("CE.M1: mutation not rejected", self.s.output)

    def test_H_unavailable_mutant_is_not_a_counterexample(self):
        for outcome in ("not_established", "unsupported"):
            with self.subTest(outcome=outcome):
                adapter = self.s.dir / "adapters/CE.py"
                original = adapter.read_text()
                try:
                    adapter.write_text(original.replace('"CE.M1": _m("mean_by_N"),',
                        f'"CE.M1": lambda p, i: Result("{outcome}", note="no answer"),'))
                    self.assertEqual(self.s.check(), 1, self.s.output)
                    self.assertIn("CE.M1: mutation not rejected", self.s.output)
                finally:
                    adapter.write_text(original)

    def test_H_unavailable_mutant_on_invalid_input_is_not_a_counterexample(self):
        for outcome in ("not_established", "unsupported", "implementation_error"):
            rejected, _ = check_contracts.killed(Result(outcome), {"outcome": "invalid_input"}, {})
            self.assertFalse(rejected, outcome)

    def test_invalid_reference_guarantee_is_not_a_mutation_witness(self):
        outputs = {"x": {"guarantee": "exact"}}
        expected = {"outcome": "value", "fields": {"x": 0}}
        for value in (hull(0, 1), 1.0):
            result = Result("value", {"x": value})
            self.assertEqual(C.compare_result(result, expected, outputs)[0], C.STRUCTURE)
            self.assertFalse(check_contracts.killed(result, expected, outputs)[0])
        self.assertTrue(check_contracts.killed(Result("value", {"x": Fraction(1)}), expected, outputs)[0])
        self.assertTrue(check_contracts.killed(Result("value", {"x": to_iv(1)}), expected, outputs)[0])

    def test_missing_mutant_implementation(self):
        self.s.edit_text("adapters/RANKMEAN.py", '"RM.M5": m5}', '}')
        self.assertEqual(self.s.check(), 1)

    def test_unreadable_yaml_fails_without_crash(self):
        (self.s.dir / "contracts" / "BROKEN.yaml").write_text("id: BROKEN\ndomain: [a [b]\n")
        self.assertEqual(self.s.check(), 1)
        self.assertIn("unreadable YAML", self.s.output)

    def test_empty_contract_directory(self):
        for p in (self.s.dir / "contracts").glob("*.yaml"):
            p.unlink()
        self.assertEqual(self.s.check(), 1)

    def test_requested_contracts_must_all_exist_before_writing_results(self):
        for selected in (("CE", "DOES_NOT_EXIST"), ("DOES_NOT_EXIST",)):
            with self.subTest(selected=selected):
                report = self.s.dir / "report.json"
                self.assertEqual(self.s.check("--only", *selected, "--record-checks", "--json", str(report)), 1)
                self.assertIn("DOES_NOT_EXIST", self.s.output)
                self.assertFalse(report.exists())
                self.assertFalse((self.s.dir / "contracts" / "DEVELOPMENT_CHECKS.json").exists())

    def test_explicit_empty_selection_is_not_all_contracts(self):
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            self.s.check("--only")
        self.assertNotEqual(raised.exception.code, 0)

    def test_development_record_contains_counts_only_for_passing_contracts(self):
        self.s.edit_yaml("CE", lambda d: d["examples"][0]["expected"]["fields"].__setitem__("loss", "999"))
        self.s.check("--record-checks")
        records = json.loads((self.s.dir / "contracts/DEVELOPMENT_CHECKS.json").read_text())
        self.assertNotIn("CE", records)
        self.assertIn("RANKMEAN", records)
        self.assertEqual(set(records["RANKMEAN"]), {"checked", "examples", "readings", "mutations", "verdict"})


class RunnerC(unittest.TestCase):
    def test_zero_tests_are_not_a_success(self):
        for text in ("Ran 0 tests in 0.0s\n\nOK", "0/0 checks passed"):
            self.assertFalse(run_checks.counted()(0, text, None)[0])

    def test_C_child_exit_9_fails_the_run(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            ok = run_checks.Check("x", "passes", ["-c", "pass"], Path(t))
            bad = run_checks.Check("x", "exit 9", ["-c", "import sys; sys.exit(9)"], Path(t))
            _, failed_ok = run_checks.execute([ok], Path(t))
            _, failed = run_checks.execute([ok, bad, ok], Path(t))
        self.assertFalse(failed_ok)
        self.assertTrue(failed)

    def test_C_counted_tests_need_exit_zero(self):
        acc = run_checks.counted(3)
        self.assertFalse(acc(9, "Ran 3 tests in 0.1s\n\nOK", None)[0])
        self.assertFalse(acc(0, "Ran 2 tests in 0.1s\n\nOK", None)[0])
        self.assertTrue(acc(0, "Ran 3 tests in 0.1s\n\nOK", None)[0])

    def test_C_custom_acceptance_cannot_override_child_failure(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            check = run_checks.Check("x", "false success", ["-c", "import sys; sys.exit(9)"], Path(t),
                                     accept=lambda rc, out, tmp: (True, "report looks consistent"))
            rows, failed = run_checks.execute([check], Path(t))
        self.assertTrue(failed)
        self.assertEqual(rows[0][2], "失败")
        self.assertIn("exit 9", rows[0][4])


class ActivationG(unittest.TestCase):
    def test_G_unreviewed_implementations_do_not_activate(self):
        c = registry.load("CE")
        with tempfile.TemporaryDirectory() as t, patch.object(registry, "VALIDATED", Path(t)):
            ok, reasons, _ = registry.activation(c, "index_unsmoothed")
        self.assertFalse(ok)
        self.assertTrue(any("implementation review file" in r for r in reasons))

    def test_G_retained_reviewed_implementations_activate(self):
        for cid in ("CE", "SOFTMAX", "NORM", "ACT", "ADAMW", "RANKMEAN", "MATMUL"):
            c = registry.load(cid)
            for profile in c.data["profiles"]:
                task_bindings = {profile: "use the selected task profile"}
                ok, reasons, _ = registry.activation(c, profile, task_bindings)
                self.assertTrue(ok, reasons)

    def test_G_current_files_must_equal_reviewed_files(self):
        c = registry.load("RANKMEAN")
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "contracts").mkdir()
            (root / "adapters").mkdir()
            (root / "contracts/RANKMEAN.yaml").write_bytes(c.path.read_bytes())
            (root / "adapters/RANKMEAN.py").write_text("reviewed adapter that is no longer current\n")
            with patch.object(registry, "VALIDATED", root):
                ok, reasons, _ = registry.activation(c, "base", {"base": "selected"})
        self.assertFalse(ok)
        self.assertTrue(any("current adapter differs" in r for r in reasons))

    def test_G_task_choice_needs_binding(self):
        c = registry.load("RANKMEAN")
        with patch.object(registry, "_validated_file_reasons", return_value=[]):
            _, reasons, _ = registry.activation(c, "base", None)
            self.assertTrue(any("task binding" in r for r in reasons))
            _, reasons, _ = registry.activation(c, "base", {"base": "selected"})
            self.assertFalse(any("task binding" in r for r in reasons))

    def test_G_drafted_needs_explicit_permission(self):
        c = registry.load("CE")
        c.data = copy.deepcopy(c.data)
        c.data["review"]["status"] = "drafted"
        with patch.object(registry, "_validated_file_reasons", return_value=[]):
            _, reasons, labels = registry.activation(c, "index_unsmoothed")
            self.assertTrue(any("drafted" in r for r in reasons))
            _, reasons, labels = registry.activation(c, "index_unsmoothed", allow_draft=True)
            self.assertFalse(any("drafted" in r for r in reasons))
            self.assertTrue(labels["draft_contract"])

    def test_G_draft_folder_is_not_loadable(self):
        with self.assertRaises(FileNotFoundError):
            registry.load("draft/CE")

    def test_G_query_refuses_unbound_task_profile(self):
        e = next(e for e in CE["examples"] if e["profile"] == "index_smoothing_R_A")
        r = registry.query("CE", "index_smoothing_R_A", e["input"])
        self.assertEqual(r.outcome, "not_established")
        self.assertIn("task binding", r.note)
        r = registry.query("CE", "index_smoothing_R_A", e["input"], task_bindings={"index_smoothing_R_A": "test"})
        self.assertNotIn("task binding", r.note)
        # Task choice does not supply implementation-review evidence.
        if r.outcome == "not_established":
            self.assertIn("implementation review file", r.note)
        else:
            v, _ = C.compare_result(r, e["expected"], CE["outputs"])
            self.assertEqual(v, C.CONSISTENT)
        self.assertEqual(r.labels["contract"], "CE")


class ToleranceI(unittest.TestCase):
    def test_I_exact_reference_no_tolerance(self):
        v, _ = C.compare_result(Result("value", {"x": Fraction(1)}), {"outcome": "value", "fields": {"x": "1+1/10**45"}},
                                {"x": {"guarantee": "exact"}})
        self.assertEqual(v, C.CONTRADICTION)

    def test_I_enclosure_must_contain(self):
        ref = expr.parse("log(2)*(1+1/10**45)")
        v, _ = C.compare_result(Result("value", {"x": ref}), {"outcome": "value", "fields": {"x": "log(2)"}},
                                {"x": {"guarantee": "rigorous_enclosure"}})
        self.assertEqual(v, C.CONTRADICTION)

    def test_I_containment_overlap_disjoint(self):
        from mpmath import iv
        ref = iv.mpf([to_iv(Fraction(1, 2)).a, to_iv(Fraction(3, 4)).b])      # exactly [1/2, 3/4]
        out = {"x": {"guarantee": "rigorous_enclosure"}}
        inside = "1/2+(1/4)*(1-1/(1+exp(-1000)))"
        straddles = "3/4+(exp(1)-exp(1))"                                      # an enclosure around 3/4 itself
        v, _ = C.compare_result(Result("value", {"x": ref}), {"outcome": "value", "fields": {"x": inside}}, out)
        self.assertEqual(v, C.CONSISTENT)
        v, _ = C.compare_result(Result("value", {"x": ref}), {"outcome": "value", "fields": {"x": straddles}}, out)
        self.assertEqual(v, C.NOT_ESTABLISHED)
        v, _ = C.compare_result(Result("value", {"x": ref}), {"outcome": "value", "fields": {"x": "3/4+1/10**300"}},
                                {"x": {"guarantee": "rigorous_enclosure"}})
        self.assertEqual(v, C.CONTRADICTION)

    def test_I_exact_output_must_not_be_an_enclosure(self):
        v, _ = C.compare_result(Result("value", {"x": to_iv(Fraction(1, 3))}), {"outcome": "value", "fields": {"x": "1/3"}},
                                {"x": {"guarantee": "exact"}})
        self.assertEqual(v, C.STRUCTURE)


class Schema(unittest.TestCase):
    """The checks of the 2026-10-15 reviewer's validator, ported to kernel_spec.schema."""

    def test_original(self):
        for p in registry.contract_paths():
            if p.stem in PILOTS:
                self.assertEqual(schema.validate(yaml.safe_load(p.read_text()), SOURCES), [], p.name)

    def test_missing_output_rejected(self):
        c = copy.deepcopy(CE)
        del c["examples"][0]["expected"]["fields"]["gradient"]
        self.assertTrue(schema.validate(c, SOURCES))

    def test_unresolved_basis_rejected(self):
        c = copy.deepcopy(CE)
        c["clauses"][0]["basis"] = ["missing.quote"]
        self.assertTrue(schema.validate(c, SOURCES))

    def test_unknown_profile_rejected(self):
        c = copy.deepcopy(CE)
        c["examples"][0]["profile"] = "missing"
        self.assertTrue(schema.validate(c, SOURCES))

    def test_duplicate_example_rejected(self):
        c = copy.deepcopy(CE)
        c["examples"].append(c["examples"][0])
        self.assertTrue(schema.validate(c, SOURCES))

    def test_executable_expected_rejected(self):
        c = copy.deepcopy(CE)
        c["examples"][0]["expected"]["fields"]["loss"] = '__import__("os").getcwd()'
        self.assertTrue(schema.validate(c, SOURCES))

    def test_boolean_not_number(self):
        c = copy.deepcopy(CE)
        c["examples"][0]["expected"]["fields"]["loss"] = True
        self.assertTrue(schema.validate(c, SOURCES))

    def test_invalid_review_status_rejected(self):
        c = copy.deepcopy(CE)
        c["review"]["status"] = "draft"
        self.assertTrue(schema.validate(c, SOURCES))

    def test_drafted_needs_author(self):
        c = copy.deepcopy(CE)
        c["review"] = {"status": "drafted"}
        self.assertTrue(schema.validate(c, SOURCES))

    def test_clause_without_example_rejected(self):
        c = copy.deepcopy(CE)
        c["clauses"].append(dict(c["clauses"][0], id="CE.C99"))
        self.assertTrue(schema.validate(c, SOURCES))


if __name__ == "__main__":
    unittest.main()
