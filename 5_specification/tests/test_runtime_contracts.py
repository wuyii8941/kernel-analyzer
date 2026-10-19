"""A passed development check cannot replace validation of the current query."""
import copy
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry, schema
from kernel_spec.compare import validate_reference_values
from kernel_spec.num import Result, hull


class RuntimeContract(unittest.TestCase):
    def setUp(self):
        # These tests exercise individual admission rules. The exact-file rule
        # has its own tests in test_checker.py.
        self.reviewed_files = patch.object(registry, "_validated_file_reasons", return_value=[])
        self.reviewed_files.start()
        self.addCleanup(self.reviewed_files.stop)

    def test_pending_implementation_is_not_activated_by_matching_checks(self):
        c = registry.load("RANKMEAN")
        c.data = copy.deepcopy(c.data)
        for status in ("pending", None, "unknown"):
            c.data["activation"]["implementation_status"] = status
            for allow_draft in (False, True):
                ok, reasons, labels = registry.activation(c, "base", {"base": "selected"}, allow_draft=allow_draft)
                self.assertFalse(ok)
                self.assertTrue(any("implementation status" in reason for reason in reasons))
                self.assertEqual(labels["implementation_status"], status)
        c.data["activation"]["implementation_status"] = "validated"
        ok, reasons, labels = registry.activation(c, "base", {"base": "selected"})
        self.assertTrue(ok, reasons)
        self.assertEqual(labels["implementation_status"], "validated")
        # A mathematical reference query is not a kernel-binding certificate.
        self.assertEqual(labels["runtime_binding_status"], "pending")

    def test_reviewed_contract_does_not_upgrade_draft_decisions(self):
        c = registry.load("RANKMEAN")
        c.data["decisions"][1]["status"] = "drafted"
        for allow_draft in (False, True):
            ok, reasons, _ = registry.activation(c, "base", {"base": "equal valid tokens"},
                                                  allow_draft=allow_draft)
            self.assertFalse(ok)
            self.assertTrue(any("RM.ZERO" in reason and "drafted" in reason for reason in reasons), reasons)
        self.assertTrue(any("RM.ZERO" in error and "drafted" in error
                            for error in schema.validate(c.data, registry.load_sources())))

    def test_draft_contract_keeps_explicit_development_access(self):
        c = registry.load("RANKMEAN")
        c.data["review"]["status"] = "drafted"
        c.data["decisions"][1]["status"] = "drafted"
        ok, reasons, labels = registry.activation(c, "base", {"base": "equal valid tokens"},
                                                  allow_draft=True)
        self.assertTrue(ok, reasons)
        self.assertTrue(labels["draft_contract"])
        self.assertEqual(labels["semantic_verdict"], "not_established")

    def test_invalid_decision_review_status_cannot_be_activated(self):
        for status in (None, True, "approved", ""):
            c = registry.load("RANKMEAN")
            c.data["decisions"][1]["status"] = status
            ok, reasons, _ = registry.activation(c, "base", {"base": "equal valid tokens"})
            self.assertFalse(ok)
            self.assertTrue(any("RM.ZERO" in reason and "status" in reason for reason in reasons), reasons)

    def query_result(self, fields):
        c = registry.load("ADAMW")
        c.adapter = SimpleNamespace(reference=lambda profile, inp: Result("value", fields))
        with patch.object(registry, "load", return_value=c), \
             patch.object(registry, "activation", return_value=(True, [], {})):
            return registry.query("ADAMW", "base", {})

    def test_query_rejects_missing_post_state(self):
        r = self.query_result({"theta": [Fraction(1)]})
        self.assertEqual(r.outcome, "implementation_error")
        self.assertIn("m", r.note)

    def test_query_rejects_extra_output(self):
        r = self.query_result(dict(theta=[Fraction(1)], m=[Fraction(0)], v=[Fraction(0)], t=1, extra=1))
        self.assertEqual(r.outcome, "implementation_error")

    def test_query_preserves_all_outputs(self):
        fields = dict(theta=[Fraction(1)], m=[Fraction(0)], v=[Fraction(0)], t=1)
        r = self.query_result(fields)
        self.assertEqual(r.outcome, "value")
        self.assertEqual(r.fields, fields)

    def test_query_rejects_enclosure_for_exact_state(self):
        fields = dict(theta=[Fraction(1)], m=[hull(0, 1)], v=[Fraction(0)], t=1)
        self.assertEqual(self.query_result(fields).outcome, "implementation_error")

    def test_query_rejects_float_as_strict_reference(self):
        fields = dict(theta=[0.1], m=[Fraction(0)], v=[Fraction(0)], t=1)
        self.assertEqual(self.query_result(fields).outcome, "implementation_error")

    def test_query_accepts_enclosed_theta_and_exact_state(self):
        fields = dict(theta=[hull(0, 1)], m=[Fraction(0)], v=[Fraction(0)], t=1)
        self.assertEqual(self.query_result(fields).outcome, "value")

    def test_structured_sets_and_probabilities_keep_their_declared_forms(self):
        validate_reference_values({"dropped": frozenset()}, {"dropped": {"guarantee": "exact"}})
        validate_reference_values({"out": [hull(0, 1), frozenset({1, 2})]},
                                  {"out": {"guarantee": "allowed_set"}})

    def test_boolean_cannot_be_a_numeric_reference_output(self):
        fields = dict(theta=[Fraction(1)], m=[Fraction(0)], v=[Fraction(0)], t=True)
        self.assertEqual(self.query_result(fields).outcome, "implementation_error")
        for value in (True, [False], {"state": [True]}, frozenset({True})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_reference_values({"out": value}, {"out": {"guarantee": "exact"}})

    def test_boolean_output_requires_boolean_leaves(self):
        for value in (1, Fraction(0), hull(0, 1), [[True, 0]], {"mask": [True, 1]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_reference_values({"out": value}, {"out": {"kind": "boolean", "guarantee": "rigorous_enclosure"}})
        validate_reference_values({"out": [[True, False]]}, {"out": {"kind": "boolean", "guarantee": "exact"}})

    def test_discrete_output_can_keep_mixed_numeric_and_boolean_values(self):
        validate_reference_values({"out": [[True, 2], [Fraction(1, 3), False]]},
                                  {"out": {"kind": "discrete", "guarantee": "exact"}})

    def test_unknown_output_kind_is_not_silently_numeric(self):
        c = registry.load("RANKMEAN")
        c.data["outputs"]["loss"]["kind"] = "boolen"
        self.assertTrue(any("kind" in e for e in schema.validate(c.data, registry.load_sources())))
        with self.assertRaises(ValueError):
            validate_reference_values({"loss": Fraction(1)}, c.data["outputs"])

    def test_nan_cannot_be_a_strict_reference(self):
        fields = dict(theta=[float("nan")], m=[Fraction(0)], v=[Fraction(0)], t=1)
        self.assertEqual(self.query_result(fields).outcome, "implementation_error")

    def test_missing_source_cannot_be_ignored(self):
        c = registry.load("CE")
        sources = copy.deepcopy(registry.load_sources())
        del sources["PCE"]
        with patch.object(registry, "load_sources", return_value=sources):
            _, reasons, _ = registry.activation(c, "index_unsmoothed")
        self.assertTrue(any("unresolved basis" in r for r in reasons), reasons)

    def test_empty_task_choice_is_not_a_choice(self):
        c = registry.load("RANKMEAN")
        for choice in (None, "", {}, False):
            _, reasons, _ = registry.activation(c, "base", {"base": choice})
            self.assertTrue(any("task binding" in r for r in reasons), (choice, reasons))

    def test_task_choice_requires_a_statement_not_python_truthiness(self):
        c = registry.load("RANKMEAN")
        for bindings in ({"base": True}, {"base": 1}, {"base": ["selected"]},
                         {"base": {"selected": True}}, {"base": " \n\t"},
                         "base", ["base"], {"unknown_profile": "selected"}):
            with self.subTest(bindings=bindings):
                ok, reasons, _ = registry.activation(c, "base", bindings)
                self.assertFalse(ok)
                self.assertTrue(any("task binding" in r for r in reasons), reasons)
        statement = "Task owner selects equal-valid-token normalization."
        ok, reasons, labels = registry.activation(c, "base", {"base": statement})
        self.assertTrue(ok, reasons)
        self.assertEqual(labels["task_choice"], statement)
        self.assertEqual(labels["semantic_verdict"], "not_established")

    def test_allow_draft_is_an_explicit_boolean(self):
        c = registry.load("RANKMEAN")
        c.data["review"]["status"] = "drafted"
        for permission in ("false", "true", 1, 0, [], {}, None):
            with self.subTest(permission=permission):
                ok, reasons, _ = registry.activation(c, "base", {"base": "selected"},
                                                     allow_draft=permission)
                self.assertFalse(ok)
                self.assertTrue(any("allow_draft" in r for r in reasons), reasons)
        ok, reasons, labels = registry.activation(c, "base", {"base": "selected"},
                                                 allow_draft=True)
        self.assertTrue(ok, reasons)
        self.assertTrue(labels["draft_contract"])
        self.assertEqual(labels["semantic_verdict"], "not_established")

    def test_invalid_profile_name_refuses_without_calling_reference(self):
        c = registry.load("RANKMEAN")
        for profile in (None, [], {}, True, "", " \n"):
            with self.subTest(profile=profile):
                ok, reasons, labels = registry.activation(c, profile)
                self.assertFalse(ok)
                self.assertTrue(any("profile must be" in r for r in reasons), reasons)
                self.assertEqual(labels["semantic_verdict"], "not_established")
        with patch.object(registry, "load", return_value=c), \
             patch.object(c.adapter, "reference") as reference:
            result = registry.query("RANKMEAN", [], {})
            self.assertEqual(result.outcome, "not_established")
            reference.assert_not_called()

    def test_task_choice_requirement_must_be_explicit_boolean(self):
        for value in (None, 0, 1, "false", "true", [], {}):
            c = registry.load("RANKMEAN")
            c.data["profiles"]["base"]["task_choice_required"] = value
            with self.subTest(value=value):
                self.assertTrue(any("task_choice_required" in e for e in schema.validate(c.data, registry.load_sources())))
                ok, reasons, _ = registry.activation(c, "base", {"base": "selected"})
                self.assertFalse(ok)
                self.assertTrue(any("task_choice_required" in e for e in reasons), reasons)
        c = registry.load("RANKMEAN")
        del c.data["profiles"]["base"]["task_choice_required"]
        _, reasons, _ = registry.activation(c, "base")
        self.assertTrue(any("task_choice_required" in e for e in reasons), reasons)

    def test_source_metadata_without_the_source_is_not_evidence(self):
        sources = copy.deepcopy(registry.load_sources())
        sources["PCE"]["snapshot"] = "sources/does-not-exist.html"
        with patch.object(registry, "load_sources", return_value=sources):
            _, reasons, _ = registry.activation(registry.load("CE"), "index_unsmoothed")
        self.assertTrue(any("sources without an archived snapshot" in r for r in reasons), reasons)

    def test_circular_decisions_do_not_establish_a_source(self):
        for bases in ((["TEST.A"],), (["TEST.B"], ["TEST.A"])):
            c = registry.load("CE")
            c.data = copy.deepcopy(c.data)
            decisions = [{"id": f"TEST.{chr(65+i)}", "basis": b, "status": "reviewed", "statement": "test"}
                         for i, b in enumerate(bases)]
            c.data["decisions"] += decisions
            c.data["clauses"][0]["basis"] = ["TEST.A"]
            with self.subTest(bases=bases):
                errors = schema.validate(c.data, registry.load_sources())
                self.assertTrue(any("circular basis" in e for e in errors), errors)
                _, reasons, _ = registry.activation(c, "index_unsmoothed")
                self.assertTrue(any("circular basis" in e for e in reasons), reasons)

    def test_decision_without_basis_cannot_be_used_at_runtime(self):
        c = registry.load("CE")
        c.data["decisions"].append({"id": "TEST.EMPTY", "basis": [], "status": "reviewed"})
        c.data["clauses"][0]["basis"] = ["TEST.EMPTY"]
        _, reasons, _ = registry.activation(c, "index_unsmoothed")
        self.assertTrue(any("no source basis" in e for e in reasons), reasons)

    def test_profile_basis_must_resolve(self):
        c = registry.load("CE")
        c.data["profiles"]["index_unsmoothed"]["basis"] = ["MISSING_PROFILE_SOURCE.DEF"]
        self.assertTrue(any("unresolved basis" in e for e in schema.validate(c.data, registry.load_sources())))
        _, reasons, _ = registry.activation(c, "index_unsmoothed")
        self.assertTrue(any("unresolved basis" in e for e in reasons), reasons)

    def test_profile_only_source_must_be_retained(self):
        c = registry.load("CE")
        c.data["profiles"]["index_unsmoothed"]["basis"] = ["PROFILE_ONLY.DEF"]
        sources = registry.load_sources()
        sources["PROFILE_ONLY"] = {"items": {"DEF": "profile definition"},
                                   "snapshot": "sources/nonexistent-profile.txt"}
        with patch.object(registry, "load_sources", return_value=sources):
            _, reasons, _ = registry.activation(c, "index_unsmoothed")
        self.assertTrue(any("PROFILE_ONLY" in e and "archived snapshot" in e for e in reasons), reasons)

    def test_malformed_profile_basis_is_refused_without_crashing(self):
        for refs in ("PCE.DEF", None, [{"source": "PCE"}], [7]):
            c = registry.load("CE")
            c.data["profiles"]["index_unsmoothed"]["basis"] = refs
            with self.subTest(refs=refs):
                self.assertTrue(schema.validate(c.data, registry.load_sources()))
                _, reasons, _ = registry.activation(c, "index_unsmoothed")
                self.assertTrue(any("invalid basis declaration" in e for e in reasons), reasons)


if __name__ == "__main__":
    unittest.main()
