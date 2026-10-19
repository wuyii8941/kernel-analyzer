import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / "corpus")]
import call_binding as B


def tensor(dtype, values, shape):
    return {"t": dtype, "v": values, "shape": shape}


class CallBinding(unittest.TestCase):
    def exp_fixture(self):
        subject = {"library": "example", "version": "1", "callable": "example.exp",
                   "boundary": "one public forward call"}
        x = tensor("f32", [0.0, 1.0], [2])
        out = tensor("f32", [1.0, 2.7182817459106445], [2])
        plan = {
            "schema": "triton-call-binding-v1", "id": "example.exp.v1", "status": "drafted",
            "subject": subject, "launch_scope": "all_triton_launches_within_call",
            "tasks": [{
                "id": "forward", "contract": "MATHFN", "profile": "real",
                "task_statement": "The public call is aten.exp on the received tensor.",
                "input_map": [
                    {"target": "/op", "literal": "aten.exp", "basis": "public API operation"},
                    {"target": "/args/0", "source": "/tensor_arguments/x/before",
                     "conversion": "tensor_record"},
                ],
                "output_map": [
                    {"target": "/out", "source": "/returns/out", "conversion": "tensor_values"},
                ],
            }],
            "state_map": [{"argument": "x", "effect": "unchanged"}],
        }
        observation = {
            "subject": subject, "launch_scope_complete": True, "launches": ["exp_kernel"],
            "tensor_arguments": {"x": {"before": x, "after": copy.deepcopy(x)}},
            "returns": {"out": out}, "public_outputs": ["/returns/out"],
        }
        return plan, observation

    def test_draft_plan_checks_mapping_without_certifying_bias(self):
        report = B.inspect(*self.exp_fixture())
        self.assertEqual(report["binding_status"], "development_checked")
        self.assertEqual(report["program_reference_mode"], "numerical_difference")
        self.assertEqual(report["reference_status"], "value")
        self.assertFalse(report["reference_usable_for_bias_measurement"])
        self.assertEqual(report["bias_verdict"], "not_established")
        self.assertEqual(report["semantic_verdict"], "not_established")
        task = report["tasks"][0]
        self.assertEqual(task["fixed_sample_difference"]["out"]["coordinates"], 2)
        # An admitted specification still cannot promote a drafted call plan
        # into a bias verdict.
        self.assertTrue(task["specification_admitted"])

    def test_program_reference_selection_is_explicit_and_strict(self):
        plan, observation = self.exp_fixture()
        plan["program_reference"] = {
            "mode": "rounding_check",
            "basis": "The declared task is the only explicitly rounded conversion in the call.",
        }
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "development_checked", report)
        self.assertEqual(report["program_reference_mode"], "rounding_check")

        invalid = (
            {"mode": "unknown", "basis": "stated"},
            {"mode": "rounding_check", "basis": ""},
            {"mode": "rounding_check"},
            {"mode": "rounding_check", "basis": "stated", "extra": True},
            "rounding_check",
        )
        for selection in invalid:
            with self.subTest(selection=selection):
                broken = copy.deepcopy(plan)
                broken["program_reference"] = selection
                rejected = B.inspect(broken, observation)
                self.assertEqual(rejected["binding_status"], "rejected")
                self.assertNotIn("program_reference_mode", rejected)

    def test_exact_real_rounding_is_not_called_a_semantic_contradiction(self):
        report = B.inspect(*self.exp_fixture())
        task = report["tasks"][0]
        self.assertIn(task["fixed_call_against_f"],
                      ("inside_declared_result", "outside_declared_result", "not_established"))
        self.assertNotEqual(task["fixed_call_against_f"], "contradiction")
        self.assertIn("not a population bias or semantic verdict", task["fixed_call_meaning"])

    def test_tensor_record_output_keeps_dtype_and_shape_while_matching_reference_values(self):
        subject = {"library": "example", "version": "1", "callable": "example.softmax",
                   "boundary": "one public forward call"}
        x = tensor("f32", [[0.0, 0.0]], [1, 2])
        out = tensor("f32", [[0.5, 0.5]], [1, 2])
        plan = {
            "schema": "triton-call-binding-v1", "id": "example.softmax.v1", "status": "drafted",
            "subject": subject, "launch_scope": "all_triton_launches_within_call",
            "tasks": [{
                "id": "softmax", "contract": "SOFTMAX", "profile": "real",
                "task_statement": "Rank-2 last-dimension softmax.",
                "input_map": [
                    {"target": "/op", "literal": "aten._softmax", "basis": "public operation"},
                    {"target": "/args/0", "source": "/tensor_arguments/x/before",
                     "conversion": "tensor_record"},
                    {"target": "/args/1", "source": "/scalars/dim"},
                    {"target": "/args/2", "source": "/scalars/half_to_float"},
                ],
                "output_map": [
                    {"target": "/out", "source": "/returns/out", "conversion": "tensor_record"},
                ],
            }],
            "state_map": [{"argument": "x", "effect": "unchanged"}],
        }
        observation = {
            "subject": subject, "launch_scope_complete": True, "launches": ["softmax_kernel"],
            "tensor_arguments": {"x": {"before": x, "after": copy.deepcopy(x)}},
            "scalars": {"dim": -1, "half_to_float": False}, "returns": {"out": out},
            "public_outputs": ["/returns/out"],
        }
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "development_checked", report)
        self.assertEqual(report["reference_status"], "value")
        self.assertEqual(report["tasks"][0]["fixed_sample_difference"]["out"]["coordinates"], 2)

    def test_every_public_output_and_tensor_argument_must_be_accounted_for(self):
        plan, observation = self.exp_fixture()
        observation["public_outputs"].append("/gradients/x")
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "rejected")
        plan, observation = self.exp_fixture()
        observation["tensor_arguments"]["other"] = copy.deepcopy(observation["tensor_arguments"]["x"])
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "rejected")

    def test_every_received_input_must_be_mapped_or_explicitly_unused(self):
        plan, observation = self.exp_fixture()
        observation["scalars"] = {"scale": 2}
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("not completely accounted", report["reason"])
        plan["unused_inputs"] = [{"source": "/scalars/scale",
                                  "basis": "This profile fixes the public scale independently."}]
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "development_checked")
        plan["unused_inputs"][0].pop("basis")
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "rejected")

    def test_partial_input_paths_do_not_account_for_a_complete_received_input(self):
        with self.assertRaisesRegex(B.BindingError, "maps only part"):
            B._received_input_root("/tensor_arguments/x/before/v/0")
        with self.assertRaisesRegex(B.BindingError, "maps only part"):
            B._received_input_root("/scalars/options/dim")

        plan, observation = self.exp_fixture()
        plan["tasks"][0]["input_map"][1]["source"] = "/tensor_arguments/x/before/v/0"
        plan["tasks"][0]["input_map"][1]["conversion"] = "identity"
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")

        plan, observation = self.exp_fixture()
        observation["scalars"] = {"options": {"dim": -1, "stable": False}}
        plan["tasks"][0]["input_map"].append(
            {"target": "/kwargs", "source": "/scalars/options/dim"})
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")

    def test_tensor_records_preserve_shapes_after_an_empty_dimension(self):
        self.assertEqual(B._tensor_record(tensor("f32", [[], [], []], [3, 0]))["shape"], [3, 0])
        self.assertEqual(B._tensor_record(tensor("f32", [], [0, 2]))["shape"], [0, 2])
        self.assertEqual(B._tensor_record(tensor("f32", [], [0, 2, 4]))["shape"], [0, 2, 4])

    def test_tensor_records_reject_values_inconsistent_with_explicit_shape(self):
        malformed = (
            tensor("f32", [[], []], [3, 0]),
            tensor("f32", [[1.0], [2.0]], [2, 2]),
            tensor("f32", [[1.0], []], [2, 1]),
            tensor("f32", [], []),
        )
        for record in malformed:
            with self.subTest(record=record), self.assertRaises(B.BindingError):
                B._tensor_record(record)

    def test_changed_input_cannot_be_declared_unchanged(self):
        plan, observation = self.exp_fixture()
        observation["tensor_arguments"]["x"]["after"]["v"][1] = 3.0
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("changed", report["reason"])

    def ce_fixture(self):
        subject = {"library": "example", "version": "1", "callable": "example.inplace_ce",
                   "boundary": "forward and scalar backward"}
        logits = tensor("f32", [[0.0, 0.0]], [1, 2])
        post = tensor("f32", [[-0.5, 0.5]], [1, 2])
        target = tensor("i64", [0], [1])
        plan = {
            "schema": "triton-call-binding-v1", "id": "example.inplace_ce.v1", "status": "drafted",
            "subject": subject, "launch_scope": "all_triton_launches_within_call",
            "tasks": [{
                "id": "ce", "contract": "CE_EXT", "profile": "liger_inplace_index_unsmoothed",
                "task_statement": "Two-dimensional, unsmoothed class-index CE with public logits post-state.",
                "input_map": [
                    {"target": "/logits", "source": "/tensor_arguments/logits/before",
                     "conversion": "tensor_values"},
                    {"target": "/target", "source": "/tensor_arguments/target/before",
                     "conversion": "tensor_values"},
                    {"target": "/weights", "literal": None, "basis": "public call default"},
                    {"target": "/eps", "literal": 0, "basis": "selected unsmoothed task"},
                    {"target": "/ignore_index", "literal": -100, "basis": "public call default"},
                    {"target": "/reduction", "literal": "mean", "basis": "public call default"},
                    {"target": "/upstream", "literal": 1, "basis": "scalar backward input"},
                ],
                "output_map": [
                    {"target": "/loss", "source": "/returns/loss"},
                    {"target": "/gradient", "source": "/gradients/logits", "conversion": "tensor_values"},
                    {"target": "/post_logits", "source": "/tensor_arguments/logits/after",
                     "conversion": "tensor_values"},
                ],
            }],
            "state_map": [
                {"argument": "logits", "effect": "output", "task": "ce", "target": "/post_logits"},
                {"argument": "target", "effect": "unchanged"},
            ],
        }
        observation = {
            "subject": subject, "launch_scope_complete": True, "launches": ["ce_kernel"],
            "tensor_arguments": {"logits": {"before": logits, "after": post},
                                 "target": {"before": target, "after": copy.deepcopy(target)}},
            "returns": {"loss": 0.6931471824645996}, "gradients": {"logits": copy.deepcopy(post)},
            "public_outputs": ["/returns/loss", "/gradients/logits", "/tensor_arguments/logits/after"],
        }
        return plan, observation

    def test_changed_input_is_accepted_only_as_a_declared_post_state(self):
        plan, observation = self.ce_fixture()
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "development_checked", report)
        self.assertEqual(report["reference_status"], "value")
        broken = copy.deepcopy(plan)
        broken["state_map"][0] = {"argument": "logits", "effect": "unchanged"}
        self.assertEqual(B.inspect(broken, observation)["binding_status"], "rejected")
        broken = copy.deepcopy(plan)
        broken["tasks"][0]["output_map"] = broken["tasks"][0]["output_map"][:-1]
        self.assertEqual(B.inspect(broken, observation)["binding_status"], "rejected")

    def test_nested_output_mapping_must_cover_the_complete_promised_structure(self):
        plan, observation = self.ce_fixture()
        gradient = plan["tasks"][0]["output_map"][1]
        gradient["target"] = "/gradient/0"
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("promised fields or positions", report["reason"])

    def test_reviewed_status_needs_the_exact_retained_readable_plan(self):
        plan, observation = self.exp_fixture()
        plan["status"] = "reviewed"
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "not_established")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.json"
            reviewed = root / "reviewed"
            reviewed.mkdir()
            current.write_text(json.dumps(plan, indent=2) + "\n")
            (reviewed / f"{plan['id']}.json").write_bytes(current.read_bytes())
            report = B.inspect(plan, observation, plan_path=current, reviewed_root=reviewed)
            self.assertEqual(report["binding_status"], "reviewed")
            (reviewed / f"{plan['id']}.json").write_text("{}\n")
            self.assertEqual(B.inspect(plan, observation, plan_path=current,
                                       reviewed_root=reviewed)["binding_status"], "not_established")

    def test_reviewed_plan_with_an_unresolved_issue_is_not_admitted(self):
        plan, observation = self.exp_fixture()
        plan["status"] = "reviewed"
        plan["unresolved"] = ["The measured package version has not been matched to the cited source version."]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.json"
            reviewed = root / "reviewed"
            reviewed.mkdir()
            current.write_text(json.dumps(plan, indent=2) + "\n")
            (reviewed / f"{plan['id']}.json").write_bytes(current.read_bytes())
            report = B.inspect(plan, observation, plan_path=current, reviewed_root=reviewed)
        self.assertEqual(report["binding_status"], "not_established")
        self.assertIn("source version", " ".join(report["binding_review_reasons"]))

    def test_binding_id_cannot_escape_the_review_directory(self):
        plan, observation = self.exp_fixture()
        plan["id"] = "../outside"
        self.assertEqual(B.inspect(plan, observation)["binding_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
