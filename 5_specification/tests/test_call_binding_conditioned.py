"""A task input taken from a value the call itself returned (conditioned reference; path A closure, native_dropout
given its returned mask). Allowed only when the plan declares it with a basis, and the returned value is itself a
compared output; otherwise every task input must still be a received input.
"""
import copy
import sys
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / "corpus")]
import call_binding as B  # noqa: E402
from test_call_binding import tensor  # noqa: E402


def fixture():
    """aux = exp(x) is returned and compared; out = x + aux is defined given the returned aux."""
    subject = {"library": "example", "version": "1", "callable": "example.exp_plus", "boundary": "one call"}
    x = tensor("f32", [0.0, 1.0], [2])
    aux = tensor("f32", [1.0, 2.7182817459106445], [2])
    out = tensor("f32", [1.0, 3.7182817459106445], [2])
    plan = {
        "schema": "triton-call-binding-v1", "id": "example.exp_plus.v1", "status": "drafted",
        "subject": subject, "launch_scope": "all_triton_launches_within_call",
        "tasks": [
            {"id": "aux", "contract": "MATHFN", "profile": "real", "task_statement": "aux = exp(x).",
             "input_map": [{"target": "/op", "literal": "aten.exp", "basis": "public operation"},
                           {"target": "/args/0", "source": "/tensor_arguments/x/before", "conversion": "tensor_record"}],
             "output_map": [{"target": "/out", "source": "/returns/aux", "conversion": "tensor_values"}]},
            {"id": "sum", "contract": "ARITH", "profile": "real", "task_statement": "out = x + aux, given aux.",
             "input_map": [{"target": "/op", "literal": "aten.add", "basis": "public operation"},
                           {"target": "/args/0", "source": "/tensor_arguments/x/before", "conversion": "tensor_record"},
                           {"target": "/args/1", "source": "/returns/aux", "conversion": "tensor_record"}],
             "output_map": [{"target": "/out", "source": "/returns/out", "conversion": "tensor_values"}]}],
        "state_map": [{"argument": "x", "effect": "unchanged"}],
        "conditioned_on_returns": [{"source": "/returns/aux", "basis": "out is defined given the returned aux"}],
    }
    observation = {
        "subject": subject, "launch_scope_complete": True, "launches": ["k"],
        "tensor_arguments": {"x": {"before": x, "after": copy.deepcopy(x)}},
        "returns": {"aux": aux, "out": out}, "public_outputs": ["/returns/aux", "/returns/out"],
    }
    return plan, observation


class Conditioned(unittest.TestCase):
    def reason(self, report):
        return str(report.get("reason"))

    def test_declared_conditioning_return_is_accepted(self):
        report = B.inspect(*fixture())
        self.assertEqual(report["binding_status"], "development_checked", report.get("reason"))
        self.assertEqual(report["conditioned_on_returns"], ["/returns/aux"])
        self.assertEqual(report["reference_status"], "value")

    def test_undeclared_return_input_is_rejected(self):
        plan, observation = fixture()
        del plan["conditioned_on_returns"]
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("not a received scalar or tensor pre-state", self.reason(report))

    def test_conditioning_return_must_be_compared(self):
        plan, observation = fixture()
        plan["tasks"] = plan["tasks"][1:]
        observation["public_outputs"] = ["/returns/out"]
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("must itself be a compared output", self.reason(report))

    def test_declaration_needs_basis_and_return_source(self):
        for bad in ([{"source": "/returns/aux", "basis": ""}], [{"source": "/tensor_arguments/x/before", "basis": "b"}],
                    [{"source": "/returns/aux"}], "/returns/aux"):
            plan, observation = fixture()
            plan["conditioned_on_returns"] = bad
            report = B.inspect(plan, observation)
            self.assertEqual(report["binding_status"], "rejected", bad)

    def test_declared_but_unused_return_is_rejected(self):
        plan, observation = fixture()
        plan["conditioned_on_returns"].append({"source": "/returns/out", "basis": "stated"})
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("is not used by any task input", self.reason(report))


if __name__ == "__main__":
    unittest.main()


class ShapeDifference(unittest.TestCase):
    def test_tensor_shape_difference_is_reported_with_both_shapes(self):
        from test_call_binding import CallBinding
        plan, observation = CallBinding.exp_fixture(None)
        plan["tasks"][0]["output_map"] = [{"target": "/out", "source": "/returns/out", "conversion": "tensor_record"}]
        observation["tensor_arguments"]["x"] = {"before": tensor("f32", [[0.0], [1.0]], [2, 1]),
                                                "after": tensor("f32", [[0.0], [1.0]], [2, 1])}
        observation["returns"]["out"] = tensor("f32", [1.0, 2.7182817459106445], [2])
        report = B.inspect(plan, observation)
        self.assertEqual(report["binding_status"], "rejected")
        self.assertIn("observed tensor out has shape (2,), the specification (2, 1)", report["reason"])
