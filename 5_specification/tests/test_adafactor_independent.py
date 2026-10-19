"""Independent formula and PyTorch 2.10 checks for the reviewed public Adafactor task."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry
from kernel_spec.num import ends
from kernel_spec.real import R


DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
STRATEGIES = ({"foreach": False}, {"foreach": True}, {"foreach": None})
TOLERANCES = {
    torch.float16: 8e-3,
    torch.bfloat16: 6e-2,
    torch.float32: 5e-6,
    torch.float64: 2e-12,
}
CASES = (
    {
        "name": "cold_vector", "theta": [1, -2, 3, -4], "gradient": [0.5, -0.25, 1, -2],
        "state": None, "lr": 0.125, "beta2_decay": -0.8, "eps1": 0.25,
        "eps2": 0.125, "d": 1.0, "weight_decay": 0.0, "maximize": False,
    },
    {
        "name": "warm_vector_decay_maximize", "theta": [1, -2, 3, -4],
        "gradient": [0.5, -0.25, 1, -2], "state": {"t": 3, "var": [1, 2, 3, 4]},
        "lr": 0.125, "beta2_decay": -1.0, "eps1": 0.25, "eps2": 0.125,
        "d": 1.0, "weight_decay": 0.25, "maximize": True,
    },
    {
        "name": "cold_matrix_clamp", "theta": [[1, -2, 3], [-4, 2, 1]],
        "gradient": [[0.01, 0, -0.02], [0, 0.03, 0]], "state": None,
        "lr": 0.125, "beta2_decay": -0.8, "eps1": 0.25, "eps2": 0.125,
        "d": 1.0, "weight_decay": 0.0, "maximize": False,
    },
    {
        "name": "warm_batched_matrix", "theta": [[[1, -2, 3], [-4, 2, 1]], [[2, 1, -1], [3, -3, 2]]],
        "gradient": [[[0.5, -0.25, 1], [-2, 0.5, 0.25]], [[-1, 2, -0.5], [0.25, -1, 0.75]]],
        "state": {"t": 3,
                  "row": [[[1], [2]], [[3], [4]]],
                  "col": [[[1, 2, 3]], [[4, 3, 2]]]},
        "lr": 0.125, "beta2_decay": -1.0, "eps1": None, "eps2": 0.125,
        "d": 1.0, "weight_decay": 0.25, "maximize": False,
    },
)


def independent(case, eps1):
    theta = np.asarray(case["theta"], dtype=np.float64)
    gradient = np.asarray(case["gradient"], dtype=np.float64)
    if case["maximize"]:
        gradient = -gradient
    state = case["state"]
    step0 = 0 if state is None else state["t"]
    step = step0 + 1
    weight = step ** case["beta2_decay"]
    rho = min(case["lr"], 1 / math.sqrt(step))
    alpha = max(case["eps2"], math.sqrt(float(np.mean(theta * theta)))) * rho
    base = theta * (1 - case["lr"] * case["weight_decay"])
    if theta.ndim == 1:
        old = np.zeros_like(theta) if state is None else np.asarray(state["var"], dtype=np.float64)
        var = old + weight * (gradient * gradient - old)
        estimate = var
        row = col = None
    else:
        row_old = np.zeros(theta.shape[:-1] + (1,)) if state is None else np.asarray(state["row"], dtype=np.float64)
        col_old = np.zeros(theta.shape[:-2] + (1, theta.shape[-1])) if state is None \
            else np.asarray(state["col"], dtype=np.float64)
        row_target = np.mean(gradient * gradient, axis=-1, keepdims=True)
        col_target = np.mean(gradient * gradient, axis=-2, keepdims=True)
        row = row_old + weight * (row_target - row_old)
        col = col_old + weight * (col_target - col_old)
        estimate = np.matmul(row, col) / np.maximum(np.mean(row, axis=-2, keepdims=True), eps1)
        var = None
    update = gradient / np.sqrt(np.maximum(estimate, eps1 * eps1))
    update /= max(1.0, math.sqrt(float(np.mean(update * update))) / case["d"])
    result = {"theta": base - alpha * update, "t": step}
    if theta.ndim == 1:
        result["var"] = var
    else:
        result.update(row=row, col=col)
    return result


def task_input(case, dtype_name, **options):
    theta = case["theta"]
    if case["state"] is None:
        state = {"theta": theta, "row": [], "col": [], "var": [], "t": None}
    elif np.asarray(theta).ndim == 1:
        state = {"theta": theta, "row": [], "col": [], "var": case["state"]["var"], "t": case["state"]["t"]}
    else:
        state = {"theta": theta, "row": case["state"]["row"], "col": case["state"]["col"],
                 "var": [], "t": case["state"]["t"]}
    return {
        "state": state, "gradient": case["gradient"], "dtype": dtype_name, "lr": Fraction(1, 8),
        "beta2_decay": Fraction(str(case["beta2_decay"])),
        "eps1": None if case["eps1"] is None else Fraction(str(case["eps1"])),
        "eps2": Fraction(str(case["eps2"])), "d": Fraction(str(case["d"])),
        "weight_decay": Fraction(str(case["weight_decay"])), "maximize": case["maximize"], **options,
    }


def assert_contains(test, reference, expected):
    if isinstance(expected, np.ndarray):
        test.assertEqual(np.asarray(reference, dtype=object).shape, expected.shape)
        for actual, wanted in zip(np.asarray(reference, dtype=object).flat, expected.flat):
            value = actual.v if isinstance(actual, R) else actual
            lo, hi = ends(value)
            test.assertLessEqual(float(lo), math.nextafter(float(wanted), math.inf))
            test.assertGreaterEqual(float(hi), math.nextafter(float(wanted), -math.inf))
    else:
        test.assertEqual(reference, expected)


class AdafactorIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("ADAFACTOR").adapter

    def test_reference_contains_independent_formula(self):
        for case in CASES:
            shape = np.asarray(case["theta"]).shape
            profile = "torch210_vector_real" if len(shape) == 1 else "torch210_factored_real"
            eps1 = case["eps1"] if case["eps1"] is not None else float(torch.finfo(torch.float64).eps)
            expected = independent(case, eps1)
            actual = self.adapter.reference(profile, task_input(case, "f64")).fields
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                assert_contains(self, actual[key], expected[key])

    def test_forty_eight_public_implementations_follow_the_same_transition(self):
        checked = 0
        dtype_names = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32", torch.float64: "f64"}
        for dtype in DTYPES:
            for case in CASES:
                eps1 = case["eps1"] if case["eps1"] is not None else float(torch.finfo(dtype).eps)
                expected = independent(case, eps1)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=str(dtype), case=case["name"], strategy=strategy):
                        parameter = torch.nn.Parameter(torch.tensor(case["theta"], dtype=dtype))
                        parameter.grad = torch.tensor(case["gradient"], dtype=dtype)
                        before_gradient = parameter.grad.clone()
                        optimizer = torch.optim.Adafactor(
                            [parameter], lr=case["lr"], beta2_decay=case["beta2_decay"],
                            eps=(case["eps1"], case["eps2"]), d=case["d"],
                            weight_decay=case["weight_decay"], maximize=case["maximize"], **strategy)
                        if case["state"] is not None:
                            state = optimizer.state[parameter]
                            state["step"] = torch.tensor(float(case["state"]["t"]))
                            if parameter.ndim == 1:
                                state["variance"] = torch.tensor(case["state"]["var"], dtype=dtype)
                            else:
                                state["row_var"] = torch.tensor(case["state"]["row"], dtype=dtype)
                                state["col_var"] = torch.tensor(case["state"]["col"], dtype=dtype)
                        optimizer.step()
                        state = optimizer.state[parameter]
                        expected_keys = {"step", "variance"} if parameter.ndim == 1 \
                            else {"step", "row_var", "col_var"}
                        self.assertEqual(set(state), expected_keys)
                        self.assertEqual(tuple(state["step"].shape), ())
                        tolerance = TOLERANCES[dtype]
                        torch.testing.assert_close(parameter.detach().float(), torch.tensor(expected["theta"]).float(),
                                                   rtol=tolerance, atol=tolerance)
                        if parameter.ndim == 1:
                            self.assertEqual(tuple(state["variance"].shape), tuple(parameter.shape))
                            torch.testing.assert_close(state["variance"].float(), torch.tensor(expected["var"]).float(),
                                                       rtol=tolerance, atol=tolerance)
                        else:
                            self.assertEqual(tuple(state["row_var"].shape), tuple(parameter.shape[:-1]) + (1,))
                            self.assertEqual(tuple(state["col_var"].shape),
                                             tuple(parameter.shape[:-2]) + (1, parameter.shape[-1]))
                            torch.testing.assert_close(state["row_var"].float(), torch.tensor(expected["row"]).float(),
                                                       rtol=tolerance, atol=tolerance)
                            torch.testing.assert_close(state["col_var"].float(), torch.tensor(expected["col"]).float(),
                                                       rtol=tolerance, atol=tolerance)
                        self.assertEqual(int(state["step"].item()), expected["t"])
                        self.assertTrue(torch.equal(parameter.grad, before_gradient))
                        # Also exercise the reviewed adapter with exactly this dtype choice.
                        result = self.adapter.reference(
                            "torch210_vector_real" if parameter.ndim == 1 else "torch210_factored_real",
                            task_input(case, dtype_names[dtype], **strategy)).fields
                        self.assertEqual(result["t"], expected["t"])
                        checked += 1
        self.assertEqual(checked, 48)

    def test_null_gradient_preserves_absent_and_initialized_state(self):
        parameter = torch.nn.Parameter(torch.ones((2, 3)))
        optimizer = torch.optim.Adafactor([parameter], lr=0.1)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(dict(optimizer.state[parameter]), {})

        cold = task_input(CASES[2], "f32")
        cold["gradient"] = None
        self.assertEqual(self.adapter.reference("torch210_factored_real", cold).fields,
                         {"theta": [[Fraction(1), -2, 3], [-4, 2, 1]], "row": [], "col": [], "t": None})

        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        before_parameter = parameter.detach().clone()
        before_state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
        parameter.grad = None
        optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        for key, value in before_state.items():
            self.assertTrue(torch.equal(optimizer.state[parameter][key], value))

    def test_complete_output_checker_rejects_missing_state_and_squeezed_axes(self):
        case = {
            "state": {"theta": [[1, 1], [1, 1]], "row": [], "col": [], "var": [], "t": None},
            "gradient": [[1, 1], [1, 1]], "dtype": "f32", "lr": Fraction(1, 10),
            "beta2_decay": -1, "eps1": 0, "eps2": 0, "d": 1, "weight_decay": 0, "maximize": False,
        }
        candidate = {"theta": [[Fraction(9, 10)] * 2] * 2, "row": [[1], [1]], "col": [[1, 1]], "t": 1}
        self.assertTrue(self.adapter.check_output("torch210_factored_real", case, candidate))
        missing = dict(candidate)
        missing.pop("col")
        self.assertFalse(self.adapter.check_output("torch210_factored_real", case, missing))
        squeezed = dict(candidate)
        squeezed["row"] = [1, 1]
        self.assertFalse(self.adapter.check_output("torch210_factored_real", case, squeezed))
        recorded = {
            "theta": {"t": "f64", "v": candidate["theta"], "shape": [2, 2]},
            "row": candidate["row"], "col": candidate["col"], "t": candidate["t"],
        }
        self.assertFalse(self.adapter.check_output("torch210_factored_real", case, recorded))
        bad_step = dict(candidate)
        bad_step["t"] = {"t": "mystery", "v": 1, "shape": []}
        self.assertFalse(self.adapter.check_output("torch210_factored_real", case, bad_step))

    def test_clamp_active_case_has_a_defined_answer(self):
        contract = registry.load("ADAFACTOR").data
        example = next(item for item in contract["examples"] if item["id"] == "AF.E11")
        result = self.adapter.reference(example["profile"], example["input"])
        self.assertEqual(result.outcome, "value")
        self.assertEqual(result.fields["row"], [[Fraction(1, 2_000_000)], [Fraction(0)]])

    def test_reference_budget_refuses_large_step_power(self):
        query = task_input(CASES[1], "f32")
        query["state"]["t"] = 10000
        result = registry.query("ADAFACTOR", "torch210_vector_real", query)
        self.assertEqual(result.outcome, "not_established")


if __name__ == "__main__":
    unittest.main()
