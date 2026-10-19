"""Independent formula and PyTorch 2.10 checks for the reviewed public Rprop task."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry


DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
STRATEGIES = ({"foreach": False}, {"foreach": True}, {"foreach": None})
TOLERANCES = {torch.float16: 3e-3, torch.bfloat16: 3e-2, torch.float32: 2e-6, torch.float64: 1e-12}
CASES = (
    {
        "name": "cold_vector", "theta": [1, -2, 3, -4], "gradient": [2, -3, 0, 1], "state": None,
        "lr": 0.125, "etaminus": 0.5, "etaplus": 1.5, "step_min": 0.03125, "step_max": 0.75,
        "maximize": False,
    },
    {
        "name": "warm_all_signs", "theta": [1, -2, 3, -4], "gradient": [2, -3, 0, 1],
        "state": {"t": 3, "prev": [1, 1, -2, -1], "eta": [0.5, 0.5, 0.02, 0.6]},
        "lr": 0.125, "etaminus": 0.5, "etaplus": 1.5, "step_min": 0.03125, "step_max": 0.75,
        "maximize": False,
    },
    {
        "name": "warm_matrix_maximize", "theta": [[1, -2, 3], [-4, 2, 1]],
        "gradient": [[2, -3, 0], [1, -2, 3]],
        "state": {"t": 2, "prev": [[-1, -1, 2], [1, -1, 0]],
                  "eta": [[0.5, 0.5, 0.02], [0.6, 0.1, 1.0]]},
        "lr": 0.125, "etaminus": 0.5, "etaplus": 1.5, "step_min": 0.03125, "step_max": 0.75,
        "maximize": True,
    },
    {
        "name": "warm_rank3", "theta": [[[1, -2], [3, -4]], [[2, 1], [-1, 3]]],
        "gradient": [[[1, -1], [2, -2]], [[3, -3], [4, -4]]],
        "state": {"t": 4, "prev": [[[1, 1], [-1, -1]], [[0, 2], [-2, 0]]],
                  "eta": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]},
        "lr": 0.125, "etaminus": 0.5, "etaplus": 1.5, "step_min": 0.03125, "step_max": 0.75,
        "maximize": False,
    },
    {
        "name": "cold_lr_clamped", "theta": [1, -2, 3, -4], "gradient": [2, -3, 0, 1], "state": None,
        "lr": 1.0, "etaminus": 0.5, "etaplus": 1.5, "step_min": 0.03125, "step_max": 0.5,
        "maximize": False,
    },
)


def independent(case):
    theta = np.asarray(case["theta"], dtype=np.float64)
    gradient = np.asarray(case["gradient"], dtype=np.float64)
    if case["maximize"]:
        gradient = -gradient
    if case["state"] is None:
        step = 0
        prev = np.zeros_like(theta)
        eta = np.full_like(theta, case["lr"])
    else:
        step = case["state"]["t"]
        prev = np.asarray(case["state"]["prev"], dtype=np.float64)
        eta = np.asarray(case["state"]["eta"], dtype=np.float64)
    product = prev * gradient
    factors = np.where(product > 0, case["etaplus"], np.where(product < 0, case["etaminus"], 1.0))
    eta = np.clip(eta * factors, case["step_min"], case["step_max"])
    processed = np.where(product < 0, 0.0, gradient)
    return {"theta": theta - eta * np.sign(processed), "prev": processed, "eta": eta, "t": step + 1}


def task_input(case, dtype_name, **options):
    if case["state"] is None:
        state = {"theta": case["theta"], "prev": [], "eta": [], "t": None}
    else:
        state = {"theta": case["theta"], "prev": case["state"]["prev"],
                 "eta": case["state"]["eta"], "t": case["state"]["t"]}
    return {
        "state": state, "gradient": case["gradient"], "dtype": dtype_name,
        "lr": Fraction(str(case["lr"])), "etaminus": Fraction(str(case["etaminus"])),
        "etaplus": Fraction(str(case["etaplus"])), "step_min": Fraction(str(case["step_min"])),
        "step_max": Fraction(str(case["step_max"])), "maximize": case["maximize"], **options,
    }


class RpropIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RPROP").adapter

    def test_reference_equals_independent_formula(self):
        for case in CASES:
            with self.subTest(case=case["name"]):
                expected = independent(case)
                actual = self.adapter.reference("torch210_dense_real", task_input(case, "f64")).fields
                for key in ("theta", "prev", "eta"):
                    np.testing.assert_array_equal(
                        np.asarray(actual[key], dtype=float), expected[key], err_msg=f"{case['name']} {key}")
                self.assertEqual(actual["t"], expected["t"])

    def test_sixty_public_implementations_follow_the_same_transition(self):
        checked = 0
        dtype_names = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32", torch.float64: "f64"}
        for dtype in DTYPES:
            for case in CASES:
                expected = independent(case)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=str(dtype), case=case["name"], strategy=strategy):
                        parameter = torch.nn.Parameter(torch.tensor(case["theta"], dtype=dtype))
                        parameter.grad = torch.tensor(case["gradient"], dtype=dtype)
                        before_gradient = parameter.grad.clone()
                        optimizer = torch.optim.Rprop(
                            [parameter], lr=case["lr"], etas=(case["etaminus"], case["etaplus"]),
                            step_sizes=(case["step_min"], case["step_max"]), maximize=case["maximize"],
                            capturable=False, differentiable=False, **strategy)
                        if case["state"] is not None:
                            state = optimizer.state[parameter]
                            state["step"] = torch.tensor(float(case["state"]["t"]))
                            state["prev"] = torch.tensor(case["state"]["prev"], dtype=dtype)
                            state["step_size"] = torch.tensor(case["state"]["eta"], dtype=dtype)
                        optimizer.step()
                        state = optimizer.state[parameter]
                        self.assertEqual(set(state), {"step", "prev", "step_size"})
                        self.assertEqual(tuple(state["prev"].shape), tuple(parameter.shape))
                        self.assertEqual(tuple(state["step_size"].shape), tuple(parameter.shape))
                        tolerance = TOLERANCES[dtype]
                        torch.testing.assert_close(parameter.detach().float(), torch.tensor(expected["theta"]).float(),
                                                   rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(state["prev"].float(), torch.tensor(expected["prev"]).float(),
                                                   rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(state["step_size"].float(), torch.tensor(expected["eta"]).float(),
                                                   rtol=tolerance, atol=tolerance)
                        self.assertEqual(int(state["step"].item()), expected["t"])
                        self.assertTrue(torch.equal(parameter.grad, before_gradient))
                        reviewed = self.adapter.reference(
                            "torch210_dense_real", task_input(case, dtype_names[dtype], **strategy)).fields
                        self.assertEqual(reviewed["t"], expected["t"])
                        checked += 1
        self.assertEqual(checked, 60)

    def test_null_gradient_preserves_absent_and_initialized_state(self):
        parameter = torch.nn.Parameter(torch.ones((2, 3)))
        optimizer = torch.optim.Rprop([parameter], lr=0.1)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(dict(optimizer.state[parameter]), {})
        cold = task_input(CASES[0], "f32")
        cold["gradient"] = None
        self.assertEqual(self.adapter.reference("torch210_dense_real", cold).fields,
                         {"theta": [Fraction(1), -2, 3, -4], "prev": [], "eta": [], "t": None})

        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        before_parameter = parameter.detach().clone()
        before_state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
        parameter.grad = None
        optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        for key, value in before_state.items():
            self.assertTrue(torch.equal(optimizer.state[parameter][key], value))

    def test_zero_product_still_clamps_loaded_and_cold_step_size(self):
        for example_id in ("RP.E7", "RP.E8", "RP.E9"):
            contract = registry.load("RPROP").data
            example = next(item for item in contract["examples"] if item["id"] == example_id)
            actual = self.adapter.reference(example["profile"], example["input"])
            self.assertEqual(actual.outcome, "value")
            self.assertEqual(actual.fields["eta"], example["expected"]["fields"]["eta"])

    def test_complete_output_checker_rejects_missing_wrong_shape_and_dtype(self):
        one = {
            "state": {"theta": [1], "prev": [], "eta": [], "t": None}, "gradient": [2], "dtype": "f32",
            "lr": Fraction(1, 8), "etaminus": Fraction(1, 2), "etaplus": Fraction(3, 2),
            "step_min": Fraction(1, 32), "step_max": Fraction(3, 4), "maximize": False,
        }
        candidate = {"theta": [Fraction(7, 8)], "prev": [2], "eta": [Fraction(1, 8)], "t": 1}
        self.assertTrue(self.adapter.check_output("torch210_dense_real", one, candidate))
        missing = dict(candidate)
        missing.pop("prev")
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, missing))
        wrong_shape = dict(candidate)
        wrong_shape["eta"] = [[Fraction(1, 8)]]
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, wrong_shape))
        wrong_dtype = dict(candidate)
        wrong_dtype["theta"] = {"t": "f64", "shape": [1], "v": candidate["theta"]}
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, wrong_dtype))

    def test_unusual_public_bounds_are_refused_by_the_reviewed_profile(self):
        query = task_input(CASES[0], "f32")
        query.update(step_min=1, step_max=-1)
        result = registry.query("RPROP", "torch210_dense_real", query)
        self.assertEqual(result.outcome, "unsupported")


if __name__ == "__main__":
    unittest.main()
