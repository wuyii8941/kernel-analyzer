"""Independent formula and PyTorch 2.10 checks for the reviewed public RAdam task."""
from fractions import Fraction
import math
import sys
import unittest
from pathlib import Path

from mpmath import mp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry
from kernel_spec.num import ends


DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
STRATEGIES = ({"foreach": False}, {"foreach": True}, {"foreach": None})
THETA = [1, -2, 4, Fraction(1, 2)]
GRADIENT = [Fraction(1, 2), Fraction(-1, 4), 1, -2]
LR = Fraction(1, 8)

CASES = (
    {"name": "cold_unrectified", "m": [], "v": [], "t": None,
     "beta1": Fraction(1, 2), "beta2": Fraction(3, 4), "eps": Fraction(1, 4),
     "weight_decay": 0, "maximize": False, "decoupled": False},
    {"name": "cold_decoupled", "m": [], "v": [], "t": None,
     "beta1": Fraction(1, 2), "beta2": Fraction(3, 4), "eps": Fraction(1, 4),
     "weight_decay": Fraction(1, 4), "maximize": False, "decoupled": True},
    {"name": "warm_threshold_unrectified", "m": [1, -1, Fraction(1, 2), 2],
     "v": [2, 1, 4, 1], "t": 4, "beta1": Fraction(1, 2), "beta2": Fraction(9, 10),
     "eps": Fraction(1, 4), "weight_decay": 0, "maximize": False, "decoupled": False},
    {"name": "warm_rectified", "m": [1, -1, Fraction(1, 2), 2],
     "v": [2, 1, 4, 1], "t": 5, "beta1": Fraction(1, 2), "beta2": Fraction(9, 10),
     "eps": Fraction(1, 4), "weight_decay": 0, "maximize": False, "decoupled": False},
    {"name": "warm_maximize_coupled", "m": [1, -1, Fraction(1, 2), 2],
     "v": [2, 1, 4, 1], "t": 5, "beta1": Fraction(1, 2), "beta2": Fraction(7, 8),
     "eps": Fraction(1, 4), "weight_decay": Fraction(1, 4), "maximize": True,
     "decoupled": False},
    {"name": "warm_zero_gradient_decoupled", "m": [1, -1, Fraction(1, 2), 2],
     "v": [2, 1, 4, 1], "t": 5, "gradient": [0, 0, 0, 0],
     "beta1": Fraction(1, 2), "beta2": Fraction(7, 8), "eps": Fraction(1, 4),
     "weight_decay": Fraction(1, 4), "maximize": False, "decoupled": True},
)


def independent(case):
    gradient = case.get("gradient", GRADIENT)
    if case["t"] is None:
        m = [Fraction(0)] * len(THETA)
        v = [Fraction(0)] * len(THETA)
        step0 = 0
    else:
        m = [Fraction(value) for value in case["m"]]
        v = [Fraction(value) for value in case["v"]]
        step0 = case["t"]
    beta1, beta2, decay = (Fraction(case[key]) for key in ("beta1", "beta2", "weight_decay"))
    received = [-Fraction(value) if case["maximize"] else Fraction(value) for value in gradient]
    if case["decoupled"]:
        bases = [Fraction(old) * (1 - LR * decay) for old in THETA]
        processed = received
    else:
        bases = [Fraction(old) for old in THETA]
        processed = [value + decay * Fraction(old) for old, value in zip(THETA, received)]
    m1 = [beta1 * old + (1 - beta1) * value for old, value in zip(m, processed)]
    v1 = [beta2 * old + (1 - beta2) * value * value for old, value in zip(v, processed)]
    step1 = step0 + 1
    correction1, correction2 = 1 - beta1 ** step1, 1 - beta2 ** step1
    rho_inf = 2 / (1 - beta2) - 1
    rho = rho_inf - 2 * step1 * beta2 ** step1 / correction2
    if rho > 5:
        rect = math.sqrt(float(
            (rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)))
        scale = math.sqrt(float(correction2)) * rect
        theta1 = [float(base) - float(LR * (moment / correction1)) * scale /
                  (math.sqrt(float(second)) + float(case["eps"]))
                  for base, moment, second in zip(bases, m1, v1)]
    else:
        theta1 = [float(base - LR * moment / correction1) for base, moment in zip(bases, m1)]
    return {"theta": theta1, "m": m1, "v": v1, "t": step1, "rho": rho}


def task_input(case, **options):
    return {
        "state": {"theta": THETA, "m": case["m"], "v": case["v"], "t": case["t"]},
        "gradient": case.get("gradient", GRADIENT), "lr": LR,
        "beta1": case["beta1"], "beta2": case["beta2"], "eps": case["eps"],
        "weight_decay": case["weight_decay"], "maximize": case["maximize"],
        "decoupled_weight_decay": case["decoupled"], "capturable": False,
        "differentiable": False, **options,
    }


class RAdamIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RADAM").adapter

    def test_reference_matches_independent_formula(self):
        mp.dps = 100
        for case in CASES:
            with self.subTest(case=case["name"]):
                expected = independent(case)
                actual = self.adapter.reference("torch210_dense_real", task_input(case)).fields
                self.assertEqual(actual["m"], expected["m"])
                self.assertEqual(actual["v"], expected["v"])
                self.assertEqual(actual["t"], expected["t"])
                for enclosure, point in zip(actual["theta"], expected["theta"]):
                    lo, hi = ends(enclosure)
                    self.assertLessEqual(float(lo), math.nextafter(point, math.inf))
                    self.assertGreaterEqual(float(hi), math.nextafter(point, -math.inf))

    def test_seventy_two_public_implementations_follow_the_same_transition(self):
        checked = 0
        tolerances = {torch.float16: 3e-3, torch.bfloat16: 3e-2,
                      torch.float32: 3e-6, torch.float64: 3e-12}
        for dtype in DTYPES:
            for case in CASES:
                expected = independent(case)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=str(dtype), case=case["name"], strategy=strategy):
                        parameter = torch.nn.Parameter(torch.tensor([float(value) for value in THETA], dtype=dtype))
                        parameter.grad = torch.tensor(
                            [float(value) for value in case.get("gradient", GRADIENT)], dtype=dtype)
                        optimizer = torch.optim.RAdam(
                            [parameter], lr=float(LR), betas=(float(case["beta1"]), float(case["beta2"])),
                            eps=float(case["eps"]), weight_decay=float(case["weight_decay"]),
                            maximize=case["maximize"], decoupled_weight_decay=case["decoupled"],
                            capturable=False, differentiable=False, **strategy)
                        if case["t"] is not None:
                            state = optimizer.state[parameter]
                            state["step"] = torch.tensor(float(case["t"]))
                            state["exp_avg"] = torch.tensor([float(value) for value in case["m"]], dtype=dtype)
                            state["exp_avg_sq"] = torch.tensor([float(value) for value in case["v"]], dtype=dtype)
                        optimizer.step()
                        state = optimizer.state[parameter]
                        self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq"})
                        self.assertEqual(int(state["step"].item()), expected["t"])
                        tolerance = tolerances[dtype]
                        torch.testing.assert_close(
                            parameter.detach().float(), torch.tensor(expected["theta"], dtype=torch.float32),
                            rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(
                            state["exp_avg"].float(), torch.tensor([float(value) for value in expected["m"]]),
                            rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(
                            state["exp_avg_sq"].float(), torch.tensor([float(value) for value in expected["v"]]),
                            rtol=tolerance, atol=tolerance)
                        checked += 1
        self.assertEqual(checked, 72)

    def test_versioned_branch_boundary_is_explicit(self):
        self.assertLess(independent(CASES[2])["rho"], 5)
        self.assertGreater(independent(CASES[3])["rho"], 5)

    def test_null_gradient_preserves_absent_and_initialized_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.RAdam([parameter], lr=0.1)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(dict(optimizer.state[parameter]), {})

        cold = task_input(CASES[0])
        cold["gradient"] = None
        self.assertEqual(self.adapter.reference("torch210_dense_real", cold).fields,
                         {"theta": [Fraction(1), -2, 4, Fraction(1, 2)],
                          "m": [], "v": [], "t": None})

        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        before = {key: value.clone() for key, value in optimizer.state[parameter].items()}
        before_parameter = parameter.detach().clone()
        parameter.grad = None
        optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        for key, value in before.items():
            self.assertTrue(torch.equal(optimizer.state[parameter][key], value))

    def test_complete_output_checker_rejects_missing_or_changed_state(self):
        one = task_input(CASES[0])
        one["state"] = {"theta": [1], "m": [], "v": [], "t": None}
        one["gradient"] = [2]
        candidate = {"theta": [Fraction(3, 4)], "m": [1], "v": [1], "t": 1}
        self.assertTrue(self.adapter.check_output("torch210_dense_real", one, candidate))
        missing = dict(candidate)
        missing.pop("v")
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, missing))
        changed = dict(candidate)
        changed["t"] = 2
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, changed))

    def test_rectified_relation_checker_rejects_a_wrong_coordinate(self):
        match = self.adapter._rectified_coordinate_matches
        self.assertTrue(match(Fraction(1), Fraction(0), Fraction(1), Fraction(1),
                              Fraction(1), Fraction(0), Fraction(1), Fraction(1), Fraction(1)))
        self.assertFalse(match(Fraction(1), Fraction(1, 10), Fraction(1), Fraction(1),
                               Fraction(1), Fraction(0), Fraction(1), Fraction(1), Fraction(1)))
        # This satisfies the twice-squared polynomial but not the original
        # equation: 2*(sqrt(1/4)+1) != 1*sqrt(1).
        self.assertFalse(match(Fraction(2), Fraction(0), Fraction(1), Fraction(1, 4),
                               Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)))


if __name__ == "__main__":
    unittest.main()
