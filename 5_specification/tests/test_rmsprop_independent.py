"""Independent formula and PyTorch 2.10 checks for the reviewed public RMSprop task."""
from fractions import Fraction
import math
import sys
import unittest
from pathlib import Path

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
    {"name": "cold", "v": [], "gave": [], "b": [], "t": None, "alpha": Fraction(3, 4),
     "eps": Fraction(1, 4), "weight_decay": 0, "momentum": 0, "centered": False, "maximize": False},
    {"name": "cold_centered_momentum", "v": [], "gave": [], "b": [], "t": None,
     "alpha": Fraction(3, 4), "eps": Fraction(1, 4), "weight_decay": 0,
     "momentum": Fraction(1, 2), "centered": True, "maximize": False},
    {"name": "warm_centered_maximize_decay", "v": [2, 1, 4, 1],
     "gave": [Fraction(1, 2), Fraction(-1, 4), 1, Fraction(-1, 2)],
     "b": [1, -1, Fraction(1, 2), 2], "t": 3, "alpha": Fraction(3, 4),
     "eps": Fraction(1, 4), "weight_decay": Fraction(1, 4), "momentum": Fraction(1, 2),
     "centered": True, "maximize": True},
    {"name": "warm_uncentered", "v": [2, 1, 4, 1], "gave": [], "b": [], "t": 3,
     "alpha": Fraction(3, 4), "eps": Fraction(1, 4), "weight_decay": Fraction(1, 4),
     "momentum": 0, "centered": False, "maximize": False},
    {"name": "warm_momentum", "v": [2, 1, 4, 1], "gave": [], "b": [1, -1, Fraction(1, 2), 2], "t": 3,
     "alpha": Fraction(3, 4), "eps": Fraction(1, 4), "weight_decay": 0,
     "momentum": Fraction(1, 2), "centered": False, "maximize": False},
)


def independent(case):
    if case["t"] is None:
        v = [Fraction(0)] * len(THETA)
        gave = [Fraction(0)] * len(THETA) if case["centered"] else []
        buffer = [Fraction(0)] * len(THETA) if case["momentum"] > 0 else []
        step = 0
    else:
        v = [Fraction(x) for x in case["v"]]
        gave = [Fraction(x) for x in case["gave"]]
        buffer = [Fraction(x) for x in case["b"]]
        step = case["t"]
    alpha, wd, momentum = (Fraction(case[key]) for key in ("alpha", "weight_decay", "momentum"))
    processed = [(-Fraction(g) if case["maximize"] else Fraction(g)) + wd * Fraction(x)
                 for x, g in zip(THETA, GRADIENT)]
    v1 = [alpha * old + (1 - alpha) * g * g for old, g in zip(v, processed)]
    gave1 = [alpha * old + (1 - alpha) * g for old, g in zip(gave, processed)] if case["centered"] else []
    adjusted = [new - avg * avg for new, avg in zip(v1, gave1)] if case["centered"] else v1
    quotient = [float(g) / (math.sqrt(float(square)) + float(case["eps"]))
                for g, square in zip(processed, adjusted)]
    if momentum > 0:
        buffer1 = [float(momentum * old) + q for old, q in zip(buffer, quotient)]
        theta1 = [float(old) - float(LR) * new for old, new in zip(THETA, buffer1)]
    else:
        buffer1 = []
        theta1 = [float(old) - float(LR) * q for old, q in zip(THETA, quotient)]
    return {"theta": theta1, "v": v1, "gave": gave1, "b": buffer1, "t": step + 1}


def task_input(case, **options):
    return {
        "state": {"theta": THETA, "v": case["v"], "gave": case["gave"], "b": case["b"], "t": case["t"]},
        "gradient": GRADIENT, "lr": LR, "alpha": case["alpha"], "eps": case["eps"],
        "weight_decay": case["weight_decay"], "momentum": case["momentum"],
        "centered": case["centered"], "maximize": case["maximize"],
        "capturable": False, "differentiable": False, **options,
    }


class RMSpropIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RMSPROP").adapter

    def test_reference_matches_independent_formula(self):
        for case in CASES:
            with self.subTest(case=case["name"]):
                expected = independent(case)
                actual = self.adapter.reference("torch210_dense_real", task_input(case)).fields
                self.assertEqual(actual["v"], expected["v"])
                self.assertEqual(actual["gave"], expected["gave"])
                self.assertEqual(actual["t"], expected["t"])
                for field in ("theta", "b"):
                    self.assertEqual(len(actual[field]), len(expected[field]))
                    for enclosure, point in zip(actual[field], expected[field]):
                        lo, hi = ends(enclosure)
                        self.assertLessEqual(float(lo), math.nextafter(point, math.inf))
                        self.assertGreaterEqual(float(hi), math.nextafter(point, -math.inf))

    def test_sixty_public_implementations_follow_the_same_state_transition(self):
        checked = 0
        tolerances = {torch.float16: 3e-3, torch.bfloat16: 3e-2, torch.float32: 3e-6, torch.float64: 3e-12}
        for dtype in DTYPES:
            for case in CASES:
                expected = independent(case)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=str(dtype), case=case["name"], strategy=strategy):
                        parameter = torch.nn.Parameter(torch.tensor([float(x) for x in THETA], dtype=dtype))
                        parameter.grad = torch.tensor([float(x) for x in GRADIENT], dtype=dtype)
                        optimizer = torch.optim.RMSprop(
                            [parameter], lr=float(LR), alpha=float(case["alpha"]), eps=float(case["eps"]),
                            weight_decay=float(case["weight_decay"]), momentum=float(case["momentum"]),
                            centered=case["centered"], maximize=case["maximize"], capturable=False,
                            differentiable=False, **strategy)
                        if case["t"] is not None:
                            state = optimizer.state[parameter]
                            state["step"] = torch.tensor(float(case["t"]))
                            state["square_avg"] = torch.tensor([float(x) for x in case["v"]], dtype=dtype)
                            if case["centered"]:
                                state["grad_avg"] = torch.tensor([float(x) for x in case["gave"]], dtype=dtype)
                            if case["momentum"] > 0:
                                state["momentum_buffer"] = torch.tensor([float(x) for x in case["b"]], dtype=dtype)
                        optimizer.step()
                        state = optimizer.state[parameter]
                        expected_keys = {"step", "square_avg"}
                        if case["centered"]:
                            expected_keys.add("grad_avg")
                        if case["momentum"] > 0:
                            expected_keys.add("momentum_buffer")
                        self.assertEqual(set(state), expected_keys)
                        self.assertEqual(int(state["step"].item()), expected["t"])
                        tol = tolerances[dtype]
                        torch.testing.assert_close(parameter.detach().float(), torch.tensor(expected["theta"]),
                                                   rtol=tol, atol=tol)
                        torch.testing.assert_close(state["square_avg"].float(),
                                                   torch.tensor([float(x) for x in expected["v"]]),
                                                   rtol=tol, atol=tol)
                        if case["centered"]:
                            torch.testing.assert_close(state["grad_avg"].float(),
                                                       torch.tensor([float(x) for x in expected["gave"]]),
                                                       rtol=tol, atol=tol)
                        if case["momentum"] > 0:
                            torch.testing.assert_close(state["momentum_buffer"].float(),
                                                       torch.tensor(expected["b"]), rtol=tol, atol=tol)
                        checked += 1
        self.assertEqual(checked, 60)

    def test_null_gradient_preserves_absent_and_initialized_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.RMSprop([parameter], lr=0.1, momentum=0.5, centered=True)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(dict(optimizer.state[parameter]), {})

        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        before = {key: value.clone() for key, value in optimizer.state[parameter].items()}
        before_parameter = parameter.detach().clone()
        parameter.grad = None
        optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        for key, value in before.items():
            self.assertTrue(torch.equal(optimizer.state[parameter][key], value))

    def test_complete_output_checker_rejects_missing_state(self):
        one = {"state": {"theta": [1], "v": [], "gave": [], "b": [], "t": None}, "gradient": [2],
               "lr": Fraction(1, 10), "alpha": Fraction(3, 4), "eps": 0, "weight_decay": 0,
               "momentum": 0, "centered": False, "maximize": False}
        candidate = {"theta": [Fraction(4, 5)], "v": [1], "gave": [], "b": [], "t": 1}
        self.assertTrue(self.adapter.check_output("torch210_dense_real", one, candidate))
        candidate.pop("v")
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, candidate))


if __name__ == "__main__":
    unittest.main()
