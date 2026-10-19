"""Independent formula and PyTorch 2.10 checks for the reviewed public Adam task."""
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
STRATEGIES = (
    {"foreach": False, "fused": False},
    {"foreach": True, "fused": False},
    {"foreach": None, "fused": None},
    {"foreach": False, "fused": True},
)
THETA = [1, -2, 4, Fraction(1, 2)]
GRADIENT = [Fraction(1, 2), Fraction(-1, 4), 1, -2]


CASES = (
    {"name": "cold", "m": [], "v": [], "vmax": [], "t": None,
     "beta1": Fraction(1, 2), "beta2": Fraction(1, 2), "eps": 0, "weight_decay": 0,
     "maximize": False, "amsgrad": False},
    {"name": "cold_amsgrad", "m": [], "v": [], "vmax": [], "t": None,
     "beta1": Fraction(1, 2), "beta2": Fraction(1, 2), "eps": Fraction(1, 4), "weight_decay": 0,
     "maximize": False, "amsgrad": True},
    {"name": "warm_amsgrad", "m": [1, -1, Fraction(1, 2), 2], "v": [2, 1, 4, 1],
     "vmax": [3, 1, 4, 2], "t": 3, "beta1": Fraction(1, 2), "beta2": Fraction(3, 4),
     "eps": Fraction(1, 4), "weight_decay": 0, "maximize": False, "amsgrad": True},
    {"name": "warm_maximize_decay", "m": [1, -1, Fraction(1, 2), 2], "v": [2, 1, 4, 1],
     "vmax": [], "t": 3, "beta1": Fraction(1, 2), "beta2": Fraction(3, 4),
     "eps": Fraction(1, 4), "weight_decay": Fraction(1, 4), "maximize": True, "amsgrad": False},
    {"name": "warm_zero_gradient", "m": [1, -1, Fraction(1, 2), 2], "v": [2, 1, 4, 1],
     "vmax": [], "t": 3, "gradient": [0, 0, 0, 0], "beta1": Fraction(1, 2),
     "beta2": Fraction(3, 4), "eps": Fraction(1, 4), "weight_decay": 0,
     "maximize": False, "amsgrad": False},
)
LR = Fraction(1, 8)


def independent(case):
    gradient = case.get("gradient", GRADIENT)
    if case["t"] is None:
        m = [Fraction(0)] * len(THETA)
        v = [Fraction(0)] * len(THETA)
        vmax = [Fraction(0)] * len(THETA) if case["amsgrad"] else []
        t0 = 0
    else:
        m, v, vmax = ([Fraction(x) for x in case[key]] for key in ("m", "v", "vmax"))
        t0 = case["t"]
    b1, b2, wd = (Fraction(case[key]) for key in ("beta1", "beta2", "weight_decay"))
    processed = [(-Fraction(g) if case["maximize"] else Fraction(g)) + wd * Fraction(x)
                 for x, g in zip(THETA, gradient)]
    m1 = [b1 * old + (1 - b1) * g for old, g in zip(m, processed)]
    v1 = [b2 * old + (1 - b2) * g * g for old, g in zip(v, processed)]
    vmax1 = [max(old, new) for old, new in zip(vmax, v1)] if case["amsgrad"] else []
    t1 = t0 + 1
    bc1, bc2 = 1 - b1 ** t1, 1 - b2 ** t1
    selected = vmax1 if case["amsgrad"] else v1
    theta1 = [float(x) - float(LR * (mn / bc1)) / (math.sqrt(float(vn / bc2)) + float(case["eps"]))
              for x, mn, vn in zip(THETA, m1, selected)]
    return {"theta": theta1, "m": m1, "v": v1, "vmax": vmax1, "t": t1}


def task_input(case, **options):
    return {
        "state": {"theta": THETA, "m": case["m"], "v": case["v"], "vmax": case["vmax"], "t": case["t"]},
        "gradient": case.get("gradient", GRADIENT), "lr": LR,
        "beta1": case["beta1"], "beta2": case["beta2"], "eps": case["eps"],
        "weight_decay": case["weight_decay"], "maximize": case["maximize"], "amsgrad": case["amsgrad"],
        "decoupled_weight_decay": False, "capturable": False, "differentiable": False, **options,
    }


class AdamIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("ADAM").adapter

    def test_reference_matches_independent_formula(self):
        mp.dps = 100
        for case in CASES:
            with self.subTest(case=case["name"]):
                expected = independent(case)
                actual = self.adapter.reference("torch210_dense_real", task_input(case)).fields
                self.assertEqual(actual["m"], expected["m"])
                self.assertEqual(actual["v"], expected["v"])
                self.assertEqual(actual["vmax"], expected["vmax"])
                self.assertEqual(actual["t"], expected["t"])
                for enclosure, point in zip(actual["theta"], expected["theta"]):
                    lo, hi = ends(enclosure)
                    # The independently evaluated binary64 point may differ in its last bit; it must lie no farther
                    # than one binary64 ulp from the much narrower strict-real enclosure.
                    self.assertLessEqual(float(lo), math.nextafter(point, math.inf))
                    self.assertGreaterEqual(float(hi), math.nextafter(point, -math.inf))

    def test_eighty_public_implementations_follow_the_same_state_transition(self):
        checked = 0
        tolerances = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 2e-6, torch.float64: 2e-12}
        for dtype in DTYPES:
            for case in CASES:
                expected = independent(case)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=str(dtype), case=case["name"], strategy=strategy):
                        parameter = torch.nn.Parameter(torch.tensor([float(x) for x in THETA], dtype=dtype))
                        parameter.grad = torch.tensor(
                            [float(x) for x in case.get("gradient", GRADIENT)], dtype=dtype)
                        optimizer = torch.optim.Adam(
                            [parameter], lr=float(LR), betas=(float(case["beta1"]), float(case["beta2"])),
                            eps=float(case["eps"]), weight_decay=float(case["weight_decay"]),
                            maximize=case["maximize"], amsgrad=case["amsgrad"],
                            decoupled_weight_decay=False, capturable=False, differentiable=False, **strategy)
                        if case["t"] is not None:
                            state = optimizer.state[parameter]
                            state["step"] = torch.tensor(float(case["t"]))
                            state["exp_avg"] = torch.tensor([float(x) for x in case["m"]], dtype=dtype)
                            state["exp_avg_sq"] = torch.tensor([float(x) for x in case["v"]], dtype=dtype)
                            if case["amsgrad"]:
                                state["max_exp_avg_sq"] = torch.tensor(
                                    [float(x) for x in case["vmax"]], dtype=dtype)
                        optimizer.step()
                        state = optimizer.state[parameter]
                        expected_keys = {"step", "exp_avg", "exp_avg_sq"} | \
                            ({"max_exp_avg_sq"} if case["amsgrad"] else set())
                        self.assertEqual(set(state), expected_keys)
                        self.assertEqual(int(state["step"].item()), expected["t"])
                        tol = tolerances[dtype]
                        torch.testing.assert_close(
                            parameter.detach().float(), torch.tensor(expected["theta"], dtype=torch.float32),
                            rtol=tol, atol=tol)
                        torch.testing.assert_close(
                            state["exp_avg"].float(), torch.tensor([float(x) for x in expected["m"]]),
                            rtol=tol, atol=tol)
                        torch.testing.assert_close(
                            state["exp_avg_sq"].float(), torch.tensor([float(x) for x in expected["v"]]),
                            rtol=tol, atol=tol)
                        if case["amsgrad"]:
                            torch.testing.assert_close(
                                state["max_exp_avg_sq"].float(),
                                torch.tensor([float(x) for x in expected["vmax"]]), rtol=tol, atol=tol)
                        checked += 1
        self.assertEqual(checked, 80)

    def test_null_gradient_preserves_absent_and_initialized_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.1, amsgrad=True)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(dict(optimizer.state[parameter]), {})
        cold = task_input(CASES[1])
        cold["gradient"] = None
        self.assertEqual(self.adapter.reference("torch210_dense_real", cold).fields,
                         {"theta": [Fraction(1), -2, 4, Fraction(1, 2)],
                          "m": [], "v": [], "vmax": [], "t": None})

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
        inp = task_input(CASES[0])
        candidate = {"theta": [Fraction(7, 8)], "m": [1], "v": [2], "vmax": [], "t": 1}
        one = dict(inp)
        one["state"] = {"theta": [1], "m": [], "v": [], "vmax": [], "t": None}
        one["gradient"] = [2]
        self.assertTrue(self.adapter.check_output("torch210_dense_real", one, candidate))
        candidate.pop("v")
        self.assertFalse(self.adapter.check_output("torch210_dense_real", one, candidate))


if __name__ == "__main__":
    unittest.main()
