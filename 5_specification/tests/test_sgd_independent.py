"""Independent coordinate formula and PyTorch 2.10 checks for reviewed public SGD."""
from fractions import Fraction
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry


DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
STRATEGIES = (
    {"foreach": False, "fused": False},
    {"foreach": True, "fused": False},
    {"foreach": None, "fused": None},
    {"foreach": False, "fused": True},
)
CASES = (
    {"name": "plain", "momentum": 0, "dampening": 0, "weight_decay": 0,
     "nesterov": False, "maximize": False, "buffer": []},
    {"name": "maximize_decay", "momentum": 0, "dampening": 0, "weight_decay": Fraction(1, 4),
     "nesterov": False, "maximize": True, "buffer": []},
    {"name": "first_momentum", "momentum": Fraction(1, 2), "dampening": Fraction(1, 2), "weight_decay": 0,
     "nesterov": False, "maximize": False, "buffer": []},
    {"name": "warm_dampening", "momentum": Fraction(1, 2), "dampening": Fraction(1, 2), "weight_decay": 0,
     "nesterov": False, "maximize": False, "buffer": [Fraction(1, 2), -1, 2, Fraction(-1, 2)]},
    {"name": "nesterov_maximize_decay", "momentum": Fraction(1, 2), "dampening": 0,
     "weight_decay": Fraction(1, 4), "nesterov": True, "maximize": True,
     "buffer": [Fraction(1, 2), -1, 2, Fraction(-1, 2)]},
)
THETA = [1, -2, 4, Fraction(1, 2)]
GRADIENT = [Fraction(1, 2), Fraction(-1, 4), 1, -2]
LR = Fraction(1, 8)


def independent(case):
    out, buffers = [], []
    for index, (old, received) in enumerate(zip(THETA, GRADIENT)):
        old, received = Fraction(old), Fraction(received)
        processed = (-received if case["maximize"] else received) + Fraction(case["weight_decay"]) * old
        if case["momentum"]:
            if case["buffer"]:
                buf = Fraction(case["momentum"]) * Fraction(case["buffer"][index]) + \
                    (1 - Fraction(case["dampening"])) * processed
            else:
                buf = processed
            buffers.append(buf)
            direction = processed + Fraction(case["momentum"]) * buf if case["nesterov"] else buf
        else:
            direction = processed
        out.append(old - LR * direction)
    return {"theta": out, "b": buffers if case["momentum"] else list(case["buffer"])}


def exact_tensor(values, dtype):
    return torch.tensor([float(value) for value in values], dtype=dtype)


class SGDIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SGD").adapter

    def test_eighty_public_implementations_match_independent_formula(self):
        checked = 0
        for dtype_name, dtype in DTYPES.items():
            for case in CASES:
                expected = independent(case)
                for strategy in STRATEGIES:
                    with self.subTest(dtype=dtype_name, case=case["name"], strategy=strategy):
                        inp = {
                            "state": {"theta": THETA, "b": case["buffer"]},
                            "gradient": GRADIENT,
                            "lr": LR,
                            "momentum": case["momentum"],
                            "dampening": case["dampening"],
                            "weight_decay": case["weight_decay"],
                            "nesterov": case["nesterov"],
                            "maximize": case["maximize"],
                            "differentiable": False,
                            **strategy,
                        }
                        self.assertEqual(self.adapter.reference("torch210", inp).fields, expected)

                        parameter = torch.nn.Parameter(exact_tensor(THETA, dtype))
                        parameter.grad = exact_tensor(GRADIENT, dtype)
                        optimizer = torch.optim.SGD(
                            [parameter], lr=float(LR), momentum=float(case["momentum"]),
                            dampening=float(case["dampening"]), weight_decay=float(case["weight_decay"]),
                            nesterov=case["nesterov"], maximize=case["maximize"],
                            differentiable=False, **strategy)
                        if case["buffer"]:
                            optimizer.state[parameter]["momentum_buffer"] = exact_tensor(case["buffer"], dtype)
                        optimizer.step()
                        actual_buffer = optimizer.state[parameter].get("momentum_buffer")
                        actual = {
                            "theta": [Fraction(value) for value in parameter.detach().tolist()],
                            "b": [] if actual_buffer is None else [Fraction(value) for value in actual_buffer.tolist()],
                        }
                        self.assertEqual(actual, expected)
                        checked += 1
        self.assertEqual(checked, 80)


if __name__ == "__main__":
    unittest.main()
