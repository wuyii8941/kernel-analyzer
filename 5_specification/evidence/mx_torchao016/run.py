from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "5_specification"))

import torch
import torchao
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx

from kernel_spec import mx


def leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        yield value


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise RuntimeError("output must be a new file")

torch.manual_seed(619)
cases = [
    ("normal_f32", torch.tensor([[500.0, 1.0, -0.75] + [0.0] * 29], dtype=torch.float32)),
    ("two_blocks_f32", torch.cat([
        torch.tensor([[500.0, 1.0, -0.75] + [0.0] * 29], dtype=torch.float32),
        torch.ones((1, 32), dtype=torch.float32),
    ], dim=0)),
    ("zero_f32", torch.zeros((1, 32), dtype=torch.float32)),
    ("code0_nonzero_f32", torch.full((1, 32), 2.0 ** -127, dtype=torch.float32)),
    ("random_f32", torch.randn((4, 64), dtype=torch.float32) * 37.0),
    ("normal_bf16", torch.tensor([[500.0, 1.0, -0.75] + [0.0] * 29], dtype=torch.bfloat16)),
    ("random_bf16", (torch.randn((4, 64), dtype=torch.float32) * 37.0).to(torch.bfloat16)),
]

report = {
    "scope": "Installed torchao 0.16.0 finite non-swizzled to_mx FLOOR behavior; not universal OCP compliance",
    "torch": torch.__version__,
    "torchao": torchao.__version__,
    "cases": [],
}
for fmt, target in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
    for name, value in cases:
        scale, elements = to_mx(value, target, 32, ScaleCalculationMode.FLOOR, False)
        observed_scale = scale.view(torch.uint8).tolist()
        observed_elements = elements.tolist()
        expected = mx.torchao016_floor_tensor(value.tolist(), fmt)
        expected_values = list(leaves(expected["elements"]))
        observed_values = [Fraction(item) for item in leaves(observed_elements)]
        element_mismatches = sum(left != right for left, right in zip(expected_values, observed_values))
        report["cases"].append({
            "name": name,
            "format": fmt,
            "input_dtype": str(value.dtype),
            "shape": list(value.shape),
            "scale_codes_match": observed_scale == expected["scale_codes"],
            "element_coordinates": len(expected_values),
            "element_mismatches": element_mismatches,
            "scale_codes": observed_scale,
            "first_observed_elements": [str(item) for item in observed_values[:8]],
        })

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
if any(not case["scale_codes_match"] or case["element_mismatches"] for case in report["cases"]):
    raise SystemExit("torchao comparison mismatch")
print(f"{len(report['cases'])} cases matched")
