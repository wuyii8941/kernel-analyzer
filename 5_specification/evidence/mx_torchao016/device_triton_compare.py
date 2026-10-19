from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "5_specification"))

import torch
import triton
from triton.language.target_info import cuda_capability_geq

from kernel_spec import mx


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise RuntimeError("output must be a new file")

bridge = types.ModuleType("triton_kernels.target_info")
bridge.cuda_capability_geq = cuda_capability_geq
sys.modules[bridge.__name__] = bridge
source = ROOT / "5_specification/sources/MX_LOCAL_TRITON_DOWNCAST.py"
spec = importlib.util.spec_from_file_location("retained_mxfp8_device_evidence", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

safe = [64.0, -64.0, 32.0, -16.0] + [0.0] * 28
overflow = [127.0, -126.5, 112.0, -112.0] + [0.0] * 28
cases = [
    ("safe", safe),
    ("finite_to_infinity", overflow),
    ("zero", [0.0] * 32),
    ("code0_nonzero", [2.0 ** -127] * 32),
]


def observed_value(value):
    if value != value:
        return "nan"
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return str(Fraction(value))


report = {
    "scope": "Retained Triton E5M2 ROUND_DOWN kernel against admitted torchao 0.16.0 FLOOR task",
    "source": str(source.relative_to(ROOT)),
    "torch": torch.__version__,
    "triton": triton.__version__,
    "device": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "cases": [],
}
for name, values in cases:
    x = torch.tensor([values], dtype=torch.float32, device="cuda")
    expected = mx.torchao016_floor_tensor(x.cpu().tolist(), "e5m2")
    expected_flat = expected["elements"][0]
    runs = []
    for initial in (0x55, 0xAA):
        elements = torch.full((1, 32), initial, dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
        scales = torch.full((1, 1), initial, dtype=torch.uint8, device="cuda")
        module._downcast_to_mxfp[(1, 1)](
            elements, *elements.stride(), scales, *scales.stride(), x, *x.stride(), *x.shape,
            32, 128, 32, 1, num_warps=4,
        )
        torch.cuda.synchronize()
        numeric = elements.float().cpu().tolist()[0]
        bits = elements.view(torch.uint8).cpu().tolist()[0]
        scale_codes = scales.cpu().tolist()
        runs.append({
            "scale_codes": scale_codes,
            "element_bits": bits,
            "elements": [observed_value(value) for value in numeric],
            "scale_codes_match": scale_codes == expected["scale_codes"],
            "element_mismatches": sum(Fraction(value) != target for value, target in zip(numeric, expected_flat)
                                      if value not in (float("inf"), float("-inf")) and value == value)
                                  + sum(not target in (float("inf"), float("-inf"))
                                        for value, target in zip(numeric, expected_flat)
                                        if value in (float("inf"), float("-inf")) or value != value),
            "special_class_mismatches": sum(not (value == value and value not in (float("inf"), float("-inf")))
                                             for value in numeric),
        })
    report["cases"].append({
        "name": name,
        "input": [str(Fraction(value)) for value in values],
        "expected_scale_codes": expected["scale_codes"],
        "expected_elements": [str(value) for value in expected_flat],
        "repeat_identical": runs[0] == runs[1],
        "runs": runs,
    })

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(torch.cuda.get_device_name(), torch.__version__, triton.__version__)
for case in report["cases"]:
    run = case["runs"][0]
    print(case["name"], case["repeat_identical"], run["scale_codes_match"],
          run["element_mismatches"], run["special_class_mismatches"])
