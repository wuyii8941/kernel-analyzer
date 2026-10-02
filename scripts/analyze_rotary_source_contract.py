#!/usr/bin/env python3
"""Extract the reviewed fused-RoPE arithmetic contract from its saved source.

This is a source-level audit, not a numerical training result.  It deliberately
does not identify the source of the natural model difference by inspection
alone; it records the exact fused operations that remain candidates after the
same-operand arithmetic probe.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "results/property/numerical_coverage_v1/"
    "ministral_position_scaling_runtime_release_v4/trace/"
    "model__0_forward_segment0_executed/output_code.py"
)
SYMBOL = (
    "triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_div_expand_floor_log_"
    "mul_neg_sin_slice_transpose_unsqueeze_view_4"
)


def _function(text: str) -> ast.FunctionDef:
    # Inductor stores the Triton body inside a Python string passed to
    # ``async_compile.triton``; parsing the wrapper file does not expose the
    # nested function as an AST node.  Extract only the reviewed body first.
    match = re.search(
        rf"@triton\.jit\s+def {re.escape(SYMBOL)}\(.*?(?=\n'''\, device_str=)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"saved source does not contain {SYMBOL}")
    tree = ast.parse(match.group(0))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == SYMBOL:
            return node
    raise ValueError(f"saved source does not contain {SYMBOL}")


def _calls(fn: ast.FunctionDef) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(ast.unparse(node.func))
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    return calls


def build() -> dict[str, object]:
    fn = _function(SOURCE.read_text())
    calls = _calls(fn)
    stores = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "store" or len(node.args) < 2:
            continue
        pointer = ast.unparse(node.args[0])
        value = ast.unparse(node.args[1])
        stores.append({"pointer": pointer, "value": value})
    assigned = []
    for node in fn.body:
        if isinstance(node, ast.Assign):
            assigned.extend(ast.unparse(target) for target in node.targets)
    required = {
        "tl_math.cos": "tl_math.cos" in calls,
        "tl_math.sin": "tl_math.sin" in calls,
        "tl_math.log": "tl_math.log" in calls,
        "libdevice.floor": "libdevice.floor" in calls,
    }
    if not all(required.values()):
        raise ValueError(f"source operation contract incomplete: {required}")
    if {row["pointer"] for row in stores} != {
        "out_ptr0 + x4",
        "out_ptr1 + (x0 + 128 * x2 + 4096 * x1)",
    }:
        raise ValueError(f"unexpected output stores: {stores}")
    return {
        "schema": "rotary-source-contract-v1",
        "status": "COMPLETE_SAVED_SOURCE_AUDIT",
        "source": str(SOURCE.relative_to(ROOT)),
        "symbol": SYMBOL,
        "input_contract": {
            "query_storage": "bf16",
            "frequency_storage": "fp32",
            "position_storage": "int64",
            "intermediate_arithmetic": "fp32 after input loads",
            "output_storage": "bf16",
        },
        "operation_order": [
            "load query/frequency/position and cast inputs to FP32",
            "phase = frequency * position",
            "cosine and sine evaluation",
            "rotary products and addition",
            "floor(position * 2^-14), add one, log, multiply beta, add one",
            "final position scale multiplication",
            "BF16 stores to both output pointers",
        ],
        "required_operations": required,
        "output_stores": stores,
        "source_candidates_left_after_probe": [
            "fused FP32 operation ordering",
            "where BF16 materialization occurs relative to the fused expression",
            "real model operand and position distributions",
        ],
        "excluded_on_tested_device": [
            "tl_math versus libdevice trigonometric entry point",
        ],
        "claim_boundary": (
            "The saved source contract identifies the remaining arithmetic source class. "
            "It does not prove which candidate dominates the natural model trajectory "
            "without a same-operands intervention on live model operands."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
