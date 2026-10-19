"""Capture generated Triton evidence for reviewed common activation bindings."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import torch
from torch._inductor.utils import run_and_get_triton_code


ROOT = Path(__file__).resolve().parents[3]
CALLS = ROOT / "5_specification/corpus/calls"
sys.path.insert(0, str(CALLS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("relu", "tanh", "sigmoid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-code", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if args.generated_code is not None and args.generated_code.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.generated_code}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    candidate = importlib.import_module(f"inductor_{args.candidate}")
    generator = torch.Generator(device="cpu").manual_seed(20261019)
    cpu_x = torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2
    cpu_x.reshape(-1)[:5] = torch.tensor([-2.0, -0.0, 0.0, 0.5, 2.0])
    cpu_inputs = {
        "x": cpu_x,
        "upstream": torch.randn(candidate.ROWS, candidate.COLS, generator=generator),
    }
    tensors = {name: value.cuda() for name, value in cpu_inputs.items()}
    before = {name: value.clone() for name, value in tensors.items()}
    generated = run_and_get_triton_code(
        candidate._compiled, tensors["x"], tensors["upstream"]
    )
    actual = candidate.call(tensors)
    eager = candidate._forward_and_vjp(tensors["x"], tensors["upstream"])
    outputs = {}
    for name, expected in zip(("y", "dx"), eager):
        got = actual[name]
        outputs[name] = {
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "finite": bool(torch.isfinite(got).all()),
            "max_abs_compiled_minus_eager": float((got - expected).abs().max().cpu()),
        }
    kernel_names = sorted(set(re.findall(
        r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE
    )))
    operation_pattern = re.compile(
        r"libdevice\.(?:tanh|exp)\b|tl\.(?:where|maximum|sigmoid)\b|[<>]=?\s*0\.0"
    )
    operation_lines = [
        line.strip() for line in generated.splitlines() if operation_pattern.search(line)
    ]
    report = {
        "task": f"PyTorch public {args.candidate} forward and complete input VJP",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "generated_operation_lines": operation_lines,
        "outputs": outputs,
        "inputs_unchanged": all(
            torch.equal(tensors[name], value) for name, value in before.items()
        ),
        "explicit_boundary_inputs": cpu_x.reshape(-1)[:5].tolist(),
        "explicit_boundary_outputs": {
            name: actual[name].reshape(-1)[:5].detach().cpu().tolist()
            for name in ("y", "dx")
        },
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["inputs_unchanged"]
        and all(item["finite"] for item in outputs.values())
        and all(item["shape"] == [candidate.ROWS, candidate.COLS]
                for item in outputs.values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.generated_code is not None:
        args.generated_code.parent.mkdir(parents=True, exist_ok=True)
        args.generated_code.write_text(generated)
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
