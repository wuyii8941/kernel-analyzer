#!/usr/bin/env python3
"""Run a declared-family diagnostic on importable TileLang callables.

The module must expose a compiled (or ``@tilelang.jit``) candidate callable
and an input factory.  This command deliberately does not infer semantics or
generate a reference.  ``softmax_saved_state`` checks a caller-declared
probability-mass property; ``reduction_order`` compares two caller-declared
same-semantic TileLang variants; ``odd_symmetry`` needs only one candidate and
automatically negates floating inputs, but still requires the caller to declare
that the operator should satisfy ``f(-x) == -f(x)``.

TileLang is optional at import time.  The command can be exercised with CPU
callables to validate the adapter contract, while actual TileLang kernels
require a compatible TileLang/CUDA environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from kernel_analyzer import (
    check_tilelang_odd_symmetry,
    check_tilelang_reduction_order,
    check_tilelang_softmax_saved_state,
)


def _load(module_name: str, attribute: str) -> Callable[..., Any]:
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return value


def _repo_output(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("--output must be inside the repository") from error
    if path.exists():
        raise ValueError(f"refusing to overwrite existing report: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", required=True,
        choices=("softmax_saved_state", "reduction_order", "odd_symmetry"),
    )
    parser.add_argument("--module", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--make-inputs", required=True)
    parser.add_argument("--variant", help="same-semantic reduction-order variant")
    parser.add_argument("--make-negated-inputs", help="optional reduction symmetry factory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--probability-axis", type=int, default=-1)
    parser.add_argument("--candidate-output-index", type=int, default=None)
    parser.add_argument("--candidate-output-key", default=None)
    parser.add_argument("--variant-output-index", type=int, default=None)
    parser.add_argument("--variant-output-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        output = _repo_output(args.output, root)
        if args.samples < 1:
            raise ValueError("--samples must be at least 1")
        if args.candidate_output_index is not None and args.candidate_output_key is not None:
            raise ValueError("choose --candidate-output-index or --candidate-output-key, not both")
        if args.variant_output_index is not None and args.variant_output_key is not None:
            raise ValueError("choose --variant-output-index or --variant-output-key, not both")
        candidate = _load(args.module, args.candidate)
        make_inputs = _load(args.module, args.make_inputs)
        if args.family == "softmax_saved_state":
            report = check_tilelang_softmax_saved_state(
                candidate,
                make_inputs,
                candidate_output_index=args.candidate_output_index,
                candidate_output_key=args.candidate_output_key,
                probability_axis=args.probability_axis,
                tolerance=args.tolerance,
                samples=args.samples,
            )
        elif args.family == "odd_symmetry":
            report = check_tilelang_odd_symmetry(
                candidate,
                make_inputs,
                candidate_output_index=args.candidate_output_index,
                candidate_output_key=args.candidate_output_key,
                tolerance=args.tolerance,
                samples=args.samples,
            )
        else:
            if not args.variant:
                raise ValueError("--variant is required for reduction_order")
            variant = _load(args.module, args.variant)
            negated = (
                _load(args.module, args.make_negated_inputs)
                if args.make_negated_inputs
                else None
            )
            report = check_tilelang_reduction_order(
                candidate,
                variant,
                make_inputs,
                candidate_output_index=args.candidate_output_index,
                candidate_output_key=args.candidate_output_key,
                variant_output_index=args.variant_output_index,
                variant_output_key=args.variant_output_key,
                make_negated_inputs=negated,
                samples=args.samples,
            )
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"TileLang family diagnostic failed: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": report.get("backend"),
                "family": report.get("family"),
                "measurement_status": report.get("measurement_status"),
                "diagnostic_status": report.get("diagnostic_status"),
                "bias_decision": report.get("bias_decision"),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
