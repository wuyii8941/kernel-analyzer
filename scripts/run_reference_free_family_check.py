#!/usr/bin/env python3
"""Run a declared-family diagnostic on importable Triton/PyTorch callables.

The module must expose the candidate and input factory.  A reduction check
also needs a same-semantic arithmetic variant; a saved-softmax check needs no
full reference because it tests the declared probability-mass contract.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from kernel_analyzer import check_reduction_order, check_softmax_saved_state


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
    parser.add_argument("--family", required=True, choices=("softmax_saved_state", "reduction_order"))
    parser.add_argument("--module", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--make-inputs", required=True)
    parser.add_argument("--variant", help="same-semantic reduction-order variant")
    parser.add_argument("--make-negated-inputs", help="optional input factory for reduction symmetry")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--probability-axis", type=int, default=-1)
    parser.add_argument("--output-index", type=int, default=None)
    parser.add_argument("--output-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        output = _repo_output(args.output, root)
        if args.samples < 2:
            raise ValueError("--samples must be at least 2")
        if args.output_index is not None and args.output_key is not None:
            raise ValueError("choose --output-index or --output-key, not both")
        candidate = _load(args.module, args.candidate)
        make_inputs = _load(args.module, args.make_inputs)
        if args.family == "softmax_saved_state":
            report = check_softmax_saved_state(
                candidate,
                make_inputs,
                output_selector=args.output_index if args.output_index is not None else args.output_key,
                probability_axis=args.probability_axis,
                tolerance=args.tolerance,
                samples=args.samples,
            )
        else:
            if not args.variant:
                raise ValueError("--variant is required for reduction_order")
            variant = _load(args.module, args.variant)
            negated = _load(args.module, args.make_negated_inputs) if args.make_negated_inputs else None
            report = check_reduction_order(
                candidate,
                variant,
                make_inputs,
                samples=args.samples,
                make_negated_inputs=negated,
            )
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"family diagnostic failed: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "family": report.get("family"),
        "measurement_status": report.get("measurement_status"),
        "diagnostic_status": report.get("diagnostic_status"),
        "bias_decision": report.get("bias_decision"),
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
