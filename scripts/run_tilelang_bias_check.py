#!/usr/bin/env python3
"""Run the lightweight TileLang bias checker from importable callables.

This is intentionally a small orchestration layer.  It does not compile
arbitrary source, synthesize a reference, or infer a root cause.  A module
must expose a compiled/callable candidate, a reference callable, and an input
factory; the statistical work is delegated to ``check_tilelang_bias``.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from kernel_analyzer import check_tilelang_bias


def _load_callable(module_name: str, attribute: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return value


def _load_optional_callable(module_name: str, attribute: str | None) -> Callable[..., Any] | None:
    if attribute is None:
        return None
    return _load_callable(module_name, attribute)


def _inside_repo(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="Python module containing the callables")
    parser.add_argument("--candidate", required=True, help="Attribute for the compiled TileLang candidate")
    parser.add_argument("--reference", required=True, help="Attribute for the reference callable")
    parser.add_argument("--make-inputs", required=True, help="Attribute returning inputs for one sample index")
    parser.add_argument("--output", required=True, help="JSON report path inside the repository")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--candidate-output-index", type=int, default=None)
    parser.add_argument("--candidate-output-key", default=None)
    parser.add_argument("--reference-output-index", type=int, default=None)
    parser.add_argument("--reference-output-key", default=None)
    parser.add_argument("--directional-margin", type=float, default=0.0)
    parser.add_argument("--aligned-projection-margin", type=float, default=0.0,
                        help="Optional threshold in output/gradient units; default tests zero")
    parser.add_argument("--check-backward", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    if not _inside_repo(output, root):
        parser.error("--output must be inside the kernel-analyzer repository")
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    candidate = _load_callable(args.module, args.candidate)
    reference = _load_callable(args.module, args.reference)
    make_inputs = _load_callable(args.module, args.make_inputs)
    report = check_tilelang_bias(
        candidate,
        reference,
        make_inputs,
        candidate_output_index=args.candidate_output_index,
        candidate_output_key=args.candidate_output_key,
        reference_output_index=args.reference_output_index,
        reference_output_key=args.reference_output_key,
        samples=args.samples,
        calibration_samples=args.calibration_samples,
        directional_margin=args.directional_margin,
        aligned_projection_margin=args.aligned_projection_margin,
        check_backward=args.check_backward,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "measurement_status": report.get("measurement_status"),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
