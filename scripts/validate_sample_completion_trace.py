#!/usr/bin/env python3
"""Validate one sample-completion trace and emit its safe predictor view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_analyzer.sample_completion import predictor_view, validate_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--predictor-output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    validation = validate_trace(payload)
    result = {
        "schema": "kernel-analyzer-sample-completion-trace-validation-v1",
        "status": validation.status,
        "case_id": validation.case_id,
        "steps": validation.steps,
        "reasons": list(validation.reasons),
        "final_label_allowed": validation.valid,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.predictor_output is not None:
        args.predictor_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictor_output.write_text(
            json.dumps(predictor_view(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not validation.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
