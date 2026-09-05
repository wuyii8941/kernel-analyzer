#!/usr/bin/env python3
"""Single entry point for the current training numerical analysis pipeline.

This orchestrates existing capture-independent analysis programs.  GPU capture
remains family-specific because execution boundaries differ; every captured
artifact is normalized by the same recompute and reporting path here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("refresh")
    recompute = subparsers.add_parser("recompute")
    recompute.add_argument("raw", type=Path)
    recompute.add_argument("output", type=Path)
    recompute.add_argument("--endpoint", choices=("PARAMETER_WRITE", "ADAMW_UPDATE"))
    recompute.add_argument("--corrected-sketch-with-legacy-name", action="store_true")
    args = parser.parse_args()

    if args.command == "validate":
        _run("validate_training_numerical_analysis.py")
        return
    if args.command == "recompute":
        forwarded = [str(args.raw), str(args.output)]
        if args.endpoint:
            forwarded.extend(["--endpoint", args.endpoint])
        if args.corrected_sketch_with_legacy_name:
            forwarded.append("--corrected-sketch-with-legacy-name")
        _run("recompute_training_numerical_report.py", *forwarded)
        return
    _run("build_mainline_case_roles.py")
    _run("audit_training_measurement_provenance.py")
    _run("validate_training_numerical_analysis.py")
    _run("build_training_numerical_analysis_report.py")
    _run("audit_training_numerical_analysis_completion.py")
    _run("check_research_docs.py")


if __name__ == "__main__":
    main()
