#!/usr/bin/env python3
"""Single entry point for the current training numerical analysis pipeline.

This orchestrates existing capture-independent analysis programs.  GPU capture
remains family-specific because execution boundaries differ; every captured
artifact is normalized by the same recompute and reporting path here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> None:
    # Keep the single entry point usable when invoked as
    # ``python scripts/run_training_numerical_analysis.py``.  In that form
    # Python puts ``scripts/`` (not the repository root) on sys.path, while
    # the delegated programs import ``kernel_analyzer`` and sometimes
    # ``scripts`` as packages.  Do not rely on a caller-specific PYTHONPATH.
    source_path = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    inherited = os.environ.get("PYTHONPATH")
    env = dict(os.environ)
    env["PYTHONPATH"] = source_path if not inherited else source_path + os.pathsep + inherited
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    family_commands = {'row-reference': 'build_row_reduction_contracts.py',
                       'row-capture': 'run_row_reduction_capture.py',
                       'family-report': 'finalize_numerical_family.py',
                       'select-training': 'select_training_validation_candidates.py',
                       'analyze': 'recompute_training_numerical_v2.py'}
    if len(sys.argv) > 1 and sys.argv[1] in family_commands:
        _run(family_commands[sys.argv[1]], *sys.argv[2:])
        return
    # New metadata-driven execution path; retain legacy commands for replay.
    if len(sys.argv) > 1 and sys.argv[1] == "coverage":
        _run("run_numerical_coverage.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "population-exceedance":
        _run("recompute_population_exceedance.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "population-bounded-mean":
        _run("recompute_bounded_population_equivalence.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "catalog-observed-kernels":
        _run("build_observed_kernel_catalog.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "queue-by-family":
        _run("build_family_first_execution_queue.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build-family-campaigns":
        _run("build_family_first_campaign_manifest.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "run-family-campaigns":
        _run("run_family_first_campaigns.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "summarize-family-campaigns":
        _run("summarize_family_first_campaigns.py", *sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "adamw8bit-population":
        _run("run_adamw8bit_population_update.py", *sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        epilog="Metadata-driven capture: coverage {freeze,run,report} --help; "
               "population prevalence: population-exceedance RAW PROTOCOL OUTPUT; "
               "bounded population mean: population-bounded-mean RAW PROTOCOL OUTPUT; "
               "all observed kernels: catalog-observed-kernels ...; "
               "family-first execution queue: queue-by-family ...; "
               "family campaign freeze/run/report: build-family-campaigns / run-family-campaigns; "
               "bounded family summary: summarize-family-campaigns; "
               "real iid optimizer-state study: adamw8bit-population {freeze,run}; "
               "source-checked families: row-reference / row-capture / family-report. "
               "Training review shortlist: select-training. "
               "Current shared analysis: analyze RAW OUTPUT --protocol PROTOCOL. "
               "The commands listed above retain the historical analysis path.")
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
