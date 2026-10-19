"""Console entry points of the installed package (``pyproject.toml`` [project.scripts]); the repository scripts
``2_tool/scripts/measure.py`` and ``run_reference_analysis.py`` call the same functions.

    kernel-analyzer-measure --declaration D.json --out R.json
    kernel-analyzer-analyze --declaration D.json --stage reference|statistics --out DIR [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _comparison_summary(result, comparison, reason=None):
    """Display an existing comparison without changing its statistical judgment."""
    if result is None:
        return {"comparison": comparison, "status": "not established", "reason": reason}
    statistics = result.get("statistics") or {}
    summary = {"comparison": result.get("comparison") or comparison, "status": "evaluated",
               "fixed_mean": (statistics.get("fixed_mean") or {}).get("summary"),
               "aligned": (statistics.get("aligned") or {}).get("summary")}
    for name in ("verdict_scope", "covers_complete_public_output", "elements_per_seed"):
        if name in result:
            summary[name] = result[name]
    if result.get("scale") is not None:
        summary["mean_abs_residual"] = result["scale"].get("mean_abs_residual")
    return summary


def _measurement_output_summary(output, *, black_box=False):
    quality = output.get("reference") or {}
    no_program_reference = black_box or not quality or quality.get("elements") == 0
    scope = ("not established: no program-reference coordinates" if no_program_reference
             else quality.get("reference_scope"))
    semantic = output.get("semantic")
    semantic_label = ("G_pi - f" if (semantic or {}).get("verdict_scope") ==
                      "G_pi_minus_admitted_task_specification" else "program reference - supplied reference")
    total = output if black_box and output.get("statistics") is not None else output.get("total")
    # Keep the old fields for existing consumers; the explicit comparisons
    # distinguish the top-level program statistics from the full task result.
    return {"status": output.get("status"), "complete_rate": quality.get("complete_rate"),
            "resolved": quality.get("resolved_fraction"), "reference_scope": scope,
            "fixed_mean": ((output.get("statistics") or {}).get("fixed_mean") or {}).get("summary"),
            "aligned": ((output.get("statistics") or {}).get("aligned") or {}).get("summary"),
            "failures": output.get("failure_classes") or output.get("failure_class"),
            "task_special_value_verdict": output.get("task_special_value_verdict"),
            "boolean_task_comparison": output.get("boolean_task_comparison"),
            "comparisons": {
                "numerical": _comparison_summary(None if no_program_reference else output,
                    "K - program reference", "no program reference established for this output"),
                "semantic": _comparison_summary(semantic, semantic_label,
                    "no program-to-task/reference comparison established"),
                "total": _comparison_summary(total, "K - supplied reference",
                    "no candidate-to-task/reference comparison established")}}


def measure_main(argv=None) -> int:
    """Unified entry (``kernel_analyzer.measure.run``): exit 0 with a report, 2 when the declaration is incomplete."""
    from . import measure

    ap = argparse.ArgumentParser(prog="kernel-analyzer-measure")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        rep = measure.run(a.declaration, a.out)
    except measure.MissingDeclaration as exc:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"status": "declaration incomplete", "missing": exc.items}, indent=1) + "\n")
        print("declaration incomplete:", *exc.items, sep="\n  ")
        return 2
    except measure.ReferenceNotAdmitted as exc:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"status": "reference not admitted", "reason": str(exc)}, indent=1) + "\n")
        print("reference not admitted:", exc)
        return 2
    for lv in rep["levels"]:
        print(json.dumps({"level": lv["level"], "status": lv["status"], "seconds": lv.get("seconds"),
                          "refinement": (lv.get("refinement") or {}).get("outcome"),
                          "outputs": {k: _measurement_output_summary(v, black_box=lv.get("mode") == "black-box")
                                      for k, v in (lv.get("outputs") or {}).items()},
                          "reason": lv.get("reason")}, default=str))
    print("failure counts", rep["failure_counts"])
    return 0


def analysis_main(argv=None) -> int:
    """Captured-package analysis (``kernel_analyzer.reference_eval.analysis``): reference or statistics stage."""
    from .reference_eval.analysis import load_declaration, reference_stage, statistics_stage

    parser = argparse.ArgumentParser(prog="kernel-analyzer-analyze")
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--stage", choices=("reference", "statistics"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory of per-unit reference arrays")
    parser.add_argument("--capture-root", type=Path, help="override the declaration's capture_root")
    parser.add_argument("--units", help="comma-separated unit directory names (reference stage)")
    parser.add_argument("--delete-captures", action="store_true")
    parser.add_argument("--report", type=Path, help="statistics report path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    decl = load_declaration(args.declaration)
    if args.capture_root:
        decl["capture_root"] = str(args.capture_root)
    if args.stage == "reference":
        units = args.units.split(",") if args.units else None
        for summary in reference_stage(decl, args.out, units, args.delete_captures):
            print(json.dumps({"unit": summary["unit"], **{v: {k: summary[v][k] for k in (
                "classes", "residual_positive", "residual_negative", "seconds")} for v in decl["variants"]}}),
                flush=True)
    else:
        if args.report is None:
            parser.error("--report is required for the statistics stage")
        report = statistics_stage(decl, args.out, args.device, arrays_out=args.out / "statistics_arrays.npz")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        for r in report["results"]:
            print(json.dumps({k: r.get(k) for k in ("comparison", "rule", "mean_projection", "t_interval",
                                                      "positive", "negative", "final_verdict")}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(measure_main())
