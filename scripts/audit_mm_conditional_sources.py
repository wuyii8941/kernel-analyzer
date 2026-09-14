"""Recompute source-isolation checks from saved conditional MM records, not labels."""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "qwen128_vproj_conditional_debias.json.gz",
    "qwen64_vproj_conditional_debias_r16.json.gz",
    "mamba_seq64_input_proj_conditional_debias.json.gz",
)


def review(document):
    modes = {}
    for state in document["states"]:
        for mode, data in state.get("arms", {}).items():
            if mode not in {"ROUNDING_ONLY", "JOINT"}:
                continue
            rows = modes.setdefault(mode, [])
            for index, summary in enumerate(data.get("repeat_local_summaries", [])):
                error = summary.get("kernel_residual_preservation_error")
                rows.append({"state_id": state.get("state_id"), "repeat": index,
                             "preservation_error": error})
    checks = {}
    for mode, rows in modes.items():
        complete = bool(rows) and all(r["preservation_error"] is not None for r in rows)
        zero = complete and all(all(r["preservation_error"].get(k) == 0
                                   for k in ("l2", "max_abs", "nonzero")) for r in rows)
        checks[mode] = {
            "repeat_count": len(rows), "records": rows,
            "all_recorded_preservation_errors_zero": zero,
            "interpretation": (
                "PRESERVATION_NOT_REQUIRED_JOINT_REMOVES_KERNEL_RESIDUAL"
                if mode == "JOINT" else
                "PRESERVED_IN_ALL_RECORDED_REPEATS" if zero else
                "SOURCE_ISOLATION_NOT_VERIFIED"),
        }
    return {
        "case_id": document["candidate_id"],
        "parameter": document["carrier_parameter"],
        "bindings": document.get("bindings", {}), "checks": checks,
        "historical_conditional_summary": document.get("conditional_debias_summary"),
        "scope": "Saved-record audit; not tensor replay, new inference, or cross-state mean proof",
        "unresolved": ["Historical executed source and binary identity require separate verification",
                       "Conditional ensemble effect does not establish a common cross-state bias",
                       "Absolute downstream repair bias lacks exact downstream reference"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("output must be a new repository file")
    cases = []
    for name in INPUTS:
        path = ROOT / "results/coverage/cases" / name
        raw = path.read_bytes()
        case = review(json.loads(gzip.decompress(raw)))
        case.update(source=str(path.relative_to(ROOT)))
        cases.append(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump({"schema": "mm-conditional-source-audit-v1", "cases": cases}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for case in cases:
        print(case["case_id"], {k: (v["repeat_count"], v["interpretation"]) for k,v in case["checks"].items()})


if __name__ == "__main__":
    main()
