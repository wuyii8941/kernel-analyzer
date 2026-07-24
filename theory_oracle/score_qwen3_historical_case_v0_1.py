#!/usr/bin/env python
"""Post-reveal score for the generic Qwen3 blind locator."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-report", required=True, type=Path)
    parser.add_argument("--buggy-source-root", required=True, type=Path)
    parser.add_argument("--fixed-source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    blind = json.loads(args.blind_report.read_text())
    buggy = (args.buggy_source_root / "torchtitan/models/qwen3/model/model.py").read_text()
    fixed = (args.fixed_source_root / "torchtitan/models/qwen3/model/model.py").read_text()
    diff = list(difflib.unified_diff(buggy.splitlines(), fixed.splitlines(), lineterm=""))
    changed_lines = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    first_trace = blind.get("trace_alignment", {}).get("first_mismatch") or {}
    changed_region = blind.get("first_changed_region")
    operation_matches_patch = first_trace.get("candidate") == "transpose" and any("transpose" in line for line in changed_lines)
    result = {
        "schema_version": "forkcert.qwen3-historical-case-score.v0.1",
        "case_id": blind["case_id"],
        "blind_observation": {
            "endpoint_exact": blind["endpoint_oracle"]["exact_match"],
            "first_changed_region": changed_region,
            "first_trace_mismatch": first_trace,
            "repeatable": blind["repeatability"]["exact_all_repeats"],
        },
        "revealed_patch": {
            "changed_source_path": "torchtitan/models/qwen3/model/model.py",
            "changed_line_count": len(changed_lines),
            "diff_sha256": __import__("hashlib").sha256("\n".join(diff).encode()).hexdigest(),
            "contains_transpose_edit": any("transpose" in line for line in changed_lines),
        },
        "scoring": {
            "patch_path_covered": changed_region in {"wo", "__root__"},
            "operation_interval_covers_patch_mechanism": operation_matches_patch,
            "claim_level": "HISTORICAL_PATCH_COVERED_OPERATION_INTERVAL" if operation_matches_patch else "REGION_ONLY",
        },
        "limitations": [
            "post-reveal patch coverage is not a proof of unique causality",
            "the fixed reference artifact remains an implementation-relative oracle",
            "the generic locator did not identify a compiler pass or generated kernel",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
