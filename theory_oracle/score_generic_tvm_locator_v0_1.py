"""Post-reveal, bug-agnostic comparison of two generic locator reports."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def _symbols(text: str, marker: str) -> set[str]:
    if marker == "R":
        return set(re.findall(r"R\.([A-Za-z_]\w*)", text))
    return set(re.findall(r"def\s+([A-Za-z_]\w*)", text))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buggy", required=True)
    p.add_argument("--fixed", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    buggy = json.loads(Path(args.buggy).read_text())
    fixed = json.loads(Path(args.fixed).read_text())
    stages = ["frontend_relax", "legalized_tir"]
    changed = [stage for stage in stages if buggy["ir_stages"][stage]["sha256"] != fixed["ir_stages"][stage]["sha256"]]
    first_changed = changed[0] if changed else None
    old_frontend = buggy["ir_stages"]["frontend_relax"]["text"]
    new_frontend = fixed["ir_stages"]["frontend_relax"]["text"]
    old_ops, new_ops = _symbols(old_frontend, "R"), _symbols(new_frontend, "R")
    old_tir = set(buggy["region_inventory"]["legalized_tir_functions"])
    new_tir = set(fixed["region_inventory"]["legalized_tir_functions"])
    diff = list(difflib.unified_diff(old_frontend.splitlines(), new_frontend.splitlines(), lineterm=""))
    result = {
        "schema_version": "forkcert.generic-tvm-post-reveal-score.v0.1",
        "case_id": buggy["case_identity"]["case_id"],
        "locator_inputs_were_bug_agnostic": bool(not buggy["automation_contract"]["bug_specific_region_input"] and not buggy["automation_contract"]["bug_specific_repair_input"]),
        "endpoint": {
            "buggy_exact": buggy["oracle"]["endpoint"]["exact_match"],
            "fixed_exact": fixed["oracle"]["endpoint"]["exact_match"],
        },
        "automatic_stage_comparison": {
            "changed_stages": changed,
            "first_changed_stage": first_changed,
            "method": "earliest differing generic IR snapshot",
        },
        "automatic_ir_region_comparison": {
            "buggy_reachable_regions": buggy["region_inventory"]["candidate_regions"],
            "fixed_reachable_symbols": sorted(new_tir),
            "frontend_new_generic_ops": sorted(new_ops - old_ops),
            "frontend_removed_generic_ops": sorted(old_ops - new_ops),
            "frontend_changed_region": "Relax function main" if old_frontend != new_frontend else None,
            "tir_new_symbols": sorted(new_tir - old_tir),
        },
        "diff_excerpt": diff[:120],
        "evaluation": {
            "stage_contains_external_fix": first_changed == "frontend_relax",
            "region_contains_external_fix": first_changed == "frontend_relax" and old_frontend != new_frontend,
            "unique_kernel_localization": False,
            "claim": "STAGE_AND_IR_REGION_CANDIDATE_ONLY",
        },
        "limitations": [
            "fixed report is used only after pre-reveal certificate freeze",
            "artifact comparison is not proof of a unique causal source",
            "no same-context repair or fixed-suffix mediation is inferred",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "first_changed_stage": first_changed, "claim": result["evaluation"]["claim"]}, sort_keys=True))


if __name__ == "__main__":
    main()
