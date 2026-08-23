#!/usr/bin/env python3
"""Build one honest denominator table for model/operator coverage.

The table distinguishes systematic census cells from directed/held-out work.
Missing downstream measurements remain null; they are never imputed from T1 or
from a historical case label.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/coverage/coverage_table_v1.json")
    args = parser.parse_args()

    model_audit = load(ROOT / "results/coverage/model_coverage_audit_v1.json")
    cells = load(ROOT / "results/coverage/four_model_full_operator_audit.json")["cells"]
    coordinate = load(ROOT / "results/coverage/cases/full_coordinate_audit.json.gz")["audited_rows"]
    per_cell: dict[tuple[str, int], Counter[str]] = {}
    for row in coordinate:
        key = (str(row["model"]), int(row["sequence_length"]))
        per_cell.setdefault(key, Counter())["audited"] += 1
        per_cell[key]["t1_pass"] += int(bool(row["t1_pass"]))
        per_cell[key]["t1_reject"] += int(not bool(row["t1_pass"]))

    coordinate_model_alias = {
        "qwen3_1p7b": "qwen",
        "mamba_130m": "mamba",
        "phi4_mini_3p8b": "phi4",
        "deepseek_r1_0528_qwen3_8b": "deepseek8b",
    }
    core_rows = []
    for row in cells:
        model = str(row["model"])
        seq = int(str(row["shape"]).split("seq", 1)[1])
        counts = per_cell.get((coordinate_model_alias[model], seq), Counter())
        core_rows.append({
            "model": model,
            "sequence_length": seq,
            "scope": "SYSTEMATIC_CENSUS",
            "eager_invocations": row["eager_invocations"],
            "candidate_invocations": row["candidate_compute_invocations"],
            "primary_fb_proof_units": row["primary_fb_proof_units"],
            "t1_audited": counts.get("audited", 0),
            "t1_pass": counts.get("t1_pass", 0),
            "t1_reject": counts.get("t1_reject", 0),
            "downstream_uniform_32_step_cases": None,
            "valid_nonzero_parameter_reachable_controls": None,
            "deep_tested_cases": None,
            "status": row["status"],
            "open_items": (["Mamba fresh timing and one seq256 T2 shard remain open"]
                            if model == "mamba_130m" else []),
        })

    directed_rows = []
    for row in model_audit["directed_or_heldout"]:
        directed_rows.append({
            "model": row["model"],
            "scope": "DIRECTED_OR_HELDOUT",
            "sequence_lengths": row.get("sequence_lengths"),
            "eager_invocations": None,
            "candidate_invocations": None,
            "primary_fb_proof_units": None,
            "t1_audited": None,
            "t1_pass": None,
            "t1_reject": None,
            "downstream_uniform_32_step_cases": None,
            "valid_nonzero_parameter_reachable_controls": None,
            "deep_tested_cases": None,
            "scope_description": row["scope"],
            "status": row["status"],
        })

    result = {
        "schema": "kernel-analyzer-model-operator-coverage-table-v1",
        "status": "COMPLETE_FOR_DECLARED_COVERAGE_LAYERS_NOT_COMPLETE_FOR_UNIFORM_SAMPLE",
        "counting_rule": "A model with a targeted artifact is not treated as a full census model.",
        "model_count_with_actual_artifacts": model_audit["distinct_model_count_with_actual_evidence"],
        "systematic_census_model_count": model_audit["systematic_census_model_count"],
        "systematic_census_cell_count": len(core_rows),
        "core_rows": sorted(core_rows, key=lambda r: (r["model"], r["sequence_length"])),
        "directed_or_heldout_rows": directed_rows,
        "totals": {
            "eager_invocations": sum(r["eager_invocations"] for r in core_rows),
            "candidate_invocations": sum(r["candidate_invocations"] for r in core_rows),
            "primary_fb_proof_units": sum(r["primary_fb_proof_units"] for r in core_rows),
            "t1_audited": sum(r["t1_audited"] for r in core_rows),
            "t1_pass": sum(r["t1_pass"] for r in core_rows),
            "t1_reject": sum(r["t1_reject"] for r in core_rows),
        },
        "uniform_sample_completion": {
            "required_cases": 20,
            "required_controls": 15,
            "current_uniform_cases": 0,
            "current_uniform_controls": 0,
            "status": "NOT_STARTED",
        },
        "claim_boundary": "This table proves the four-model census denominator and reports directed coverage separately. It does not promote targeted rows to full coverage or to scientific cases.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": result["model_count_with_actual_artifacts"], "core_cells": len(core_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
