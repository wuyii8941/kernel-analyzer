#!/usr/bin/env python3
"""Bind new TCMP case plans from the frozen metadata pool.

This helper is deliberately metadata-only.  It never reads numerical values,
formation labels, trajectory labels, or historical verdicts.  A plan can be
used for engineering reach only after the runner validates the frozen release
and exact F+B endpoint at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "results/property/tcmp_allop_v1/semantic_family_heldout_pool_v1.json"
REGISTRY = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_carriers.json"
OUT = ROOT / "results/property/tcmp_allop_v1/heldout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", required=True,
                        help="model:sequence_length:task_id")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()

    pool = json.loads(POOL.read_text())
    rows = [row for row in pool["rows"]
            if row.get("pool_status") == "PRE_MEASUREMENT_CANDIDATE"]
    registry = json.loads(REGISTRY.read_text())["cells"]
    registry_by_key = {
        (cell["representative"].get("model"),
         int(cell.get("sequence_length", cell["representative"].get("sequence_length"))),
         cell["representative"].get("task_id")): cell
        for cell in registry
        if cell.get("representative")
    }
    for spec in args.task:
        model, seq_text, task_id = spec.split(":", 2)
        seq = int(seq_text)
        matches = [row for row in rows if row.get("model") == model
                   and int(row.get("sequence_length")) == seq
                   and row.get("task_id") == task_id]
        if len(matches) != 1:
            raise SystemExit(f"expected one frozen-pool match for {spec}, got {len(matches)}")
        row = matches[0]
        cell = registry_by_key.get((model, seq, task_id))
        if cell is None:
            raise SystemExit(f"carrier registry is missing {spec}")
        rep = cell["representative"]
        if not bool(rep.get("exact_endpoint_executable")):
            raise SystemExit(
                f"frozen pool representative is not exact-runtime executable: {spec}"
            )
        carrier = cell.get("nearest_carrier")
        if not carrier or not carrier.get("name"):
            raise SystemExit(f"nearest carrier is missing for {spec}")
        case_id = f"{model}_seq{seq}_{task_id.replace(':', '_')}"
        target = args.output_root / case_id
        target.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "kernel-analyzer-tcmp-case-plan-v2",
            "case_id": case_id,
            "model": model,
            "sequence_length": seq,
            "task_id": task_id,
            "source_family": row["family"],
            "implementation_kind": row["implementation_kind"],
            "region_symbol": row["region_symbol"],
            "source_internal_region": task_id,
            "closure_task": task_id,
            "complete_fb_unit_required": True,
            "metadata_binding": {
                "pool_cell_id": row["cell_id"],
                "pool_rank": row["frozen_pool_rank"],
                "member_count": row["member_count"],
                "exact_aot_endpoint_id": rep.get("exact_aot_endpoint_id"),
                "nearest_carrier": carrier,
                "semantic_region_executable": bool(row["semantic_region_executable"]),
                "has_exact_downstream_closure": bool(row["has_exact_downstream_closure"]),
            },
            "cases": [{
                "case_id": case_id,
                "task_id": task_id,
                "carrier": carrier["name"],
                "reference_method": "AOT_REPLAY",
                "claim_boundary": (
                    f"Exact {row['family']} endpoint {task_id}; the complete "
                    "F+B boundary and declared parameter reach must be validated "
                    "by the runner before any formation verdict."
                ),
            }],
        }
        (target / "case_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"case_id": case_id, "output": str(target / "case_plan.json"),
                          "carrier": carrier["name"], "family": row["family"]}))


if __name__ == "__main__":
    main()
