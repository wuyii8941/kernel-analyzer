#!/usr/bin/env python3
"""Analyze the frozen optimizer/checkpoint comparison with one correction family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.training_bias_profile import BRANCHES, holm_adjusted_p  # noqa: E402
from scripts.analyze_training_bias_profile_v2_empirical import _branch_summary, _primary_view  # noqa: E402


COLD_PATHS = {
    "deepseek8b_seq256_backward_1714_in_out_ptr0": ROOT / "results/property/training_bias_profile_v2/prospective_batch_1/raw/deepseek8b_seq256_backward_1714_in_out_ptr0.json",
    "deepseek8b_seq128_backward_1256_out_ptr0": ROOT / "results/property/training_bias_profile_v2/prospective_batch_2/raw/deepseek8b_seq128_backward_1256_out_ptr0.json",
    "phi4_seq64_backward_495_out_ptr1": ROOT / "results/property/training_bias_profile_v2/prospective_batch_1/raw/phi4_seq64_backward_495_out_ptr1.json",
}
SECONDARY_STAGES = (
    "ADAMW_MOMENT1_WRITE",
    "ADAMW_MOMENT2_WRITE",
    "NEXT_STEP_COMMON_GRADIENT_UPDATE",
)
SECONDARY_CONDITIONS = {
    "warm_step_8", "warm_step_32", "warm_step_32_moments_reset",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    conditions = [row["name"] for row in protocol["conditions"]]
    payloads = {}
    raw_p = {}
    secondary_raw_p = {}
    for case in protocol["cases"]:
        case_id = case["case_id"]
        for condition in conditions:
            if condition == "cold_start":
                path = COLD_PATHS[case_id]
                stage = "ADAMW_UPDATE"
            elif condition == "stateless_sgd_at_warm_step_32":
                path = args.raw_root / "warm_step_32" / f"{case_id}.json"
                stage = "PARAMETER_GRADIENT"
            else:
                path = args.raw_root / condition / f"{case_id}.json"
                stage = "ADAMW_UPDATE"
            key = (case_id, condition)
            if not path.exists():
                payloads[key] = None
                for branch in BRANCHES:
                    raw_p[f"{case_id}|{condition}|{branch}"] = 1.0
                if condition in SECONDARY_CONDITIONS:
                    for secondary_stage in SECONDARY_STAGES:
                        for branch in BRANCHES:
                            secondary_raw_p[
                                f"{case_id}|{condition}|{secondary_stage}|{branch}"
                            ] = 1.0
                continue
            item = json.loads(path.read_text())
            if item.get("status") != "COMPLETE":
                payloads[key] = None
                for branch in BRANCHES:
                    raw_p[f"{case_id}|{condition}|{branch}"] = 1.0
                if condition in SECONDARY_CONDITIONS:
                    for secondary_stage in SECONDARY_STAGES:
                        for branch in BRANCHES:
                            secondary_raw_p[
                                f"{case_id}|{condition}|{secondary_stage}|{branch}"
                            ] = 1.0
                continue
            payloads[key] = (item, stage, str(path))
            views = item["stages"][stage]
            profile = views[_primary_view(views)]["profile"]
            for branch in BRANCHES:
                raw_p[f"{case_id}|{condition}|{branch}"] = (
                    float(profile["population_inference"]["branches"][branch]["raw_studentized_signflip_p"])
                    if profile.get("status") == "POPULATION_INFERENCE_COMPLETE" else 1.0
                )
            if condition in SECONDARY_CONDITIONS:
                for secondary_stage in SECONDARY_STAGES:
                    secondary_views = item.get("stages", {}).get(secondary_stage)
                    for branch in BRANCHES:
                        secondary_key = f"{case_id}|{condition}|{secondary_stage}|{branch}"
                        if not secondary_views:
                            secondary_raw_p[secondary_key] = 1.0
                            continue
                        secondary_profile = secondary_views[_primary_view(secondary_views)]["profile"]
                        secondary_raw_p[secondary_key] = (
                            float(secondary_profile["population_inference"]["branches"][branch]["raw_studentized_signflip_p"])
                            if secondary_profile.get("status") == "POPULATION_INFERENCE_COMPLETE" else 1.0
                        )
    adjusted = holm_adjusted_p(raw_p)
    secondary_adjusted = holm_adjusted_p(secondary_raw_p) if secondary_raw_p else {}
    cases = {}
    for case in protocol["cases"]:
        case_id = case["case_id"]
        case_rows = {}
        for condition in conditions:
            stored = payloads[(case_id, condition)]
            if stored is None:
                case_rows[condition] = {"status": "ABSTAIN_MEASUREMENT_MISSING"}
                continue
            item, stage, path = stored
            views = item["stages"][stage]
            summaries = {
                branch: _branch_summary(
                    views, branch, adjusted[f"{case_id}|{condition}|{branch}"]
                ) for branch in BRANCHES
            }
            case_rows[condition] = {
                "status": "COMPLETE",
                "source_stage": stage,
                "source_artifact": path,
                "branches": summaries,
                "confirmed_effects": [name for name, value in summaries.items() if value["status"] == "CONFIRMED"],
            }
            if condition in SECONDARY_CONDITIONS:
                secondary = {}
                for secondary_stage in SECONDARY_STAGES:
                    secondary_views = item.get("stages", {}).get(secondary_stage)
                    if not secondary_views:
                        secondary[secondary_stage] = {"status": "ABSTAIN_MEASUREMENT_MISSING"}
                        continue
                    branch_rows = {
                        branch: _branch_summary(
                            secondary_views,
                            branch,
                            secondary_adjusted[f"{case_id}|{condition}|{secondary_stage}|{branch}"],
                        )
                        for branch in BRANCHES
                    }
                    secondary[secondary_stage] = {
                        "status": "COMPLETE",
                        "branches": branch_rows,
                        "confirmed_effects": [
                            name for name, value in branch_rows.items()
                            if value["status"] == "CONFIRMED"
                        ],
                    }
                case_rows[condition]["optimizer_state_writes"] = secondary
        cases[case_id] = {"role": case["role"], "conditions": case_rows}
    result = {
        "schema": "kernel-analyzer-optimizer-condition-benchmark-summary-v1",
        "status": "COMPLETE" if all(
            row["status"] == "COMPLETE"
            for case in cases.values() for row in case["conditions"].values()
        ) else "COMPLETE_WITH_ABSTENTIONS",
        "protocol": str(args.protocol),
        "multiplicity": {
            "primary": {"method": "Holm family-wise correction", "test_count": len(raw_p)},
            "optimizer_state_secondary": {
                "method": "separate Holm family-wise correction",
                "test_count": len(secondary_raw_p),
            },
        },
        "cases": cases,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "tests": len(raw_p)}))


if __name__ == "__main__":
    main()
