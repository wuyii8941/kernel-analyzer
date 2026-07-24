#!/usr/bin/env python3
"""Partial matched-state influence profiles for all declared scalar endpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    REQUIRED_PHASES,
    sha256_file,
)
from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    collect_endpoint,
    endpoint_extractors,
    load_complete_state_bundles,
)
from theory_oracle.analyze_qwen3_partial_boundary_influence_v0_1 import (
    influence_profile,
)
from theory_oracle.bias_oracle_population_v0_2 import EffectRecord


SCHEMA_VERSION = "forkcert.qwen3-partial-core-endpoint-influence.v0.1"
SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = "forkcert.multi-transition-capture-plan.v0.1"


def sample_variance(values: list[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sample variance received a nonfinite effect")
    if len(values) < 2:
        return 0.0
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def summarize_records(records: list[EffectRecord]) -> dict[str, Any]:
    by_state: dict[tuple[str, str], list[EffectRecord]] = defaultdict(list)
    for record in records:
        by_state[(record.phase, record.state_id)].append(record)
    state_rows = []
    errors = []
    for (phase, state_id), current in sorted(by_state.items()):
        repeat_ids = {record.repeat_id for record in current}
        if repeat_ids != {1, 2} or len(current) != 2:
            errors.append(f"{state_id}: incomplete repeat grid")
            continue
        effects = [record.effect for record in sorted(current, key=lambda row: row.repeat_id)]
        if any(not math.isfinite(effect) for effect in effects):
            errors.append(f"{state_id}: nonfinite paired effect")
            continue
        mean = sum(effects) / len(effects)
        state_rows.append(
            {
                "phase": phase,
                "state_id": state_id,
                "repeat_effects": effects,
                "state_effect_mean": mean,
                "same_state_paired_effect_variance": sample_variance(effects),
            }
        )
    sign_counts = {
        "positive": sum(row["state_effect_mean"] > 0 for row in state_rows),
        "zero": sum(row["state_effect_mean"] == 0 for row in state_rows),
        "negative": sum(row["state_effect_mean"] < 0 for row in state_rows),
    }
    return {
        "valid": not errors,
        "errors": errors,
        "states": len(state_rows),
        "state_effect_sign_counts": sign_counts,
        "states_with_observed_nonzero_runtime_variance": sum(
            row["same_state_paired_effect_variance"] > 0 for row in state_rows
        ),
        "phase_profiles": {
            phase: influence_profile(
                [
                    (row["state_id"], float(row["state_effect_mean"]))
                    for row in state_rows
                    if row["phase"] == phase
                ]
            )
            for phase in REQUIRED_PHASES
        },
        "state_rows": state_rows,
        "population_B_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    results_root = Path(args.results_root).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    targets = plan.get("targets", [])
    if plan.get("schema_version") != PLAN_SCHEMA or not isinstance(targets, list) or len(targets) != 24:
        errors.append("partial core diagnostic requires the frozen 24-state plan")
        targets = targets if isinstance(targets, list) else []
    present = []
    missing = []
    for target in targets:
        step_dir = results_root / f"step{int(target['optimizer_step']):03d}"
        bundle = (step_dir / "record_bundle.json").is_file()
        validation = (step_dir / "record_validation.json").is_file()
        if bundle and validation:
            present.append(target)
        elif bundle or validation:
            errors.append(f"partial evidence for {target.get('state_id')}")
        else:
            missing.append(target.get("state_id"))
    bundles, evidence, load_errors = load_complete_state_bundles(
        {**plan, "targets": present}, results_root
    )
    errors.extend(load_errors)
    endpoint_results = {}
    for name, (endpoint_class, extractor) in endpoint_extractors().items():
        records, availability, unavailable = collect_endpoint(bundles, extractor)
        profile = summarize_records(records) if records else None
        endpoint_results[name] = {
            "endpoint_class": endpoint_class,
            "availability": dict(availability),
            "unavailable_records": unavailable,
            "profile": profile,
            "signed_B_family_eligible": endpoint_class.startswith("SIGNED_"),
            "population_B_claim_allowed": False,
        }
        if profile is not None:
            errors.extend(f"{name}: {error}" for error in profile["errors"])
    valid = not errors
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": (
            "VALID_PARTIAL_CORE_ENDPOINT_DESCRIPTION"
            if valid
            else "INVALID_PARTIAL_CORE_ENDPOINT_DESCRIPTION"
        ),
        "construction": {
            "observed_complete_states": len(bundles),
            "expected_states": len(targets),
            "missing_state_ids": missing,
            "phases_pooled": False,
            "population_inference_allowed": False,
        },
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "results_root": str(results_root),
        "state_evidence": evidence,
        "endpoints": endpoint_results,
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "errors": errors,
        "nonclaims": [
            "partial state-effect means are not population B",
            "leave-one-state-out influence is not independent-trajectory inference",
            "zero observed repeat variance at R=2 is not proof that runtime N is zero",
            "nonnegative magnitude/disagreement endpoints are not Bias",
            "missing states and phases are retained in construction, not silently deleted",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "construction": result["construction"], "errors": errors}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
