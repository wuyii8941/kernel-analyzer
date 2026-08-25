#!/usr/bin/env python3
"""Finalize an already-observed paired-loss split without rerunning a model.

The input checkpoint was produced by the four-counterfactual consequence
runner.  If any recorded candidate/repair loss gap exceeds the frozen 1e-8
threshold, the scientific consequence question is already answered.  This
tool verifies that terminal condition, recomputes all summaries from the raw
checkpoint rows, and writes a consequence-only artifact.  It deliberately
does not assign a persistence label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.persistence_property import (  # noqa: E402
    aligned_level_statistics_from_gram,
    path_statistics_from_gram,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--architecture", required=True)
    value.add_argument("--model", type=Path, required=True)
    value.add_argument("--release-dir", type=Path, required=True)
    value.add_argument("--case-plan", type=Path, required=True)
    value.add_argument("--case-id", required=True)
    value.add_argument("--planned-horizon", type=int, default=4096)
    value.add_argument("--learning-rate", type=float, default=1e-4)
    return value


def statistics(rows: list[dict[str, Any]], compact: bool) -> dict[str, Any]:
    state_ids = [str(row["step_id"]) for row in rows]
    if len(rows) < 2:
        return {
            "schema": "kernel-analyzer-consequence-only-statistics-v1",
            "state_ids": state_ids,
            "level_ids": [],
            "levels": {},
            "resultant_cosines": {},
            "not_computed_reason": (
                "The paired-loss terminal condition was reached before the "
                "minimum two-row Gram audit. No persistence statistic is "
                "reported or inferred."
            ),
        }
    arrays = {
        name: np.asarray([row[f"derived_{name}_vector"] for row in rows], dtype=np.float64)
        for name in ("local", "feedback", "actual")
    }
    if not compact:
        matrix = np.stack([
            vector
            for index in range(len(rows))
            for vector in (arrays["local"][index], arrays["feedback"][index], arrays["actual"][index])
        ])
        return aligned_level_statistics_from_gram(
            matrix @ matrix.T,
            state_ids=state_ids,
            level_ids=("local", "feedback", "actual"),
            sign_flip_draws=4000,
            seed=20260820,
        )
    result: dict[str, Any] = {
        "schema": "kernel-analyzer-aligned-level-statistics-v1",
        "state_ids": state_ids,
        "level_ids": ["local", "feedback", "actual"],
        "measurement_geometry": "COUNT_SKETCH_256",
        "levels": {},
        "resultant_cosines": {},
    }
    for index, (name, values) in enumerate(arrays.items()):
        result["levels"][name] = path_statistics_from_gram(
            values @ values.T,
            state_ids=state_ids,
            sign_flip_draws=1000,
            seed=20260820 + index,
        )
    for left, right in (("local", "feedback"), ("local", "actual"), ("feedback", "actual")):
        left_sum, right_sum = arrays[left].sum(axis=0), arrays[right].sum(axis=0)
        result["resultant_cosines"][f"{left}__{right}"] = float(
            left_sum @ right_sum
            / max(float(np.linalg.norm(left_sum) * np.linalg.norm(right_sum)), 1e-30)
        )
    return result


def main() -> None:
    args = parser().parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("case_id")) != args.case_id:
        raise ValueError("checkpoint case ID does not match the requested case")
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("checkpoint contains no completed consequence step")
    gaps = [float(row.get("metadata", {}).get("paired_loss_gap", 0.0)) for row in rows]
    if not any(abs(value) > 1e-8 for value in gaps):
        raise SystemExit(3)
    plan = json.loads(args.case_plan.read_text(encoding="utf-8"))
    selected = [row for row in plan["cases"] if str(row.get("case_id")) == args.case_id]
    if len(selected) != 1:
        raise ValueError("case plan does not select exactly one case")
    candidate_master = payload["candidate_master"].float()
    repair_master = payload["repair_master"].float()
    compact = bool(payload.get("compact_long"))
    measured_steps = len(rows)
    result = {
        "schema": "kernel-analyzer-bias-consequence-certificate-v2_1",
        "status": "COMPLETE_PAIRED_LOSS_SPLIT",
        "runner": "scripts/finalize_consequence_checkpoint_on_loss_split.py",
        "source_runner": "scripts/run_bound_endpoint_consequence_v21.py",
        "case_id": args.case_id,
        "architecture": args.architecture,
        "model": str(args.model.resolve()),
        "release": str(args.release_dir.resolve()),
        "case_plan": str(args.case_plan.resolve()),
        "carrier": str(selected[0]["carrier"]),
        "carrier_coordinates": int(candidate_master.numel()),
        "steps": measured_steps,
        "planned_horizon_steps": args.planned_horizon,
        "trajectory_status": "EARLY_STOP_CONFIRMED_LOSS_SPLIT",
        "consequence_only_early_stop": True,
        "compact_long_horizon": compact,
        "measurement_geometry": "COUNT_SKETCH_256" if compact else "FULL_COORDINATE",
        "step_ids": [str(row["step_id"]) for row in rows],
        "statistics": statistics(rows, compact),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "final_drift_l2": float(torch.linalg.vector_norm(candidate_master - repair_master)),
        "max_recurrence_relative": max(
            float(row.get("metadata", {}).get("recurrence_relative", 0.0)) for row in rows
        ),
        "loss_audit": {
            "recorded": True,
            "recorded_steps": measured_steps,
            "split_step_count": sum(abs(value) > 1e-8 for value in gaps),
            "any_period_split": True,
            "max_abs_gap": max(abs(value) for value in gaps),
            "final_gap": gaps[-1],
            "last_512_mean": sum(gaps[-512:]) / len(gaps[-512:]),
            "last_512_max_abs": max(abs(value) for value in gaps[-512:]),
            "tolerance": 1e-8,
        },
        "source_checkpoint": str(args.checkpoint.resolve()),
        "claim_boundary": (
            "A four-counterfactual checkpoint reached the predeclared paired-loss "
            "terminal condition. This establishes a paired loss consequence for "
            "the tested implementation contrast; it does not establish 4096-step "
            "persistence or full-parameter training convergence. A separate "
            "formation artifact is required before calling the contrast biased."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "event": "PAIRED_LOSS_SPLIT_FINALIZED",
        "case_id": args.case_id,
        "steps": measured_steps,
        "max_abs_gap": result["loss_audit"]["max_abs_gap"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
