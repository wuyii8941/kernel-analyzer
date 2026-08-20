#!/usr/bin/env python3
"""Promote replicated development signals into a disjoint state confirmation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_oracle_recovery/prospective"
OUTPUT = ROOT / "results/property/bias_oracle_recovery/confirmation"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def development_role(case: dict[str, Any]) -> tuple[str, int | None]:
    rows = case["reference_relative_parameter_gradient"]
    calibration = rows["calibration"]
    confirmation = rows["confirmation"]
    means = [float(calibration["mean_coefficient"]),
             float(confirmation["mean_coefficient"])]
    if any(not math.isfinite(value) or value == 0.0 for value in means):
        return "UNRESOLVED", None
    signs = [1 if value > 0.0 else -1 for value in means]
    directional = "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
    if signs[0] == signs[1] and (
        calibration["status"] == directional
        or confirmation["status"] == directional
    ):
        return "CANDIDATE", signs[0]
    if signs[0] != signs[1]:
        return "SIGN_CHANGING_CONTROL", None
    return "UNRESOLVED", None


def tail_bank(model: str) -> dict[str, Any]:
    source = load(ROOT / f"results/coverage/{model}_seq128_input_bank.json")
    old = load(ROOT / f"results/coverage/{model}_seq64_input_bank.json")
    source_rows = source.get("states", source.get("records"))
    old_rows = old.get("states", old.get("records"))
    if len(source_rows) != 32 or len(old_rows) != 32:
        raise RuntimeError("confirmation bank requires two complete 32-state banks")
    token_key = "input_ids" if model == "qwen" else "token_ids"
    old_tokens = {tuple(row[token_key]) for row in old_rows}
    rows = []
    for index, row in enumerate(source_rows):
        tokens = list(row[token_key][64:128])
        if len(tokens) != 64 or tuple(tokens) in old_tokens:
            raise RuntimeError("tail confirmation state overlaps the development bank")
        if model == "qwen":
            rows.append({
                "cluster_id": f"{row['cluster_id']}:tail64",
                "input_ids": tokens,
                "layer_role": row.get("layer_role", "unknown"),
                "length": 64,
                "length_bucket": "seq64",
                "sequence_id": f"{row['sequence_id']}:tail64",
                "split": "prospective_confirmation",
            })
        else:
            rows.append({
                "state_id": f"deepseek8b-tail64-{index:02d}",
                "token_ids": tokens,
            })
    return {"states": rows, "source": f"{model}_seq128_tail_tokens_64_127"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    plans: dict[str, list[dict[str, Any]]] = {"qwen": [], "deepseek8b": []}
    for model in plans:
        plan = load(SOURCE / f"{model}_plan.json")
        by_id = {str(row["case_id"]): row for row in plan["cases"]}
        for result in sorted((SOURCE / model).glob("*.json")):
            payload = load(result)
            role, direction = development_role(payload)
            case_id = str(payload["case_id"])
            cases.append({
                "case_id": case_id,
                "model": model,
                "development_role": role,
                "frozen_direction": direction,
            })
            if role in {"CANDIDATE", "SIGN_CHANGING_CONTROL"}:
                plans[model].append(by_id[case_id])
    candidates = [row for row in cases if row["development_role"] == "CANDIDATE"]
    controls = [
        row for row in cases
        if row["development_role"] == "SIGN_CHANGING_CONTROL"
    ]
    if len(candidates) != 3 or len(controls) != 2:
        raise RuntimeError(
            "development selection changed; review instead of silently changing confirmation"
        )
    for model, rows in plans.items():
        (OUTPUT / f"{model}_plan.json").write_text(
            json.dumps({"cases": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (OUTPUT / f"{model}_state_bank.json").write_text(
            json.dumps(tail_bank(model), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    protocol = {
        "schema": "kernel-analyzer-moving-frame-confirmation-v1",
        "status": "FROZEN_BEFORE_TAIL_STATE_MEASUREMENT",
        "development_selection_rule": {
            "candidate": (
                "SAME_NONZERO_PARTITION_MEAN_SIGN_AND_AT_LEAST_ONE_"
                "PARTITION_DIRECTIONAL"
            ),
            "control": "OPPOSITE_PARTITION_MEAN_SIGNS",
            "threshold_changed": False,
        },
        "cases": cases,
        "confirmation": {
            "states": 32,
            "state_source": "TOKENS_64_TO_127_OF_DISJOINT_SEQ128_NATURAL_STATES",
            "candidate_success": (
                "POOLED_95_PERCENT_BOOTSTRAP_INTERVAL_EXCLUDES_ZERO_IN_"
                "FROZEN_DIRECTION"
            ),
            "minimum_absolute_mean_coefficient": 1e-5,
            "bootstrap_draws": 4000,
            "control_role": (
                "REGRESSION_FOR_SPURIOUS_UNIVERSAL_DIRECTION_NOT_A_SAFETY_LABEL"
            ),
            "threshold_changes_after_measurement": "FORBIDDEN",
        },
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "candidates": [row["case_id"] for row in candidates],
        "controls": [row["case_id"] for row in controls],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
