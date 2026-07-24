#!/usr/bin/env python
"""Evaluate the frozen fixed-start Qwen GRPO held-out transport bank."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from forkcert.detector import clip_active, clip_boundary


HERE = Path(__file__).resolve().parent
BASE_EVALUATOR = HERE / "evaluate_qwen3_grpo_grad_event_bank_v0_4.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("grad_event_bank_v04", BASE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def stage_for_step(step: int) -> str:
    if step in {2, 5, 8}:
        return "early"
    if step in {11, 14, 17}:
        return "middle"
    if step in {20, 23, 26, 29}:
        return "late"
    raise ValueError(f"unexpected measured optimizer step: {step}")


def boundary_band(distance: float) -> str:
    if distance < 0 or not math.isfinite(distance):
        raise ValueError(f"invalid boundary distance: {distance}")
    if distance < 1e-4:
        return "[0,1e-4)"
    if distance < 1e-3:
        return "[1e-4,1e-3)"
    if distance < 1e-2:
        return "[1e-3,1e-2)"
    return "[1e-2,inf)"


def transport_label(*, valid: bool, unstable: int, event_runs: int) -> str:
    if not valid:
        return "INVALID"
    if unstable:
        return "INDETERMINATE_REPEAT_VARIABILITY"
    if event_runs >= 2:
        return "SUPPORTED_EVENT_REPLICATION"
    if event_runs == 1:
        return "INDETERMINATE_SINGLE_EVENT_RUN"
    return "EVENT_NOT_OBSERVED_IN_BANK"


def run_profile(name: str, token_path: Path, epsilon: float) -> dict[str, Any]:
    rows = BASE.read_jsonl(token_path)
    state_effects: dict[str, list[float]] = defaultdict(list)
    state_steps: dict[str, int] = {}
    stage_effects: dict[str, list[float]] = defaultdict(list)
    exposure: dict[str, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "stable_events": 0})
    event_counts = {"0->1": 0, "1->0": 0}

    for row in rows:
        state_id = str(row["state_id"])
        step = int(row["optimizer_step"])
        stage = stage_for_step(step)
        state_steps[state_id] = step
        ref1 = float(row["logp_ref_first"])
        ref2 = float(row["logp_ref_second"])
        alt1 = float(row["logp_alt_first"])
        alt2 = float(row["logp_alt_second"])
        effect = ((alt1 - ref1) + (alt2 - ref2)) / 2.0
        state_effects[state_id].append(effect)
        stage_effects[stage].append(effect)

        sign = int(row["advantage_sign"])
        if sign == 0:
            continue
        old = float(row["old_logp"])
        ref = (ref1 + ref2) / 2.0
        distance = abs((ref - old) - clip_boundary(sign, epsilon))
        band = boundary_band(distance)
        exposure[band]["tokens"] += 1

        ref_decisions = (
            clip_active(ref1, old, sign, epsilon),
            clip_active(ref2, old, sign, epsilon),
        )
        alt_decisions = (
            clip_active(alt1, old, sign, epsilon),
            clip_active(alt2, old, sign, epsilon),
        )
        stable = ref_decisions[0] == ref_decisions[1] and alt_decisions[0] == alt_decisions[1]
        if stable and ref_decisions[0] != alt_decisions[0]:
            direction = "0->1" if not ref_decisions[0] else "1->0"
            event_counts[direction] += 1
            exposure[band]["stable_events"] += 1

    means = {state: fmean(values) for state, values in state_effects.items()}
    ordered_states = sorted(means, key=lambda state: state_steps[state])
    return {
        "run": name,
        "state_means": [
            {
                "state_id": state,
                "optimizer_step": state_steps[state],
                "stage": stage_for_step(state_steps[state]),
                "mean_effect": means[state],
            }
            for state in ordered_states
        ],
        "stage_mean_effect": {
            stage: fmean(stage_effects[stage]) for stage in ("early", "middle", "late")
        },
        "state_mean_signs": {
            "negative": sum(value < 0 for value in means.values()),
            "zero": sum(value == 0 for value in means.values()),
            "positive": sum(value > 0 for value in means.values()),
        },
        "event_counts": event_counts,
        "boundary_exposure": {
            band: exposure[band]
            for band in ("[0,1e-4)", "[1e-4,1e-3)", "[1e-3,1e-2)", "[1e-2,inf)")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("a", "b", "c"):
        parser.add_argument(f"--{prefix}-tokens", required=True)
        parser.add_argument(f"--{prefix}-states", required=True)
        parser.add_argument(f"--{prefix}-metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    trajectories = []
    profiles = []
    events: list[dict[str, Any]] = []
    unstable = 0
    for name in ("A", "B", "C"):
        prefix = name.lower()
        token_path = Path(getattr(args, f"{prefix}_tokens"))
        record, found, unstable_count = BASE.trajectory_record(
            name=name,
            token_path=token_path,
            state_path=Path(getattr(args, f"{prefix}_states")),
            metadata_path=Path(getattr(args, f"{prefix}_metadata")),
            epsilon=0.2,
        )
        trajectories.append(record)
        profiles.append(run_profile(name, token_path, 0.2))
        events.extend(found)
        unstable += unstable_count

    valid = all(row["mechanics_valid"] for row in trajectories)
    event_runs = sum(profile["event_counts"]["0->1"] + profile["event_counts"]["1->0"] > 0 for profile in profiles)
    applicable = sum(row["applicable_tokens"] for row in trajectories)
    count_01 = sum(profile["event_counts"]["0->1"] for profile in profiles)
    count_10 = sum(profile["event_counts"]["1->0"] for profile in profiles)

    if not valid:
        compatibility = "INVALID"
    elif unstable:
        compatibility = "INDETERMINATE"
    else:
        compatibility = "REJECT" if events else "ACCEPT_FINITE_BANK_ONLY"

    payload = {
        "schema_version": "forkcert.qwen3-grpo-heldout-transport.v0.1",
        "design": str((HERE / "QWEN3_GRPO_HELDOUT_TRANSPORT_DESIGN_V0_1_2026-07-18.md").resolve()),
        "construction_validity": "VALID" if valid else "INVALID",
        "finite_bank_semantic_compatibility": compatibility,
        "transport_event_replication": transport_label(valid=valid, unstable=unstable, event_runs=event_runs),
        "population_prevalence": "INDETERMINATE_THREE_RUN_CLUSTERS",
        "correctness": "UNINSTANTIATED",
        "total_rows": sum(row["rows"] for row in trajectories),
        "total_state_clusters": sum(row["rollout_states"] for row in trajectories),
        "total_run_clusters": len(trajectories),
        "applicable_tokens_descriptive_denominator": applicable,
        "stable_event_count": len(events),
        "repeat_unstable_event_count": unstable,
        "event_run_count": event_runs,
        "direction_0_to_1_count": count_01,
        "direction_1_to_0_count": count_10,
        "finite_bank_directional_shift": (count_01 - count_10) / applicable if applicable else None,
        "finite_bank_disagreement": len(events) / applicable if applicable else None,
        "trajectories": trajectories,
        "run_profiles": profiles,
        "events": sorted(events, key=lambda row: (row["trajectory"], row["optimizer_step"], row["flat_index"])),
        "nonclaims": [
            "no eager-as-truth claim",
            "no correctness claim",
            "no token-iid population interval",
            "no checkpoint/model/hardware generality",
            "no predictive-score claim",
            "no natural optimizer-transition claim",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

