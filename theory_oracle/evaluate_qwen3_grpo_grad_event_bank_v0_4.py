#!/usr/bin/env python
"""Evaluate the frozen finite-bank Qwen GRPO grad-enabled paired scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from forkcert.detector import clip_active, clip_boundary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def direction(reference: bool, candidate: bool) -> str:
    if not reference and candidate:
        return "0->1"
    if reference and not candidate:
        return "1->0"
    return "same"


def trajectory_record(
    *, name: str, token_path: Path, state_path: Path, metadata_path: Path, epsilon: float
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    rows = read_jsonl(token_path)
    states = read_jsonl(state_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state_map = {str(row["state_id"]): row for row in states}
    state_ids = {str(row["state_id"]) for row in rows}
    keys = [
        (
            str(row["state_id"]),
            str(row["case_id"]),
            int(row["token_index"]),
            int(row["token_id"]),
        )
        for row in rows
    ]
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in (
            "logp_ref_first",
            "logp_ref_second",
            "logp_alt_first",
            "logp_alt_second",
            "old_logp",
            "advantage",
        )
    )
    state_gates = all(
        bool(row.get(field))
        for row in states
        for field in (
            "autograd_enabled",
            "all_outputs_require_grad",
            "accelerate_native_amp",
            "accelerate_forward_wrapped",
            "candidate_identity_valid",
            "gradients_preserved",
            "tensor_versions_preserved",
            "trainer_steps_preserved",
            "rng_restored_exactly",
        )
    )
    state_shapes = all(
        int(row.get("batch_size", -1)) == 4
        and int(row.get("completion_length", -1)) == 128
        and str(row.get("attention_backend_locked")) == "MATH"
        and str(row.get("accelerate_mixed_precision")) == "fp16"
        for row in states
    )
    compile_audit = metadata.get("grad_compile_audit") or {}
    candidate_audit = (
        int(compile_audit.get("backend_compiles", 0)) > 0
        and int(compile_audit.get("runtime_invocations", 0)) > 0
        and bool(compile_audit.get("graph_code_sha256"))
    )

    events: list[dict[str, Any]] = []
    unstable_events = 0
    applicable = 0
    effects: list[float] = []
    state_effects: dict[str, list[float]] = {}
    ref_repeat: list[float] = []
    alt_repeat: list[float] = []
    boundary_distances: list[float] = []
    for row in rows:
        ref1 = float(row["logp_ref_first"])
        ref2 = float(row["logp_ref_second"])
        alt1 = float(row["logp_alt_first"])
        alt2 = float(row["logp_alt_second"])
        effect = ((alt1 - ref1) + (alt2 - ref2)) / 2.0
        effects.append(effect)
        state_effects.setdefault(str(row["state_id"]), []).append(effect)
        ref_repeat.append(abs(ref2 - ref1))
        alt_repeat.append(abs(alt2 - alt1))
        sign = int(row["advantage_sign"])
        if sign == 0:
            continue
        applicable += 1
        old = float(row["old_logp"])
        decisions = {
            "reference_first": clip_active(ref1, old, sign, epsilon),
            "reference_second": clip_active(ref2, old, sign, epsilon),
            "candidate_first": clip_active(alt1, old, sign, epsilon),
            "candidate_second": clip_active(alt2, old, sign, epsilon),
        }
        first_direction = direction(
            decisions["reference_first"], decisions["candidate_first"]
        )
        second_direction = direction(
            decisions["reference_second"], decisions["candidate_second"]
        )
        stable = first_direction == second_direction and (
            decisions["reference_first"] == decisions["reference_second"]
            and decisions["candidate_first"] == decisions["candidate_second"]
        )
        if not stable:
            unstable_events += 1
        boundary = clip_boundary(sign, epsilon)
        boundary_distances.append(abs(((ref1 + ref2) / 2.0 - old) - boundary))
        if stable and first_direction != "same":
            events.append(
                {
                    "trajectory": name,
                    "state_id": str(row["state_id"]),
                    "optimizer_step": int(row["optimizer_step"]),
                    "rollout_batch": int(row["rollout_batch"]),
                    "policy_iteration": int(row["policy_iteration"]),
                    "batch_index": int(row["batch_index"]),
                    "flat_index": int(row["flat_index"]),
                    "case_id": str(row["case_id"]),
                    "token_index": int(row["token_index"]),
                    "token_id": int(row["token_id"]),
                    "advantage_sign": sign,
                    "old_logp": old,
                    "logp_ref": (ref1 + ref2) / 2.0,
                    "logp_alt": (alt1 + alt2) / 2.0,
                    "signed_delta": effect,
                    "boundary": boundary,
                    "reference_boundary_distance": boundary_distances[-1],
                    "ref_clip": decisions["reference_first"],
                    "alt_clip": decisions["candidate_first"],
                    "direction": first_direction,
                    "repeat_stable": True,
                }
            )

    cluster_means = [fmean(values) for values in state_effects.values()]
    mechanics_valid = (
        bool(rows)
        and len(states) == 10
        and state_ids == set(state_map)
        and all(sum(str(row["state_id"]) == state_id for row in rows) == 512 for state_id in state_ids)
        and len(keys) == len(set(keys))
        and finite
        and state_gates
        and state_shapes
        and candidate_audit
    )
    record = {
        "trajectory": name,
        "token_path": str(token_path.resolve()),
        "token_sha256": sha256_file(token_path),
        "state_path": str(state_path.resolve()),
        "state_sha256": sha256_file(state_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "rows": len(rows),
        "rollout_states": len(states),
        "mechanics_valid": mechanics_valid,
        "finite": finite,
        "unique_token_state_keys": len(keys) == len(set(keys)),
        "state_gates_valid": state_gates,
        "state_shapes_valid": state_shapes,
        "candidate_audit_valid": candidate_audit,
        "compile_audit": compile_audit,
        "applicable_tokens": applicable,
        "stable_event_count": len(events),
        "repeat_unstable_event_count": unstable_events,
        "direction_0_to_1": sum(event["direction"] == "0->1" for event in events),
        "direction_1_to_0": sum(event["direction"] == "1->0" for event in events),
        "average_implementation_relative_shift": {
            "token_weighted": fmean(effects) if effects else None,
            "state_weighted": fmean(cluster_means) if cluster_means else None,
        },
        "state_conditioned_heterogeneity": {
            "state_mean_min": min(cluster_means) if cluster_means else None,
            "state_mean_max": max(cluster_means) if cluster_means else None,
            "state_mean_sd": pstdev(cluster_means) if len(cluster_means) > 1 else 0.0,
        },
        "within_state_runtime_variability": {
            "reference_nonzero_repeat_tokens": sum(value != 0.0 for value in ref_repeat),
            "candidate_nonzero_repeat_tokens": sum(value != 0.0 for value in alt_repeat),
            "reference_max_abs_repeat_delta": max(ref_repeat) if ref_repeat else None,
            "candidate_max_abs_repeat_delta": max(alt_repeat) if alt_repeat else None,
        },
        "reference_boundary_exposure": {
            "minimum_distance": min(boundary_distances) if boundary_distances else None,
            "maximum_distance": max(boundary_distances) if boundary_distances else None,
        },
    }
    return record, events, unstable_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("a", "b"):
        parser.add_argument(f"--{prefix}-tokens", required=True)
        parser.add_argument(f"--{prefix}-states", required=True)
        parser.add_argument(f"--{prefix}-metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    trajectories = []
    events: list[dict[str, Any]] = []
    unstable = 0
    for name in ("A", "B"):
        prefix = name.lower()
        record, found, unstable_count = trajectory_record(
            name=name,
            token_path=Path(getattr(args, f"{prefix}_tokens")),
            state_path=Path(getattr(args, f"{prefix}_states")),
            metadata_path=Path(getattr(args, f"{prefix}_metadata")),
            epsilon=0.2,
        )
        trajectories.append(record)
        events.extend(found)
        unstable += unstable_count

    events.sort(
        key=lambda event: (
            event["trajectory"],
            event["optimizer_step"],
            event["rollout_batch"],
            event["flat_index"],
        )
    )
    mechanics_valid = all(row["mechanics_valid"] for row in trajectories)
    applicable = sum(row["applicable_tokens"] for row in trajectories)
    count_01 = sum(event["direction"] == "0->1" for event in events)
    count_10 = sum(event["direction"] == "1->0" for event in events)
    if not mechanics_valid:
        verdict = "INVALID"
    elif unstable:
        verdict = "INDETERMINATE"
    else:
        verdict = "REJECT" if events else "ACCEPT"
    payload = {
        "schema_version": "forkcert.qwen3-grpo-grad-event-bank.v0.4",
        "design": str(
            (Path(__file__).parent / "QWEN3_GRPO_GRAD_EVENT_BANK_DESIGN_V0_4_2026-07-17.md").resolve()
        ),
        "mechanics_valid": mechanics_valid,
        "finite_bank_grad_context_compatibility_verdict": verdict,
        "compiler_correctness": "NO CLAIM",
        "update_effect": "NOT IN SCOPE",
        "population_inference": "NOT CLAIMED; deterministic instrumented reference-trajectory strata",
        "trajectories": trajectories,
        "total_rows": sum(row["rows"] for row in trajectories),
        "total_rollout_states": sum(row["rollout_states"] for row in trajectories),
        "applicable_tokens": applicable,
        "stable_semantic_disagreement_count": len(events),
        "repeat_unstable_event_count": unstable,
        "direction_0_to_1_count": count_01,
        "direction_1_to_0_count": count_10,
        "directional_semantic_shift": (count_01 - count_10) / applicable if applicable else None,
        "semantic_disagreement": len(events) / applicable if applicable else None,
        "witness_selection_rule": "first stable disagreement ordered by trajectory, optimizer_step, rollout_batch, flat_index",
        "events": events,
        "first_stable_event_for_one_step_followup": events[0] if events else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
