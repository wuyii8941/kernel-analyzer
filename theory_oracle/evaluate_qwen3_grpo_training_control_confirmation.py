#!/usr/bin/env python
"""Score the frozen Qwen3 GRPO training-control confirmation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from forkcert.detector import clip_active, clip_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-online", required=True)
    parser.add_argument("--b-online", required=True)
    parser.add_argument("--a-metadata", required=True)
    parser.add_argument("--b-metadata", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trajectory_record(
    name: str, online_path: Path, metadata_path: Path, epsilon: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(online_path)
    metadata = json.loads(metadata_path.read_text())
    audit = metadata.get("compile_audit") or {}
    graph_hashes = list(audit.get("graph_code_sha256") or [])
    candidate_identity = (
        int(audit.get("backend_compiles", 0)) > 0
        and int(audit.get("runtime_invocations", 0)) > 0
        and bool(graph_hashes)
        and all(bool(row.get("candidate_identity_valid")) for row in rows)
        and all(int(row.get("compiled_first_runtime_invocations", 0)) > 0 for row in rows)
        and all(int(row.get("compiled_second_runtime_invocations", 0)) > 0 for row in rows)
    )
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in ("logp_ref", "logp_alt", "old_logp", "advantage")
    )
    self_stable = all(
        float(row["delta_self_ref"]) == 0.0 and float(row["delta_self_alt"]) == 0.0
        for row in rows
    )
    keys = [
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"]), int(row["optimizer_step"]))
        for row in rows
    ]
    unique_keys = len(keys) == len(set(keys))
    state_keys = {
        (int(row["optimizer_step"]), int(row["rollout_batch"]), int(row["policy_iteration"]))
        for row in rows
    }
    applicable = []
    events = []
    for row in rows:
        sign = int(row["advantage_sign"])
        if sign == 0:
            continue
        ref = float(row["logp_ref"])
        alt = float(row["logp_alt"])
        old = float(row["old_logp"])
        ref_clip = clip_active(ref, old, sign, epsilon)
        alt_clip = clip_active(alt, old, sign, epsilon)
        item = {
            "trajectory": name,
            "optimizer_step": int(row["optimizer_step"]),
            "rollout_batch": int(row["rollout_batch"]),
            "case_id": str(row["case_id"]),
            "token_index": int(row["token_index"]),
            "token_id": int(row["token_id"]),
            "advantage_sign": sign,
            "old_logp": old,
            "logp_ref": ref,
            "logp_alt": alt,
            "signed_delta": alt - ref,
            "boundary": clip_boundary(sign, epsilon),
            "reference_boundary_distance": abs((ref - old) - clip_boundary(sign, epsilon)),
            "ref_clip": ref_clip,
            "alt_clip": alt_clip,
            "direction": "0->1" if (not ref_clip and alt_clip) else "1->0" if (ref_clip and not alt_clip) else "same",
        }
        applicable.append(item)
        if ref_clip != alt_clip:
            events.append(item)
    deltas = [float(row["logp_alt"]) - float(row["logp_ref"]) for row in rows]
    applicable_deltas = [item["signed_delta"] for item in applicable]
    cluster_deltas: dict[tuple[int, int, int], list[float]] = {}
    for row in rows:
        key = (int(row["optimizer_step"]), int(row["rollout_batch"]), int(row["policy_iteration"]))
        cluster_deltas.setdefault(key, []).append(float(row["logp_alt"]) - float(row["logp_ref"]))
    cluster_means = [fmean(values) for values in cluster_deltas.values()]
    record = {
        "trajectory": name,
        "online_path": str(online_path.resolve()),
        "online_sha256": sha256_file(online_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "rows": len(rows),
        "rollout_states": len(state_keys),
        "state_keys": [list(value) for value in sorted(state_keys)],
        "candidate_identity_valid": candidate_identity,
        "finite": finite,
        "self_stable": self_stable,
        "unique_token_state_keys": unique_keys,
        "compile_audit": audit,
        "applicable_tokens": len(applicable),
        "zero_advantage_tokens": len(rows) - len(applicable),
        "event_count": len(events),
        "direction_0_to_1": sum(item["direction"] == "0->1" for item in events),
        "direction_1_to_0": sum(item["direction"] == "1->0" for item in events),
        "mean_signed_delta_all": fmean(deltas) if deltas else None,
        "mean_signed_delta_applicable": fmean(applicable_deltas) if applicable_deltas else None,
        "positive_delta_tokens": sum(value > 0 for value in deltas),
        "negative_delta_tokens": sum(value < 0 for value in deltas),
        "zero_delta_tokens": sum(value == 0 for value in deltas),
        "cluster_mean_shift_min": min(cluster_means) if cluster_means else None,
        "cluster_mean_shift_max": max(cluster_means) if cluster_means else None,
        "cluster_mean_shift_sd": pstdev(cluster_means) if len(cluster_means) > 1 else 0.0,
    }
    return record, events


def main() -> None:
    args = parse_args()
    epsilon = 0.2
    inputs = [
        ("A", Path(args.a_online), Path(args.a_metadata)),
        ("B", Path(args.b_online), Path(args.b_metadata)),
    ]
    trajectories = []
    events = []
    for name, online, metadata in inputs:
        record, found = trajectory_record(name, online, metadata, epsilon)
        trajectories.append(record)
        events.extend(found)
    mechanics_valid = (
        len(trajectories) == 2
        and len({tuple(item["compile_audit"].get("graph_code_sha256") or []) for item in trajectories}) >= 1
        and all(item["candidate_identity_valid"] for item in trajectories)
        and all(item["finite"] and item["self_stable"] and item["unique_token_state_keys"] for item in trajectories)
        and all(item["rows"] == 5120 and item["rollout_states"] == 10 for item in trajectories)
        and len({item["online_sha256"] for item in trajectories}) == 2
    )
    events.sort(
        key=lambda item: (
            item["trajectory"], item["optimizer_step"], item["case_id"], item["token_index"]
        )
    )
    count_01 = sum(item["direction"] == "0->1" for item in events)
    count_10 = sum(item["direction"] == "1->0" for item in events)
    verdict = "INVALID" if not mechanics_valid else "REJECT" if events else "ACCEPT"
    payload = {
        "schema_version": "forkcert.qwen3-grpo-training-control-confirmation.v0.1",
        "contract": str((Path(__file__).parent / "QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md").resolve()),
        "epsilon": epsilon,
        "mechanics_valid": mechanics_valid,
        "finite_bank_strict_compatibility_verdict": verdict,
        "compiler_correctness": "NO CLAIM",
        "numerical_transition": "UNINSTANTIATED",
        "population_inference": "NOT CLAIMED; deterministic reference-trajectory strata",
        "trajectories": trajectories,
        "total_rows": sum(item["rows"] for item in trajectories),
        "total_rollout_states": sum(item["rollout_states"] for item in trajectories),
        "applicable_tokens": sum(item["applicable_tokens"] for item in trajectories),
        "zero_advantage_tokens": sum(item["zero_advantage_tokens"] for item in trajectories),
        "semantic_disagreement": len(events),
        "direction_0_to_1": count_01,
        "direction_1_to_0": count_10,
        "directional_clipping_shift_count": count_01 - count_10,
        "events": events,
        "first_event_for_one_step_followup": events[0] if events else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
