#!/usr/bin/env python
"""Evaluate matched-state B/H/N/U on Qwen GRPO scorer token fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forkcert.oracle import AcceptanceCriteria, TrainingOracle, Verdict


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64).ravel()
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_arrays(rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["state_id"]), []).append(row)
    result = {}
    for state_id, group in grouped.items():
        group.sort(key=lambda row: int(row["flat_index"]))
        indices = [int(row["flat_index"]) for row in group]
        if indices != list(range(512)):
            raise ValueError(f"{state_id}: expected exact flat indices 0..511")
        result[state_id] = tuple(
            np.array([float(row[field]) for row in group], dtype=np.float64)
            for field in ("logp_ref_first", "logp_ref_second", "logp_alt_first", "logp_alt_second")
        )
    return result


def profile_payload(profile) -> dict[str, Any]:
    state_means = np.asarray(profile.per_input_bias, dtype=np.float64)
    return {
        "observable": profile.name,
        "signed_global_mean_shift": profile.bias,
        "B_norm": profile.bias_norm,
        "B_relative": profile.relative_bias_norm,
        "B_relative_sampling_interval_approx95": [
            profile.relative_bias_lower, profile.relative_bias_upper,
        ],
        "H_repeat_corrected": profile.heterogeneity,
        "H_cv": profile.heterogeneity_cv,
        "N_paired_difference": profile.runtime_var,
        "N_ref": profile.ref_runtime_var,
        "N_candidate": profile.cand_runtime_var,
        "N_cv": profile.runtime_cv,
        "runtime_identified": profile.runtime_identified,
        "state_signed_mean_min": float(state_means.min()),
        "state_signed_mean_max": float(state_means.max()),
        "state_signed_mean_sd": float(state_means.std(ddof=1)) if len(state_means) > 1 else 0.0,
        "element_mean_effect_sha256": array_sha256(profile.element_bias),
        "n_state_clusters": profile.n_inputs,
        "n_repeats": profile.n_repeats,
        "field_shape": list(profile.output_shape),
        "uncertainty_method": profile.uncertainty_method,
    }


def build_oracle(query_id: str, distribution: str) -> TrainingOracle:
    return TrainingOracle(
        AcceptanceCriteria(),
        query_id=query_id,
        state_distribution=distribution,
        randomness_protocol="frozen algorithmic RNG; two measured same-state calls per implementation",
        coupling_protocol="same state tensors and restored RNG for corresponding eager/compiled calls",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", action="append", nargs=2, metavar=("NAME", "TOKENS"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    combined = build_oracle(
        "qwen3-grpo-heldout-bhn-v0.2-combined",
        "predeclared fixed-start held-out A/B/C rollout-state clusters",
    )
    trajectory_payloads = []
    total_states = 0
    for name, path_text in args.trajectory:
        path = Path(path_text)
        states = state_arrays(read_jsonl(path))
        if len(states) != 10:
            raise ValueError(f"trajectory {name}: expected 10 states, got {len(states)}")
        local = build_oracle(
            f"qwen3-grpo-heldout-bhn-v0.2-{name}",
            f"predeclared held-out trajectory {name} rollout-state clusters",
        )
        for state_id, (ref1, ref2, cand1, cand2) in states.items():
            global_id = f"{name}:{state_id}"
            local.record_operator_state("current_token_logprob_field", global_id, [ref1, ref2], [cand1, cand2])
            combined.record_operator_state("current_token_logprob_field", global_id, [ref1, ref2], [cand1, cand2])
        profile = local.operator_profiles()["current_token_logprob_field"]
        trajectory_payloads.append({
            "trajectory": name,
            "tokens_path": str(path.resolve()),
            "profile": profile_payload(profile),
        })
        total_states += len(states)

    combined_profile = combined.operator_profiles()["current_token_logprob_field"]
    verdict = combined.operator_verdicts()["current_token_logprob_field"]
    if verdict.verdict != Verdict.UNINSTANTIATED:
        raise RuntimeError("B/H/N descriptive bank must not invent an acceptance authority")
    payload = {
        "schema_version": "forkcert.qwen3-grpo-bhn.v0.2",
        "query_scope": {
            "reference": "grad-enabled eager Trainer scorer",
            "candidate": "grad-enabled tracked Inductor scorer",
            "observable": "aligned 4x128 current-token log-probability field",
            "state_distribution": combined.state_distribution,
            "randomness_protocol": combined.randomness_protocol,
            "coupling_protocol": combined.coupling_protocol,
        },
        "valid": True,
        "total_state_clusters": total_states,
        "trajectories": trajectory_payloads,
        "combined_profile": profile_payload(combined_profile),
        "acceptance_verdict": "UNINSTANTIATED",
        "semantic_event_ledger": "SEPARATE",
        "transition_ledger": "SEPARATE",
        "correctness": "UNINSTANTIATED",
        "nonclaims": [
            "B is implementation-relative, not truth-relative bias",
            "zero B does not imply safety",
            "nonzero B does not imply linear long-run accumulation",
            "whole-scorer observable is not operator causal attribution",
            "normal-approximation intervals are not token-iid population intervals",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
