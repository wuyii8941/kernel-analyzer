#!/usr/bin/env python
"""Evaluate four fresh-process arms of the frozen Qwen natural transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.qwen3-grpo-natural-transition-evaluation.v0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-1", required=True)
    parser.add_argument("--eager-2", required=True)
    parser.add_argument("--compiled-1", required=True)
    parser.add_argument("--compiled-2", required=True)
    parser.add_argument(
        "--effect-vector-dir",
        help="Optional directory for paired update-delta (U2) artifacts.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_result(path: str, arm: str, repeat: int) -> dict[str, Any]:
    item = json.loads(Path(path).read_text())
    if item.get("schema_version") != "forkcert.qwen3-grpo-natural-transition-arm.v0.2":
        raise ValueError(f"unsupported arm schema: {path}")
    if item.get("arm") != arm or int(item.get("repeat")) != repeat:
        raise ValueError(f"arm/repeat identity mismatch: {path}")
    return item


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_profile(ref: list[float], cand: list[float]) -> dict[str, Any]:
    effects = [b - a for a, b in zip(ref, cand, strict=True)]
    mean_effect = sum(effects) / len(effects)
    repeat_variance = (
        sum((value - mean_effect) ** 2 for value in effects) / (len(effects) - 1)
        if len(effects) > 1
        else None
    )
    return {
        "reference_repeats": ref,
        "candidate_repeats": cand,
        "paired_effect_repeats": effects,
        "state_effect_signed_mean": mean_effect,
        "state_effect_magnitude": abs(mean_effect),
        "B_signed_mean_effect": mean_effect,
        "B_field_status": "DEPRECATED_ALIAS_NOT_POPULATION_B",
        "H_state_heterogeneity": "UNIDENTIFIABLE_ONE_STATE",
        "N_paired_repeat_variance": repeat_variance,
        "reference_self_range": max(ref) - min(ref),
        "candidate_self_range": max(cand) - min(cand),
        "U_state_sampling": "NOT_ESTIMATED_SELECTED_STATE",
    }


def vector_profile(
    paths: dict[str, list[str]],
    *,
    effect_vector_dir: Path | None = None,
    artifact_prefix: str | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    handles: dict[str, list[Any]] = {}
    keys: list[str] | None = None
    for arm, arm_paths in paths.items():
        handles[arm] = []
        for path in arm_paths:
            item = Path(path)
            if not item.is_file():
                raise FileNotFoundError(item)
            handle = safe_open(item, framework="pt", device="cpu")
            current_keys = sorted(handle.keys())
            if keys is None:
                keys = current_keys
            elif keys != current_keys:
                raise ValueError("vector artifact parameter keys differ across arms/repeats")
            handles[arm].append(handle)
    if not keys:
        raise ValueError("empty vector artifacts")

    ref_self_sq = 0.0
    cand_self_sq = 0.0
    effect_sq = 0.0
    effect_dot_ref = 0.0
    ref_mean_sq = 0.0
    cand_mean_sq = 0.0
    paired_noise_total_variance = 0.0
    max_abs_effect = 0.0
    pair_effect_dot_reference = [0.0, 0.0]
    pair_reference_sq = [0.0, 0.0]
    pair_effect_sq = [0.0, 0.0]
    paired_effect_tensors: list[dict[str, Any]] | None = (
        [{}, {}] if effect_vector_dir is not None else None
    )
    per_parameter = []
    for name in keys:
        r1 = handles["eager"][0].get_tensor(name).double()
        r2 = handles["eager"][1].get_tensor(name).double()
        c1 = handles["compiled"][0].get_tensor(name).double()
        c2 = handles["compiled"][1].get_tensor(name).double()
        if not (r1.shape == r2.shape == c1.shape == c2.shape):
            raise ValueError(f"shape mismatch for {name}")
        ref_mean = (r1 + r2) / 2
        cand_mean = (c1 + c2) / 2
        effect = cand_mean - ref_mean
        effect_1 = c1 - r1
        effect_2 = c2 - r2
        for index, (pair_effect, pair_reference) in enumerate(
            ((effect_1, r1), (effect_2, r2))
        ):
            pair_effect_dot_reference[index] += float(
                (pair_effect * pair_reference).sum().item()
            )
            pair_reference_sq[index] += float((pair_reference * pair_reference).sum().item())
            pair_effect_sq[index] += float((pair_effect * pair_effect).sum().item())
            if paired_effect_tensors is not None:
                paired_effect_tensors[index][name] = pair_effect.to(dtype=r1.dtype).contiguous()
        noise_variance_sum = float(((effect_1 - effect_2) ** 2).sum().item() / 2.0)
        current_effect_sq = float((effect * effect).sum().item())
        current_max = float(effect.abs().max().item())
        ref_self_sq += float(((r2 - r1) ** 2).sum().item())
        cand_self_sq += float(((c2 - c1) ** 2).sum().item())
        effect_sq += current_effect_sq
        ref_mean_sq += float((ref_mean * ref_mean).sum().item())
        cand_mean_sq += float((cand_mean * cand_mean).sum().item())
        effect_dot_ref += float((effect * ref_mean).sum().item())
        paired_noise_total_variance += noise_variance_sum
        max_abs_effect = max(max_abs_effect, current_max)
        per_parameter.append(
            {
                "name": name,
                "coordinates": effect.numel(),
                "state_effect_l2": math.sqrt(current_effect_sq),
                "state_effect_max_abs": current_max,
                "B_effect_l2": math.sqrt(current_effect_sq),
                "B_field_status": "DEPRECATED_ALIAS_NOT_POPULATION_B",
                "N_paired_coordinate_variance_sum": noise_variance_sum,
            }
        )
    ref_norm = math.sqrt(ref_mean_sq)
    cand_norm = math.sqrt(cand_mean_sq)
    effect_norm = math.sqrt(effect_sq)
    u1_repeats = [
        pair_effect_dot_reference[index] / pair_reference_sq[index]
        if pair_reference_sq[index]
        else None
        for index in range(2)
    ]
    vector_artifacts = None
    if paired_effect_tensors is not None:
        if not artifact_prefix:
            raise ValueError("artifact_prefix is required with effect_vector_dir")
        effect_vector_dir.mkdir(parents=True, exist_ok=True)
        vector_artifacts = []
        for index, tensors in enumerate(paired_effect_tensors, start=1):
            path = effect_vector_dir / f"{artifact_prefix}_paired_delta_repeat{index}.safetensors"
            save_file(tensors, path)
            vector_artifacts.append(
                {"repeat": index, "path": str(path.resolve()), "sha256": sha256_file(path)}
            )
    return {
        "state_effect_l2": effect_norm,
        "state_effect_relative_to_reference_l2": effect_norm / ref_norm if ref_norm else None,
        "state_effect_max_abs_coordinate": max_abs_effect,
        "state_effect_alignment_with_reference": effect_dot_ref / (effect_norm * ref_norm)
        if effect_norm and ref_norm
        else None,
        "state_effect_U1_reference_aligned_shift": effect_dot_ref / ref_mean_sq
        if ref_mean_sq
        else None,
        "paired_U1_repeats": u1_repeats,
        "paired_effect_l2_repeats": [math.sqrt(value) for value in pair_effect_sq],
        "paired_effect_vector_artifacts": vector_artifacts,
        "B_effect_l2": effect_norm,
        "B_effect_alignment_with_reference": effect_dot_ref / (effect_norm * ref_norm)
        if effect_norm and ref_norm
        else None,
        "B_field_status": "DEPRECATED_ALIAS_NOT_POPULATION_B",
        "reference_mean_l2": ref_norm,
        "candidate_mean_l2": cand_norm,
        "reference_self_difference_l2": math.sqrt(ref_self_sq),
        "candidate_self_difference_l2": math.sqrt(cand_self_sq),
        "H_state_heterogeneity": "UNIDENTIFIABLE_ONE_STATE",
        "N_paired_coordinate_variance_sum": paired_noise_total_variance,
        "N_paired_effect_sd_l2": math.sqrt(paired_noise_total_variance),
        "U_state_sampling": "NOT_ESTIMATED_SELECTED_STATE",
        "per_parameter": per_parameter,
    }


def artifact_paths(rows: list[dict[str, Any]], key: str) -> list[str]:
    paths = []
    for row in rows:
        record = row.get("vector_artifacts", {}).get(key)
        if not record:
            raise ValueError(f"missing {key} vector artifact")
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{key} artifact hash mismatch: {path}")
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    eager = [load_result(args.eager_1, "eager", 1), load_result(args.eager_2, "eager", 2)]
    compiled = [
        load_result(args.compiled_1, "compiled", 1),
        load_result(args.compiled_2, "compiled", 2),
    ]
    rows = eager + compiled
    all_arms_valid = all(row["valid"] for row in rows)
    pre_fields = [
        "parameter_digest",
        "buffer_digest",
        "optimizer_digest",
        "scheduler_digest",
        "scaler_digest",
        "rng",
    ]
    pre_state_identical = all(
        all(row["pre_state"][field] == rows[0]["pre_state"][field] for row in rows[1:])
        for field in pre_fields
    )
    snapshot_identical = all(row["snapshot"] == rows[0]["snapshot"] for row in rows[1:])
    candidate_identity_valid = all(row["compiler"]["candidate_identity_valid"] for row in compiled)
    scorer_anchors_exact = all(row["anchors"]["scorer_anchor_exact"] for row in rows)
    identity_sources = [
        row.get("anchors", {}).get("identity_source", "legacy_online_anchor") for row in rows
    ]
    realization_contracts = [
        row.get("anchors", {}).get("realization_contract_sha256") for row in rows
    ]
    realization_contract_exact = len(set(identity_sources)) == 1 and (
        (
            identity_sources[0] == "prospective_realization_contract"
            and all(value is not None for value in realization_contracts)
            and len(set(realization_contracts)) == 1
        )
        or identity_sources[0] == "legacy_online_anchor"
    )
    realization_identity_exact = all(
        row.get("realization") == rows[0].get("realization") for row in rows[1:]
    )
    construction_valid = all(
        [
            all_arms_valid,
            pre_state_identical,
            snapshot_identical,
            candidate_identity_valid,
            scorer_anchors_exact,
            realization_contract_exact,
            realization_identity_exact,
        ]
    )

    loss_profile = scalar_profile(
        [row["continuous"]["loss"] for row in eager],
        [row["continuous"]["loss"] for row in compiled],
    )
    gradient_profile = vector_profile(
        {
            "eager": artifact_paths(eager, "clipped_gradients"),
            "compiled": artifact_paths(compiled, "clipped_gradients"),
        }
    )
    update_profile = vector_profile(
        {
            "eager": artifact_paths(eager, "parameter_updates"),
            "compiled": artifact_paths(compiled, "parameter_updates"),
        },
        effect_vector_dir=Path(args.effect_vector_dir).resolve()
        if args.effect_vector_dir
        else None,
        artifact_prefix="parameter_update",
    )

    eager_post = [row["post_state"] for row in eager]
    compiled_post = [row["post_state"] for row in compiled]
    post_fields = ["parameter_digest", "buffer_digest", "optimizer_digest", "scheduler_digest", "scaler_digest"]
    eager_self_exact = all(eager_post[0][field] == eager_post[1][field] for field in post_fields)
    compiled_self_exact = all(compiled_post[0][field] == compiled_post[1][field] for field in post_fields)
    cross_state_equal = all(eager_post[0][field] == compiled_post[0][field] for field in post_fields)
    self_stable = eager_self_exact and compiled_self_exact
    if not construction_valid:
        transition_verdict = "INVALID"
    elif not self_stable:
        transition_verdict = "INDETERMINATE_RUNTIME_VARIABILITY"
    elif cross_state_equal:
        transition_verdict = "ACCEPT_EXACT_SELECTED_STATE"
    else:
        transition_verdict = "REJECT_EXACT_SELECTED_STATE"

    event_fields = [
        "clip_decisions",
        "gradient_clip_triggered",
        "optimizer_step_skipped",
        "nonfinite_gradient",
        "nonfinite_update",
        "amp_scale_after",
    ]
    event_disagreement = {
        field: eager[0]["semantic"][field] != compiled[0]["semantic"][field] for field in event_fields
    }
    event_self_stable = all(
        eager[0]["semantic"][field] == eager[1]["semantic"][field]
        and compiled[0]["semantic"][field] == compiled[1]["semantic"][field]
        for field in event_fields
    )

    output = {
        "schema_version": SCHEMA_VERSION,
        "construction_valid": construction_valid,
        "validity": {
            "all_arms_valid": all_arms_valid,
            "pre_state_identical": pre_state_identical,
            "snapshot_identical": snapshot_identical,
            "candidate_identity_valid": candidate_identity_valid,
            "scorer_anchors_exact": scorer_anchors_exact,
            "realization_contract_exact": realization_contract_exact,
            "realization_identity_exact": realization_identity_exact,
            "eager_next_state_self_exact": eager_self_exact,
            "compiled_next_state_self_exact": compiled_self_exact,
        },
        "query_scope": {
            "state": f"preselected Qwen GRPO optimizer-step-{rows[0]['snapshot']['optimizer_step']} pre-minibatch state",
            "reference": "eager scorer forward plus common natural GRPO transition",
            "candidate": "tracked Inductor scorer forward plus common natural GRPO transition",
            "repeats": "two fresh processes per arm",
            "population": "selected-state effect only; not population B",
        },
        "profiles": {
            "loss": loss_profile,
            "clipped_gradient": gradient_profile,
            "parameter_update": update_profile,
        },
        "semantic_events": {
            "self_stable": event_self_stable,
            "cross_arm_disagreement": event_disagreement,
            "any_disagreement": any(event_disagreement.values()),
        },
        "next_state": {
            "self_stable": self_stable,
            "cross_arm_exact_equal": cross_state_equal,
            "verdict": transition_verdict,
            "compared_components": post_fields,
        },
        "verdict_ledgers": {
            "construction": "VALID" if construction_valid else "INVALID",
            "selected_state_transition_compatibility": transition_verdict,
            "population_prevalence": "NOT_ESTIMATED_ONE_SELECTED_STATE",
            "correctness": "UNINSTANTIATED",
            "operator_attribution": "NOT_CLAIMED_SCORER_FORWARD_TREATMENT",
            "long_run_harm": "NOT_CLAIMED",
        },
        "nonclaims": [
            "selected-state transition impact is not population prevalence",
            "implementation-relative impact is not mathematical correctness",
            "scorer-forward intervention is not operator causal attribution",
            "one-step update difference does not imply long-run harm",
        ],
    }
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "construction_valid": construction_valid,
        "transition_verdict": transition_verdict,
        "loss": loss_profile,
        "gradient_state_effect_l2": gradient_profile["state_effect_l2"],
        "update_state_effect_l2": update_profile["state_effect_l2"],
        "update_paired_U1_repeats": update_profile["paired_U1_repeats"],
        "semantic_disagreement": output["semantic_events"],
    }, indent=2))
    if not construction_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
