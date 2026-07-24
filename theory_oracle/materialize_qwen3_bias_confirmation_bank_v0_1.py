#!/usr/bin/env python
"""Materialize a confirmation trajectory bank from a frozen design and J only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DESIGN_VERSION = "forkcert.qwen3-bias-oracle-confirmation-bank-design.v0.1"
PRECISION_VERSION = "forkcert.bias-oracle-confirmation-precision.v0.1"
BANK_VERSION = "forkcert.qwen3-bias-oracle-confirmation-bank.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha_rank(key: str, *parts: object) -> str:
    payload = "/".join([key, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eligible_offsets(design: dict[str, Any]) -> list[int]:
    population = design["dataset_block_population"]
    start = int(population["start_offset_inclusive"])
    stop = int(population["stop_offset_inclusive"])
    block = int(population["block_size"])
    if start < 0 or stop < start or block <= 0 or (stop - start) % block:
        raise ValueError("invalid dataset block population")
    offsets = list(range(start, stop + 1, block))
    excluded = {int(value) for value in population["calibration_offsets_excluded"]}
    if not excluded.issubset(offsets):
        raise ValueError("calibration offset exclusion is outside eligible population")
    return [value for value in offsets if value not in excluded]


def ranked_offsets(design: dict[str, Any]) -> list[int]:
    key = str(design["selection_key_sha256"])
    return sorted(
        eligible_offsets(design),
        key=lambda value: (sha_rank(key, "dataset-offset", value), value),
    )


def trajectory_seeds(design: dict[str, Any], count: int) -> list[int]:
    key = str(design["selection_key_sha256"])
    excluded = {int(value) for value in design["calibration_seeds_excluded"]}
    observed = set(excluded)
    seeds: list[int] = []
    for index in range(count):
        nonce = 0
        while True:
            digest = sha_rank(key, "trajectory-seed", index, nonce)
            seed = int(digest[:16], 16) % (2**31 - 1)
            if seed != 0 and seed not in observed:
                observed.add(seed)
                seeds.append(seed)
                break
            nonce += 1
            if nonce > 1000:
                raise RuntimeError("could not derive a unique trajectory seed")
    return seeds


def selected_steps(
    design: dict[str, Any], trajectory_id: str
) -> dict[str, list[int]]:
    selection = design["state_selection"]
    namespace = str(selection["namespace"])
    count = int(selection["states_per_phase"])
    result: dict[str, list[int]] = {}
    for phase, bounds in selection["phases"].items():
        start, stop = (int(bounds[0]), int(bounds[1]))
        candidates = range(start, stop + 1)
        ranked = sorted(
            candidates,
            key=lambda step: (
                sha_rank(namespace, trajectory_id, phase, step),
                step,
            ),
        )
        result[str(phase)] = sorted(ranked[:count])
    return result


def validate_design(
    design: dict[str, Any], design_path: Path, precision: dict[str, Any]
) -> tuple[Path, int]:
    errors: list[str] = []
    if design.get("schema_version") != DESIGN_VERSION:
        errors.append("unsupported confirmation bank design")
    if design.get("status") != "FROZEN_BEFORE_COMPLETE_CALIBRATION_RESULTS":
        errors.append("confirmation bank design was not frozen prospectively")
    if design.get("query_id") != "Q-R" or design.get("trajectory_anchor") != "EAGER_TRAJECTORY":
        errors.append("confirmation bank has wrong query or trajectory anchor")
    key = design.get("selection_key_sha256")
    if not isinstance(key, str) or len(key) != 64:
        errors.append("selection key must be a frozen sha256 value")
    if precision.get("schema_version") != PRECISION_VERSION:
        errors.append("unsupported precision plan")
    if precision.get("valid") is not True or precision.get("verdict") != "VALID_FROZEN_PRECISION_PLAN":
        errors.append("precision plan is not valid/frozen")
    count = precision.get("planned_confirmation_trajectories")
    if not isinstance(count, int) or count < 8:
        errors.append("precision plan must require at least eight trajectories")
        count = 0
    try:
        offsets = eligible_offsets(design)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        offsets = []
    if count > len(offsets):
        errors.append("planned trajectory count exceeds the frozen block population")
    base_link = design.get("base_config", {})
    try:
        base_path = resolve_from(design_path.parent, str(base_link["path"]))
    except (KeyError, TypeError):
        base_path = Path()
        errors.append("base config link is missing")
    if not base_path.is_file():
        errors.append("base config is missing")
    elif sha256_file(base_path) != base_link.get("sha256"):
        errors.append("base config hash mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    return base_path, count


def build_capture_plan(
    design: dict[str, Any],
    trajectory_id: str,
    seed: int,
    offset: int,
    capture_root: Path,
) -> dict[str, Any]:
    phase_population = {
        "early": "1:100",
        "middle": "101:200",
        "late": "201:300",
    }
    steps = selected_steps(design, trajectory_id)
    targets = []
    for phase in ("early", "middle", "late"):
        for step in steps[phase]:
            targets.append(
                {
                    "optimizer_step": step,
                    "state_id": f"{trajectory_id}-{phase}-step{step:03d}",
                    "phase": phase,
                    "eligible_step_population": phase_population[phase],
                    "relative_dir": f"{phase}/step_{step:06d}",
                    "history_selection": "EVERY_OPTIMIZER_PRE_STEP",
                }
            )
    return {
        "schema_version": "forkcert.multi-transition-capture-plan.v0.1",
        "purpose": "Independent Qwen3 Bias Oracle confirmation trajectory; population inference only through the separately frozen manifest.",
        "capture_root": str(capture_root.resolve()),
        "identity": {
            "query_id": "Q-R",
            "trajectory_id": trajectory_id,
            "trajectory_anchor": "EAGER_TRAJECTORY",
            "trajectory_seed": seed,
            "data_slice_id": f"forkcert_builtin_arithmetic[{offset}:{offset + 64}]",
            "state_selection_prng_seed": design["state_selection"]["namespace"],
        },
        "targets": targets,
        "nonclaims": [
            "a confirmation input plan is not evidence of implementation bias",
            "eager is a baseline and is not asserted to be mathematical truth",
        ],
    }


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    design_path = Path(args.design).resolve()
    precision_path = Path(args.precision).resolve()
    design = load_json(design_path)
    precision = load_json(precision_path)
    try:
        base_path, count = validate_design(design, design_path, precision)
    except ValueError as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "verdict": "UNINSTANTIATED_OR_INVALID_CONFIRMATION_BANK",
                    "errors": [str(error)],
                },
                indent=2,
            )
        )
        raise SystemExit(2) from None
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    offsets = ranked_offsets(design)[:count]
    seeds = trajectory_seeds(design, count)
    config_dir = Path(args.config_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    results_root = Path(args.results_root).resolve()
    data_root = Path(args.data_root).resolve()
    rows: list[dict[str, Any]] = []
    for index, (offset, seed) in enumerate(zip(offsets, seeds, strict=True)):
        trajectory_id = f"confirmation-v0-{index:03d}"
        stem = f"qwen3_bias_oracle_{trajectory_id.replace('-', '_')}"
        config = json.loads(json.dumps(base_config))
        config["dataset"]["offset"] = offset
        config["dataset"]["max_prompts"] = 64
        config["training"]["seed"] = seed
        config_path = config_dir / f"{stem}.yaml"
        plan_path = plan_dir / f"{stem}_capture_plan.json"
        trajectory_results = results_root / trajectory_id
        trajectory_data = data_root / trajectory_id
        plan = build_capture_plan(
            design,
            trajectory_id,
            seed,
            offset,
            trajectory_data / "captures",
        )
        config_content = yaml.safe_dump(config, sort_keys=False)
        plan_content = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        write_frozen(config_path, config_content)
        write_frozen(plan_path, plan_content)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_seed": seed,
                "data_slice_id": plan["identity"]["data_slice_id"],
                "source_config_path": str(config_path),
                "source_config_sha256": sha256_file(config_path),
                "capture_plan_path": str(plan_path),
                "capture_plan_sha256": sha256_file(plan_path),
                "results_root": str(trajectory_results),
                "data_root": str(trajectory_data),
            }
        )
    payload = {
        "schema_version": BANK_VERSION,
        "valid": True,
        "verdict": "VALID_FROZEN_CONFIRMATION_TRAJECTORY_BANK",
        "design": {"path": str(design_path), "sha256": sha256_file(design_path)},
        "precision": {
            "path": str(precision_path),
            "sha256": sha256_file(precision_path),
            "planned_confirmation_trajectories": count,
        },
        "trajectory_specs": rows,
        "selection_audit": {
            "eligible_blocks_after_calibration_exclusion": len(eligible_offsets(design)),
            "selected_offsets_in_order": offsets,
            "calibration_mean_or_sign_read_by_builder": False,
            "confirmation_outcome_read_by_builder": False,
        },
        "population_B_claim_allowed_before_complete_evaluation": False,
    }
    out = Path(args.out).resolve()
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_frozen(out, content)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "trajectories": len(rows),
                "bank_sha256": sha256_file(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
