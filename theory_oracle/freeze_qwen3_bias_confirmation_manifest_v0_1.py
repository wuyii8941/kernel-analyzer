#!/usr/bin/env python
"""Freeze a Qwen3 Bias Oracle confirmation manifest before source collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (  # noqa: E402
    ANALYSIS_CODE_PATHS,
    validate_confirmation_manifest,
)


PRECISION_VERSION = "forkcert.bias-oracle-confirmation-precision.v0.1"
BANK_VERSION = "forkcert.qwen3-bias-oracle-confirmation-bank.v0.1"
MANIFEST_VERSION = "forkcert.qwen3-bias-oracle-confirmation-manifest.v0.1"
CALIBRATION_EXCLUSION = {
    "trajectory_ids": [f"calibration-{index}" for index in range(4)],
    "trajectory_seeds": [2001284755, 1810598814, 1677250702, 797459759],
    "data_slice_ids": [
        "forkcert_builtin_arithmetic[7296:7360]",
        "forkcert_builtin_arithmetic[3840:3904]",
        "forkcert_builtin_arithmetic[5696:5760]",
        "forkcert_builtin_arithmetic[3200:3264]",
    ],
}
OUTCOME_MARKERS = (
    "source_dump.jsonl",
    "source_dump.metadata.json",
    "capture_batch_audit.json",
    "remaining_chain_ledger.json",
    "trajectory_scalar_summary.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_outcome_paths(bank: dict[str, Any]) -> list[Path]:
    found: list[Path] = []
    for row in bank.get("trajectory_specs", []):
        root = Path(row["results_root"])
        for name in OUTCOME_MARKERS:
            path = root / name
            if path.exists():
                found.append(path)
        if root.is_dir() and any(root.glob("step*/record_validation.json")):
            found.extend(root.glob("step*/record_validation.json"))
    return found


def build_manifest(
    precision: dict[str, Any],
    precision_path: Path,
    bank: dict[str, Any],
    bank_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if (
        precision.get("schema_version") != PRECISION_VERSION
        or precision.get("valid") is not True
        or precision.get("verdict") != "VALID_FROZEN_PRECISION_PLAN"
    ):
        errors.append("precision plan is not valid/frozen")
    if (
        bank.get("schema_version") != BANK_VERSION
        or bank.get("valid") is not True
        or bank.get("verdict") != "VALID_FROZEN_CONFIRMATION_TRAJECTORY_BANK"
    ):
        errors.append("trajectory bank is not valid/frozen")
    count = precision.get("planned_confirmation_trajectories")
    if not isinstance(count, int) or count < 8:
        errors.append("precision plan must contain J >= 8")
    bank_precision = bank.get("precision", {})
    if (
        bank_precision.get("sha256") != sha256_file(precision_path)
        or bank_precision.get("planned_confirmation_trajectories") != count
    ):
        errors.append("trajectory bank is not bound to this precision plan")
    rows = bank.get("trajectory_specs")
    if not isinstance(rows, list) or len(rows) != count:
        errors.append("trajectory bank size does not equal precision J")
        rows = []
    outcome_paths = confirmation_outcome_paths(bank)
    if outcome_paths:
        errors.append(
            "confirmation outcome exists before manifest freeze: "
            + ", ".join(str(path) for path in outcome_paths)
        )
    multiplicity = precision.get("multiplicity", {})
    endpoint_family = multiplicity.get("endpoint_family")
    if not isinstance(endpoint_family, list) or not endpoint_family:
        errors.append("precision endpoint family is empty")
    if errors:
        raise ValueError("; ".join(errors))

    evaluator = (
        ROOT / "theory_oracle" / "evaluate_qwen3_bias_oracle_confirmation_v0_1.py"
    )
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "query_id": "Q-R",
        "trajectory_anchor": "EAGER_TRAJECTORY",
        "precision_plan": {
            "path": str(precision_path),
            "sha256": sha256_file(precision_path),
            "planned_confirmation_trajectories": count,
        },
        "trajectory_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "analysis_code": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in ANALYSIS_CODE_PATHS.items()
        },
        "calibration_exclusion": CALIBRATION_EXCLUSION,
        "trajectory_inputs": rows,
        "required_trajectory_input_fields": [
            "trajectory_id",
            "trajectory_seed",
            "data_slice_id",
            "source_config_path",
            "source_config_sha256",
            "capture_plan_path",
            "capture_plan_sha256",
            "results_root",
            "data_root",
        ],
        "state_design": {
            "phases": ["early", "middle", "late"],
            "states_per_phase": 8,
            "states_per_trajectory": 24,
            "paired_repeats_per_state": 2,
            "weighting": "equal trajectory; equal phase within trajectory; equal state within phase",
        },
        "analysis": {
            "endpoint_family": endpoint_family,
            "phase_conditioned_endpoint_family": multiplicity.get(
                "phase_conditioned_endpoint_family"
            ),
            "confirmatory_comparisons": multiplicity.get(
                "confirmatory_comparisons"
            ),
            "multiplicity": multiplicity.get("method"),
            "per_interval_alpha": multiplicity.get("per_interval_alpha"),
            "tail_scope": precision.get("tail", {}).get("scope"),
            "sensitivity": precision.get("sensitivity"),
            "calibration_trajectories_in_confirmation_df": False,
            "correctness_authority": None,
        },
        "prospective_freeze_evidence": {
            "confirmation_outcome_markers_observed": [],
            "calibration_mean_or_sign_used_for_trajectory_selection": False,
            "trajectory_count_source": "frozen precision plan",
        },
        "freeze_rule": "this artifact and every linked hash must remain unchanged after the first confirmation source trajectory starts",
    }


def write_frozen(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing frozen manifest differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    precision_path = Path(args.precision).resolve()
    bank_path = Path(args.bank).resolve()
    out = Path(args.out).resolve()
    try:
        manifest = build_manifest(
            load_json(precision_path),
            precision_path,
            load_json(bank_path),
            bank_path,
        )
        _, inputs, errors = validate_confirmation_manifest(manifest, out)
        if errors:
            raise ValueError("manifest self-validation failed: " + "; ".join(errors))
        if len(inputs) != manifest["precision_plan"]["planned_confirmation_trajectories"]:
            raise ValueError("manifest self-validation changed trajectory count")
        write_frozen(out, manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "verdict": "UNINSTANTIATED_OR_INVALID_CONFIRMATION_MANIFEST",
                    "errors": [str(error)],
                },
                indent=2,
            )
        )
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "valid": True,
                "verdict": "VALID_FROZEN_CONFIRMATION_MANIFEST",
                "trajectories": len(manifest["trajectory_inputs"]),
                "manifest_sha256": sha256_file(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
