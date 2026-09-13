#!/usr/bin/env python3
"""Freeze campaigns for unmeasured Triton implementation signatures.

This is the coverage layer between "one position per broad operator family"
and "every observed position".  It selects at most one canonical position for
each previously unmeasured

    operator family x phase x Triton symbol signature x reference method

without reading numerical outcomes.  A signature is excluded only when an
existing catalogue row or an explicitly supplied prior campaign proves a
valid measurement.  The produced manifest is compatible with
``run_family_first_campaigns.py`` so capture, verification, failure recording,
and reporting continue to use the existing implementation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_family_first_campaign_manifest import (
    GENERIC_METHODS,
    SPECIALIZED_METHODS,
    discover_runtime_configs,
)


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _methods(row: dict[str, Any], include_specialized: bool) -> list[str]:
    allowed = set(GENERIC_METHODS)
    if include_specialized:
        allowed.update(SPECIALIZED_METHODS)
    declared = {
        str(item.get("reference_method"))
        for item in row.get("reference_candidates", [])
        if item.get("reference_method")
    }
    return sorted(declared & allowed)


def _signature_key(row: dict[str, Any], method: str) -> tuple[str, ...]:
    phase = row.get("phase")
    if not phase:
        task_id = str(row.get("task_id", ""))
        prefix = task_id.split(":", 1)[0].upper()
        phase = prefix if prefix in {"FORWARD", "BACKWARD"} else "UNDECLARED"
    return (
        str(row.get("operator_family", "UNDECLARED")),
        str(row.get("implementation_kind", "UNDECLARED")),
        str(phase),
        str(row.get("symbol_signature") or row.get("symbol") or "UNDECLARED"),
        method,
    )


def _valid_prior_tasks(manifests: list[Path]) -> set[tuple[str, str]]:
    """Return only tasks with authoritative valid outputs in prior campaigns."""
    result: set[tuple[str, str]] = set()
    for manifest_path in manifests:
        payload = _read(manifest_path)
        for campaign in payload.get("campaigns", []):
            # A repaired runtime release may preserve the original frozen task
            # package while changing only executable paths.  Map validity back
            # to the original catalogue release so that a successful retry is
            # not scheduled again as a supposedly unseen signature.
            release = str(Path(campaign.get("original_release", campaign["release"])).resolve())
            task_id = str(campaign["task_id"])
            output = Path(campaign["campaign_output"])
            adapter = campaign.get("adapter")
            if adapter not in {None, "GENERIC_COVERAGE"}:
                completion = output / "completion_verification.json"
                if not completion.exists():
                    continue
                records = _read(completion).get("records", [])
                valid = any(
                    str(row.get("task_id")) == task_id
                    and row.get("status") in {"VERIFIED", "RECORDED_MEASUREMENT_CHECKED"}
                    for row in records
                )
            else:
                run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
                status = output / "runs" / run_id / "status.json"
                valid = status.exists() and _read(status).get("status") == "VALID"
            if valid:
                result.add((release, task_id))
    return result


def build(
    catalog: dict[str, Any],
    queue: dict[str, Any],
    runtime_configs: dict[str, dict[str, Any]],
    *,
    output_root: Path,
    include_specialized: bool = True,
    prior_valid_tasks: set[tuple[str, str]] | None = None,
    max_campaigns: int | None = None,
) -> dict[str, Any]:
    prior_valid_tasks = prior_valid_tasks or set()
    catalog_rows = catalog.get("positions", [])
    catalog_by_identity = {
        (str(Path(row["release"]).resolve()), str(row["task_id"])): row
        for row in catalog_rows
    }

    measured_keys: set[tuple[str, ...]] = set()
    for row in catalog_rows:
        if row.get("implementation_kind") != "TRITON":
            continue
        if row.get("support_status") != "VALID_MEASUREMENT_COMPLETED":
            continue
        methods = _methods(row, include_specialized)
        # Historical rows may not retain a reference method.  They still prove
        # that this structural signature has been measured, so use a wildcard.
        if methods:
            measured_keys.update(_signature_key(row, method) for method in methods)
        else:
            base = _signature_key(row, "*")
            measured_keys.add(base)

    for identity in prior_valid_tasks:
        row = catalog_by_identity.get(identity)
        if row is None or row.get("implementation_kind") != "TRITON":
            continue
        methods = _methods(row, include_specialized)
        if methods:
            measured_keys.update(_signature_key(row, method) for method in methods)
        else:
            measured_keys.add(_signature_key(row, "*"))

    def already_measured(key: tuple[str, ...]) -> bool:
        return key in measured_keys or (*key[:-1], "*") in measured_keys

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for row in queue.get("rows", []):
        if row.get("implementation_kind") != "TRITON":
            continue
        methods = _methods(row, include_specialized)
        if len(methods) != 1:
            skipped.append({
                "release": row.get("release"),
                "task_id": row.get("task_id"),
                "operator_family": row.get("operator_family"),
                "reason": "NO_UNIQUE_SUPPORTED_REFERENCE_METHOD",
                "declared_methods": methods,
            })
            continue
        method = methods[0]
        key = _signature_key(row, method)
        if key in seen_keys or already_measured(key):
            continue
        release = str(Path(row["release"]).resolve())
        runtime = runtime_configs.get(release)
        if runtime is None:
            skipped.append({
                "release": release,
                "task_id": row.get("task_id"),
                "operator_family": row.get("operator_family"),
                "signature_key": list(key),
                "reason": "RUNTIME_CONFIG_NOT_UNIQUELY_RECOVERED",
            })
            seen_keys.add(key)
            continue
        binding = next(
            item for item in row.get("reference_candidates", [])
            if item.get("reference_method") == method
        )
        specialized = SPECIALIZED_METHODS.get(method)
        specialized_plan = binding.get("bound_plan") if specialized else None
        if specialized and (not specialized_plan or not Path(specialized_plan).exists()):
            skipped.append({
                "release": release,
                "task_id": row.get("task_id"),
                "operator_family": row.get("operator_family"),
                "signature_key": list(key),
                "reason": "SPECIALIZED_PLAN_MISSING",
            })
            seen_keys.add(key)
            continue
        case_hash = hashlib.sha256((release + "\0" + str(row["task_id"])).encode()).hexdigest()[:16]
        case_id = "signature-" + str(row["operator_family"]).lower().replace("_", "-") + "-" + case_hash
        campaign = {
            "operator_family": row["operator_family"],
            "implementation_kind": "TRITON",
            "phase": key[2],
            "symbol": row.get("symbol"),
            "symbol_signature": row.get("symbol_signature"),
            "signature_key": list(key),
            "release": release,
            "original_release": release,
            "task_id": row["task_id"],
            "case": {
                "case_id": case_id,
                "task_id": row["task_id"],
                "carrier": row.get("carrier"),
                "reference_method": method,
                "family": row["operator_family"],
                "selection_rule": "FIRST_UNMEASURED_TRITON_STRUCTURAL_SIGNATURE_NO_NUMERICAL_OUTCOMES",
            },
            "adapter": specialized["adapter"] if specialized else "GENERIC_COVERAGE",
            "capture_script": specialized["script"] if specialized else None,
            "specialized_plan_argument": specialized["plan_argument"] if specialized else None,
            "specialized_plan": specialized_plan,
            "runtime": dict(runtime),
            "campaign_output": str((output_root / case_id).resolve()),
        }
        selected.append(campaign)
        seen_keys.add(key)
        if max_campaigns is not None and len(selected) >= max_campaigns:
            break

    family_counts: dict[str, int] = {}
    for campaign in selected:
        family = str(campaign["operator_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "schema": "triton-unmeasured-signature-campaign-manifest-v1",
        "scope": "ONE_CANONICAL_POSITION_PER_PREVIOUSLY_UNMEASURED_TRITON_STRUCTURAL_SIGNATURE",
        "selection_uses_numerical_outcomes": False,
        "signature_definition": "operator_family + implementation_kind + phase + symbol_signature + reference_method",
        "catalog_valid_measurements_used_only_for_exclusion": True,
        "prior_campaigns_used_only_for_valid_measurement_exclusion": True,
        "selected_campaign_count": len(selected),
        "selected_operator_family_counts": dict(sorted(family_counts.items())),
        "measured_signature_count_before_selection": len(measured_keys),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "campaigns": selected,
        "execution_policy": {
            "automatic_continuation_to_next_independent_signature_after_failure": True,
            "automatic_retry_or_replacement_after_failure": False,
            "overwrite_existing_output": False,
            "partial_results_are_not_measurements": True,
            "same_signature_is_not_repeated_across_models_or_positions": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-campaigns", type=int)
    parser.add_argument("--exclude-specialized", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() or not args.output_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    if args.max_campaigns is not None and args.max_campaigns < 1:
        parser.error("--max-campaigns must be positive")
    catalog = _read(args.catalog)
    queue = _read(args.queue)
    protocols = sorted(args.protocol_root.glob("**/protocol.json"))
    runtime_configs = discover_runtime_configs(protocols)
    prior_valid = _valid_prior_tasks(args.prior_manifest)
    result = build(
        catalog,
        queue,
        runtime_configs,
        output_root=args.output_root,
        include_specialized=not args.exclude_specialized,
        prior_valid_tasks=prior_valid,
        max_campaigns=args.max_campaigns,
    )
    result["input_sha256"] = {
        "catalog": _sha(args.catalog),
        "queue": _sha(args.queue),
        "prior_manifests": {str(path.resolve()): _sha(path) for path in args.prior_manifest},
    }
    result["runtime_config_protocol_count"] = len(protocols)
    args.output_root.mkdir(parents=True)
    for campaign in result["campaigns"]:
        plan_path = args.output_root / (campaign["case"]["case_id"] + ".plan.json")
        if campaign.get("specialized_plan"):
            source = _read(Path(campaign["specialized_plan"]))
            matches = [
                case for case in source.get("cases", [])
                if str(case.get("task_id")) == str(campaign["task_id"])
            ]
            if len(matches) != 1:
                raise SystemExit("Specialized plan must contain exactly one selected task: " + str(campaign["task_id"]))
            plan = {"cases": matches}
        else:
            plan = {"cases": [campaign["case"]]}
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        campaign["case_plan"] = str(plan_path.resolve())
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "selected_campaign_count": result["selected_campaign_count"],
        "selected_operator_family_counts": result["selected_operator_family_counts"],
        "skipped_count": result["skipped_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
