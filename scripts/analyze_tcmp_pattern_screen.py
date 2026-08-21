#!/usr/bin/env python3
"""Screen one value-blind representative per implementation pattern.

This is a prioritization screen, not a Flash-style case verdict.  Runtime
invocations remain the coverage denominator; state is the statistical unit.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_implementation_census import _campaign_rows, _identity, _read, _records


def _sign_flip_p(errors: np.ndarray) -> tuple[float, float]:
    observed = float(np.linalg.norm(errors.sum(axis=0)))
    denominator = math.sqrt(float(np.square(errors).sum()))
    amplification = observed / denominator if denominator else 0.0
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(errors)):
        value = np.asarray(signs)[:, None] * errors
        null.append(float(np.linalg.norm(value.sum(axis=0))))
    p_value = sum(value >= observed - 1e-15 for value in null) / len(null)
    return p_value, amplification


def _bh(rows: list[dict[str, Any]], q: float) -> None:
    eligible = sorted(
        (row for row in rows if row["p_value"] is not None),
        key=lambda row: (row["p_value"], row["implementation_pattern_id"], row["endpoint"]),
    )
    cutoff = -1
    for index, row in enumerate(eligible, start=1):
        if row["p_value"] <= q * index / len(eligible):
            cutoff = index
    for index, row in enumerate(eligible, start=1):
        row["screen_positive_bh_q_0_10"] = index <= cutoff


def analyze(paths: list[Path]) -> dict[str, Any]:
    by_exact_endpoint: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    invocations = 0
    unresolved = 0
    legacy: list[tuple[str, dict[str, Any]]] = []
    signature_to_exact: defaultdict[tuple[Any, ...], set[str]] = defaultdict(set)

    def operation_pattern(record: dict[str, Any]) -> str:
        value = str(record.get("symbol") or record.get("operation") or record.get("function") or "UNRESOLVED")
        return re.sub(r"(?:_\d+)+$", "", value)

    def signature(record: dict[str, Any], endpoint: str, metric: dict[str, Any]) -> tuple[Any, ...]:
        sketch = metric.get("directional_error_sketch", {})
        return (
            str(record.get("phase", "UNRESOLVED")), operation_pattern(record), endpoint,
            int(sketch.get("tensor_numel", -1)), tuple(sketch.get("flat_coordinate_indices", [])),
        )

    def append_observation(exact_id: str, state_id: str, record: dict[str, Any]) -> None:
        for endpoint, metric in record.get("endpoint_metrics", {}).items():
            sketch = metric.get("directional_error_sketch", {})
            values = sketch.get("signed_delta_values")
            if values is None:
                continue
            by_exact_endpoint[(exact_id, endpoint)].append({
                "state_id": state_id,
                "values": values,
                "coordinates": sketch.get("flat_coordinate_indices", []),
                "rms": metric.get("rms"),
                "finite": bool(metric.get("candidate_finite", False) and metric.get("reference_finite", False)),
            })
            signature_to_exact[signature(record, endpoint, metric)].add(exact_id)

    for path in paths:
        document = _read(path)
        campaigns = _campaign_rows(path, document)
        for state_id, record in _records(document):
            invocations += 1
            identity = _identity(record, campaigns.get(record.get("region_id"), {}))
            if identity is None:
                unresolved += 1
                legacy.append((state_id, record))
                continue
            exact_id = identity["exact_implementation_id"]
            identities[exact_id] = identity
            # The first repeat is returned by _records; recover both repeats
            # from the state document for state-level averaging below.
            append_observation(exact_id, state_id, record)

    # Old screens are not rerun merely to add ABI metadata.  Bind a legacy
    # observation only when its value-blind operation/phase/endpoint/tensor
    # signature maps to exactly one implementation pattern in the new census.
    rebound = 0
    for state_id, record in legacy:
        bindings = []
        for endpoint, metric in record.get("endpoint_metrics", {}).items():
            exact_ids = signature_to_exact.get(signature(record, endpoint, metric), set())
            patterns = {identities[value]["implementation_pattern_id"] for value in exact_ids}
            if len(patterns) != 1:
                continue
            pattern = next(iter(patterns))
            eligible = [value for value in exact_ids if identities[value]["implementation_pattern_id"] == pattern]
            exact_id = min(
                eligible,
                key=lambda value: (-len({row["state_id"] for row in by_exact_endpoint[(value, endpoint)]}), value),
            )
            sketch = metric["directional_error_sketch"]
            by_exact_endpoint[(exact_id, endpoint)].append({
                "state_id": state_id,
                "values": sketch["signed_delta_values"],
                "coordinates": sketch.get("flat_coordinate_indices", []),
                "rms": metric.get("rms"),
                "finite": bool(metric.get("candidate_finite", False) and metric.get("reference_finite", False)),
            })
            bindings.append(endpoint)
        if bindings:
            rebound += 1
            unresolved -= 1

    # Pick the ABI stratum with the largest state support without reading its
    # numerical values.  Lexicographic exact ID is the deterministic tie-break.
    candidates: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (exact_id, endpoint), observations in by_exact_endpoint.items():
        pattern = identities[exact_id]["implementation_pattern_id"]
        candidates[(pattern, endpoint)].append((exact_id, len({row["state_id"] for row in observations})))

    rows = []
    for (pattern, endpoint), choices in sorted(candidates.items()):
        exact_id, _ = min(choices, key=lambda item: (-item[1], item[0]))
        observations = by_exact_endpoint[(exact_id, endpoint)]
        by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            by_state[observation["state_id"]].append(observation)
        state_errors = []
        finite = True
        coordinate_rule = None
        rms = []
        for state_id in sorted(by_state):
            values = by_state[state_id]
            coordinates = tuple(values[0]["coordinates"])
            if any(tuple(row["coordinates"]) != coordinates for row in values):
                raise RuntimeError(f"coordinate mismatch within state: {pattern}:{endpoint}:{state_id}")
            if coordinate_rule is None:
                coordinate_rule = coordinates
            elif coordinate_rule != coordinates:
                raise RuntimeError(f"coordinate mismatch across states: {pattern}:{endpoint}")
            state_errors.append(np.mean([row["values"] for row in values], axis=0))
            finite &= all(row["finite"] for row in values)
            rms.extend(row["rms"] for row in values if row["rms"] is not None)
        errors = np.asarray(state_errors, dtype=np.float64)
        if not finite or not np.isfinite(errors).all():
            p_value, amplification, verdict = None, None, "NONFINITE_RISK"
        elif len(errors) < 4:
            p_value, amplification, verdict = None, None, "UNRESOLVED_INSUFFICIENT_STATES"
        elif not np.any(errors):
            p_value, amplification, verdict = 1.0, 0.0, "EXACT_ZERO"
        else:
            p_value, amplification = _sign_flip_p(errors)
            verdict = "FINITE_SCREENED"
        identity = identities[exact_id]
        rows.append({
            "implementation_pattern_id": pattern,
            "representative_exact_implementation_id": exact_id,
            "semantic_family_id": identity["semantic_family_id"],
            "operation": identity["pattern_payload"]["operation_pattern"],
            "phase": identity["pattern_payload"]["phase"],
            "endpoint": endpoint,
            "states": len(errors),
            "p_value": p_value,
            "amplification": amplification,
            "rms_mean": float(np.mean(rms)) if rms else None,
            "verdict": verdict,
            "representative_selection": "MAX_STATE_SUPPORT_THEN_LEXICOGRAPHIC_EXACT_ID_VALUE_BLIND",
            "screen_positive_bh_q_0_10": False,
        })
    _bh(rows, 0.10)
    positives = sum(row["screen_positive_bh_q_0_10"] for row in rows)
    return {
        "schema": "kernel-analyzer-tcmp-pattern-screen-v1",
        "status": "COMPLETE" if unresolved == 0 else "PARTIAL_IDENTITY_UNRESOLVED",
        "claim_boundary": "Sampled-coordinate precision screen for deep-measurement prioritization; not a complete-vector bias or persistence verdict.",
        "denominator": {
            "runtime_invocations": invocations,
            "identity_unresolved_invocations": unresolved,
            "legacy_invocations_rebound_value_blind": rebound,
            "pattern_endpoints": len(rows),
            "bh_screen_positive_pattern_endpoints": positives,
        },
        "inference": {"unit": "STATE", "test": "EXACT_SIGN_FLIP", "multiple_testing": "BH_Q_0.10"},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.screens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps(result["denominator"], sort_keys=True))


if __name__ == "__main__":
    main()
