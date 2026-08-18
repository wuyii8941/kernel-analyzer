#!/usr/bin/env python3
"""Build a v2 common-state formation certificate.

This command is deliberately verdict-blind, not candidate-blind: candidate
and repair measurements are required ground truth, while historical labels and
trajectory drift are forbidden.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_formation import BiasFormationTrace, FormationPolicy  # noqa: E402


FORBIDDEN_KEYS = {
    "candidate_output", "candidate_outputs", "oracle_verdict", "case_verdict",
    "t1_verdict", "t2_verdict", "t3_verdict", "t4_verdict", "seup_verdict",
    "historical_verdict", "trajectory_drift", "paired_trajectory_drift",
}


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("formation input must be a JSON object")
    return payload


def _find_leaks(value: Any, prefix: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normal = str(key).strip().lower().replace("-", "_")
            current = f"{prefix}.{key}" if prefix else str(key)
            if normal in FORBIDDEN_KEYS or normal.endswith("_verdict"):
                leaks.append(current)
            leaks.extend(_find_leaks(nested, current))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            leaks.extend(_find_leaks(nested, f"{prefix}[{index}]"))
    return leaks


def _layer(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    layers = row.get("layers")
    return layers.get(name) if isinstance(layers, Mapping) else None


def build(payload: Mapping[str, Any]) -> dict[str, Any]:
    leaks = _find_leaks(payload)
    if leaks:
        raise ValueError("formation input contains historical/verdict/trajectory fields: " + ", ".join(leaks[:12]))
    split = payload.get("state_split")
    if not isinstance(split, Mapping):
        raise ValueError("state_split is required")
    calibration = [str(x) for x in split.get("calibration_state_ids", ())]
    confirmation = [str(x) for x in split.get("confirmation_state_ids", split.get("evaluation_state_ids", ()))]
    if not calibration or not confirmation:
        raise ValueError("calibration_state_ids and confirmation_state_ids are required")
    policy_payload = payload.get("policy", {})
    if not isinstance(policy_payload, Mapping):
        raise ValueError("policy must be an object")
    policy = FormationPolicy(
        min_states=int(policy_payload.get("min_states", 4)),
        centered_ratio_upper=float(policy_payload.get("centered_ratio_upper", 0.01)),
        biased_ratio_lower=float(policy_payload.get("biased_ratio_lower", 0.05)),
        bootstrap_samples=int(policy_payload.get("bootstrap_samples", 256)),
        bootstrap_seed=int(policy_payload.get("bootstrap_seed", 20260818)),
        energy_floor=float(policy_payload.get("energy_floor", 1e-30)),
    )
    trace = BiasFormationTrace(str(payload.get("case_id", "")), calibration, confirmation, policy)
    rows = payload.get("rows", ())
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each formation row must be an object")
        trace.add(
            str(row.get("state_id", "")),
            str(row.get("partition", "")),
            common_state_digest=row.get("common_state_digest"),
            local_endpoint=_layer(row, "LOCAL_ENDPOINT") if "LOCAL_ENDPOINT" in row or "layers" in row else _layer(row, "local_endpoint"),
            parameter_gradient=_layer(row, "PARAMETER_GRADIENT") if "PARAMETER_GRADIENT" in row or "layers" in row else _layer(row, "parameter_gradient"),
            effective_update=_layer(row, "EFFECTIVE_UPDATE") if "EFFECTIVE_UPDATE" in row or "layers" in row else _layer(row, "effective_update"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        )
    result = trace.finalize()
    result["input_schema"] = str(payload.get("schema", "UNDECLARED"))
    result["input_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(_load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    opener = gzip.open if args.output.name.endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "case_id": result["case_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
