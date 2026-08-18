#!/usr/bin/env python3
"""Build a v2.1 open-loop Bias Formation certificate.

This is a post-capture reducer.  It never runs a model and never infers a
formation label from T1--T4 or SEUP.  Capture adapters must provide one
complete local/gradient/update vector for every frozen calibration and
confirmation state, plus the component-wise common-state certificate.
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

from kernel_analyzer.bias_formation_v21 import (  # noqa: E402
    BiasFormationTrace,
    FormationPolicy,
)


FORBIDDEN_KEYS = {
    "candidate_output", "candidate_outputs", "candidate_tensor", "candidate_value",
    "oracle_verdict", "case_verdict", "t1_verdict", "t2_verdict", "t3_verdict",
    "t4_verdict", "seup_verdict", "final_drift_label", "historical_verdict",
    "trajectory_drift", "paired_trajectory_drift", "parameter_drift",
}


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("formation input must be a JSON object")
    return value


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
    aliases = {
        "LOCAL_ENDPOINT": "local_endpoint",
        "PARAMETER_GRADIENT": "parameter_gradient",
        "EFFECTIVE_UPDATE": "effective_update",
    }
    if aliases[name] in row:
        return row[aliases[name]]
    layers = row.get("layers")
    if not isinstance(layers, Mapping):
        return None
    return layers.get(name, layers.get(aliases[name]))


def _policy(payload: Mapping[str, Any]) -> FormationPolicy:
    value = payload.get("policy", {})
    if not isinstance(value, Mapping):
        raise ValueError("policy must be an object")
    # v2.1 margins are frozen in the library.  The payload may repeat them for
    # provenance, but it cannot lower the state/bootstrap requirements.
    policy = FormationPolicy(
        min_states=int(value.get("min_states", 16)),
        centered_margin=float(value.get("centered_margin", 0.20)),
        bias_margin=float(value.get("bias_margin", 0.25)),
        canceling_margin=float(value.get("canceling_margin", 0.20)),
        bootstrap_samples=int(value.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(value.get("bootstrap_seed", 20260818)),
        energy_floor=float(value.get("energy_floor", 1e-30)),
    )
    if policy.min_states < 16 or policy.bootstrap_samples < 2000:
        raise ValueError("v2.1 requires at least 16 states and 2000 bootstrap draws")
    return policy


def build(payload: Mapping[str, Any]) -> dict[str, Any]:
    leaks = _find_leaks(payload)
    if leaks:
        raise ValueError("formation input contains historical/verdict/trajectory fields: " + ", ".join(leaks[:12]))
    split = payload.get("state_split")
    if not isinstance(split, Mapping):
        raise ValueError("state_split is required")
    calibration = [str(x) for x in split.get("calibration_state_ids", ())]
    confirmation = [str(x) for x in split.get("confirmation_state_ids", ())]
    if len(calibration) < 16 or len(confirmation) < 16:
        raise ValueError("v2.1 requires at least 16 calibration and 16 confirmation states")
    policy = _policy(payload)
    trace = BiasFormationTrace(
        str(payload.get("case_id", "")), calibration, confirmation, policy,
    )
    rows = payload.get("rows", ())
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each formation row must be an object")
        if "common_state_digest" in row:
            raise ValueError("component-wise common_state_certificate is required; common_state_digest is obsolete")
        common = row.get("common_state_certificate", row.get("common_state"))
        if common is None:
            raise ValueError("each state needs a component-wise common_state_certificate")
        trace.add(
            str(row.get("state_id", "")),
            str(row.get("partition", "")),
            common_state_certificate=common,
            local_endpoint=_layer(row, "LOCAL_ENDPOINT"),
            parameter_gradient=_layer(row, "PARAMETER_GRADIENT"),
            effective_update=_layer(row, "EFFECTIVE_UPDATE"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        )
    result = trace.finalize()
    result["input_schema"] = str(payload.get("schema", "UNDECLARED"))
    result["input_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result["formation_label_source"] = "v2_1_open_loop_population_certificate"
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
    print(json.dumps({
        "output": str(args.output), "case_id": result["case_id"],
        "status": result["status"], "formation_label_source": result["formation_label_source"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
