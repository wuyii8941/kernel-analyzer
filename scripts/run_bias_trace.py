#!/usr/bin/env python3
"""Build one candidate-blind BiasTransitionCertificate.

The runner consumes a normalized trace, not raw candidate tensors.  A capture
backend is responsible for computing the declared projections from its exact
F+B replay and for writing the input file.  Missing layers remain in the
certificate as ``ABSTAIN_UNRESOLVED``; this command never imputes a value and
never reads T1--T4/SEUP verdicts.
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

from kernel_analyzer.bias import BiasTrace, BiasTracePolicy  # noqa: E402


FORBIDDEN_KEYS = {
    "candidate_output", "candidate_outputs", "candidate_tensor", "candidate_value",
    "oracle_verdict", "case_verdict", "t1_verdict", "t2_verdict", "t3_verdict",
    "t4_verdict", "seup_verdict", "final_drift_label", "historical_verdict",
}


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trace input must be a JSON object")
    return value


def _find_leaks(value: Any, prefix: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normal = str(key).strip().lower().replace("-", "_")
            path = f"{prefix}.{key}" if prefix else str(key)
            if normal in FORBIDDEN_KEYS or normal.endswith("_verdict"):
                leaks.append(path)
            leaks.extend(_find_leaks(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            leaks.extend(_find_leaks(nested, f"{prefix}[{index}]"))
    return leaks


def _value(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    layers = row.get("layers")
    if isinstance(layers, Mapping):
        return layers.get(name)
    return None


def build(payload: Mapping[str, Any]) -> dict[str, Any]:
    leaks = _find_leaks(payload)
    if leaks:
        raise ValueError("candidate/verdict leakage in trace input: " + ", ".join(leaks[:12]))
    split = payload.get("state_split")
    if not isinstance(split, Mapping):
        raise ValueError("state_split is required")
    calibration = [str(x) for x in split.get("calibration_state_ids", ())]
    evaluation = [str(x) for x in split.get("evaluation_state_ids", ())]
    if not calibration or not evaluation:
        raise ValueError("both calibration and evaluation state IDs are required")
    policy_payload = payload.get("policy", {})
    policy = BiasTracePolicy(
        min_states=int(policy_payload.get("min_states", 4)),
        centred_z_abs_max=float(policy_payload.get("centred_z_abs_max", 2.0)),
        persistence_ratio_min=float(policy_payload.get("persistence_ratio_min", 0.6)),
        mean_norm_floor=float(policy_payload.get("mean_norm_floor", 1e-30)),
    )
    trace = BiasTrace(
        case_id=str(payload.get("case_id", "")),
        calibration_state_ids=calibration,
        evaluation_state_ids=evaluation,
        policy=policy,
    )
    rows = payload.get("rows", ())
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each trace row must be an object")
        trace.add(
            str(row.get("state_id", "")),
            str(row.get("partition", "")),
            local_endpoint=_value(row, "local_endpoint"),
            parameter_gradient=_value(row, "parameter_gradient"),
            effective_update=_value(row, "effective_update"),
            trajectory_drift=_value(row, "trajectory_drift"),
            feedback=_value(row, "feedback"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        )
    result = trace.finalize()
    result["input_schema"] = str(payload.get("schema", "UNDECLARED"))
    result["input_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result["candidate_blind"] = True
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
        "output": str(args.output),
        "case_id": result["case_id"],
        "status": result["status"],
        "first_noncentered_layer": result["first_noncentered_layer"],
        "candidate_blind": result["candidate_blind"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
