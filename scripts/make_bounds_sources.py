#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"] if isinstance(payload, dict) else payload


def source_from_measurement(row: dict, index: int) -> dict:
    mechanism = str(row.get("mechanism", "unknown"))
    path_text = (str(row.get("path_ref", "")) + str(row.get("path_alt", ""))).lower()
    dtype = "bf16" if "bf16" in path_text else "fp16" if "fp16" in path_text else "fp32"
    reduction_length = 128000 if "log_softmax" in str(row.get("variable", "")) else 4096
    scale = max(abs(float(row.get("max_activation_diff_l2", 0.0))), abs(float(row.get("final_logprob_delta", 0.0))), 1e-12)
    propagation = max(float(row.get("propagation_gain_first_to_last") or 0.0), 1.0)
    return {
        "name": f"phase15_{row.get('level', index)}_{str(row.get('variable', 'source')).replace(' ', '_')}",
        "mechanism": mechanism,
        "dtype": dtype,
        "reduction_length": reduction_length,
        "sum_abs": scale * reduction_length,
        "reduction": "tree",
        "materialization_count_delta": 1 if mechanism in {"materialization_points", "rounding_precision", "mixed"} else 0,
        "local_scale": scale,
        "propagation": propagation,
        "logprob_lipschitz": 1.0 if str(row.get("level")) == "L4" else 2.0,
        "notes": "empirical heuristic inferred from observed deltas; not a legal rounding-error certificate",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coarse Phase 2 ErrorSource JSON from Phase 1.5 measurements.")
    parser.add_argument("--measurements", default="results/phase15_measurements.jsonl")
    parser.add_argument("--out", default="results/phase2_sources.initial.json")
    parser.add_argument("--top-k", type=int, default=None, help="Optional cap. By default all measured sources are included to avoid under-bounding.")
    args = parser.parse_args()

    all_rows = sorted(load_rows(Path(args.measurements)), key=lambda row: abs(float(row.get("final_logprob_delta", 0.0))), reverse=True)
    rows = all_rows[: args.top_k] if args.top_k is not None else all_rows
    omitted = all_rows[len(rows) :]
    sources = [source_from_measurement(row, i) for i, row in enumerate(rows)]
    level_sources: dict[str, list[str]] = {}
    for row, source in zip(rows, sources):
        level_sources.setdefault(str(row.get("level")), []).append(source["name"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_type": "empirical_heuristic_from_phase15",
        "certificate_kind": "empirical_heuristic",
        "coverage": {
            "measurements_sha256": file_sha256(Path(args.measurements)),
            "measured_levels": sorted({str(row.get("level")) for row in all_rows}),
            "level_sources": level_sources,
        },
        "selection": {
            "top_k": args.top_k,
            "measured_count": len(all_rows),
            "selected_count": len(rows),
            "omitted_count": len(omitted),
            "omitted_levels": [row.get("level") for row in omitted],
        },
        "sources": sources,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
