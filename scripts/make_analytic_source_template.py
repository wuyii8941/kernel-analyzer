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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draft_source(row: dict, index: int) -> dict:
    level = str(row.get("level", index))
    variable = str(row.get("variable", "source"))
    path_text = f"{row.get('path_ref', '')} {row.get('path_alt', '')}".lower()
    dtype = "bf16" if "bf16" in path_text else "fp16" if "fp16" in path_text else "fp32"
    return {
        "name": f"analytic_{level}_{variable.replace(' ', '_')}",
        "mechanism": str(row.get("mechanism", "unknown")),
        "dtype": dtype,
        "reduction_length": 1,
        "sum_abs": 0.0,
        "reduction": "tree",
        "materialization_count_delta": 0,
        "local_scale": 0.0,
        "propagation": 1.0,
        "logprob_lipschitz": 1.0 if level == "L4" else 2.0,
        "reduction_path_count": 2,
        "assumptions_verified": False,
        "algorithm_order_known": False,
        "input_norm_measured": False,
        "propagation_certified": False,
        "notes": (
            f"TODO for {level} ({variable}): replace placeholder dimensions/norms with operator-input measurements, "
            "document both algorithms and rounding points, prove propagation, then set validation flags."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hash-bound analytic source draft from Phase 1.5 measurements.")
    parser.add_argument("--measurements", default="results/phase15_measurements.jsonl")
    parser.add_argument("--out", default="results/phase2_sources.analytic_draft.json")
    args = parser.parse_args()

    measurement_path = Path(args.measurements)
    rows = load_rows(measurement_path)
    nonzero = [
        row
        for row in rows
        if max(
            abs(float(row.get("final_logprob_delta", 0.0))),
            abs(float(row.get("max_logprob_delta", 0.0))),
            abs(float(row.get("max_activation_diff_l2", 0.0))),
        )
        > 0.0
    ]
    sources = [draft_source(row, index) for index, row in enumerate(nonzero)]
    level_sources = {
        str(row.get("level")): [source["name"]]
        for row, source in zip(nonzero, sources, strict=True)
    }
    payload = {
        "certificate_kind": "analytic_draft",
        "template_complete": False,
        "coverage": {
            "measurements_sha256": file_sha256(measurement_path),
            "measured_levels": sorted({str(row.get("level")) for row in rows}),
            "level_sources": level_sources,
        },
        "sources": sources,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "draft_sources": len(sources), "levels": sorted(level_sources)}, indent=2))


if __name__ == "__main__":
    main()
