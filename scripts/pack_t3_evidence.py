#!/usr/bin/env python3
"""Stream complete carrier vectors into compact T3 Gram certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.statistics import coherence_certificate_from_gram
from scripts.evolving_region_intervention_batch import arm_file_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--vectors-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    import torch

    design = json.loads(args.design.read_text())
    queue = json.loads(args.queue.read_text())
    states = design["confirmation_states"]
    unit_regions = {}
    for unit_id in queue["selected_proof_unit_ids"]:
        unit_regions[unit_id] = sorted(
            row["region_id"] for row in queue["rows"]
            if unit_id in row["proof_unit_ids"] and row["selected_for_batch"]
        )
    region_certificates = {}
    vector_sources = []
    for region_id, declaration in design["regions"].items():
        parameter_names = list(declaration["carrier_parameters"])
        vectors = []
        coordinate_count = None
        for index, state in enumerate(states):
            path = args.vectors_root / ("c%d" % index) / arm_file_name(region_id)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("state_id") != state["state_id"] or not payload.get("repeat_exact"):
                raise ValueError("state/repeat mismatch: %s" % path)
            parts = [payload["deltas"][name].reshape(-1).double() for name in parameter_names]
            vector = torch.cat(parts) if len(parts) > 1 else parts[0]
            coordinate_count = int(vector.numel())
            vectors.append(vector)
            vector_sources.append({
                "state_id": state["state_id"], "region_id": region_id,
                "bytes": path.stat().st_size, "sha256": file_sha256(path),
            })
            del payload, parts
        gram = [[float(torch.dot(left, right)) for right in vectors] for left in vectors]
        alpha = 0.05 / max(1, len(queue["selected_proof_unit_ids"]))
        certificate = coherence_certificate_from_gram(
            gram, coordinate_count=coordinate_count, alpha=alpha,
            bootstrap_samples=args.bootstrap_samples, seed=0,
        )
        region_certificates[region_id] = {
            "carrier_parameters": parameter_names,
            "gram": gram,
            "certificate": certificate,
        }
        del vectors

    unit_rows = {}
    for unit_id, region_ids in unit_regions.items():
        rows = [region_certificates[value] for value in region_ids]
        if len(rows) != 1:
            unit_rows[unit_id] = {"status": "UNRESOLVED_MULTI_REGION_T3_COMPOSITION"}
            continue
        row = rows[0]
        unit_rows[unit_id] = {
            "status": "PASS" if row["certificate"]["status"] == "PASS" else "FAIL",
            "complete_coordinates": True,
            "independent_states": True,
            "repeat_exact": True,
            "state_ids": [value["state_id"] for value in states],
            "pilot_state_ids": list(design["pilot_state_ids"]),
            "streaming_complete_gram": True,
            "precomputed_certificate": row["certificate"],
            "carrier_parameters": row["carrier_parameters"],
            "candidate_region_ids": region_ids,
            "natural": True,
        }
    output = {
        "schema": "kernel-analyzer-t3-complete-carrier-evidence-v1",
        "design": str(args.design),
        "region_certificates": region_certificates,
        "unit_rows": unit_rows,
        "raw_vector_sources": vector_sources,
        "raw_vector_retention": "DELETE_AFTER_COMPACT_GRAM_AND_HASH",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(
        output, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "units": {
        key: value["status"] for key, value in unit_rows.items()
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
