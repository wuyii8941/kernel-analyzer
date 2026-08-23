#!/usr/bin/env python3
"""Merge frozen Phi carrier shards and render the carrier distribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    documents = [json.loads(path.read_text()) for path in args.shards]
    if any(row.get("status") != "COMPLETE" for row in documents):
        raise RuntimeError("all carrier shards must be complete")
    if any(row.get("selection_sha256") != manifest["selection_sha256"] for row in documents):
        raise RuntimeError("carrier shard manifest mismatch")
    rows = sorted(
        [item for document in documents for item in document["rows"]],
        key=lambda item: int(item["index"]),
    )
    if [row["carrier"] for row in rows] != [row["carrier"] for row in manifest["carriers"]]:
        raise RuntimeError("merged carriers do not exactly match the frozen manifest")
    amplitudes = [float(row["measurement"]["coherence_amplification"]) for row in rows]
    random_by_seed = {}
    for row in rows:
        for item in row.get("random_nulls") or []:
            random_by_seed.setdefault(str(item["seed"]), []).append(
                float(item["coherence_amplification"])
            )
    random_summary = {
        seed: {
            "carrier_count": len(values),
            "minimum_A": min(values),
            "mean_A": sum(values) / len(values),
            "maximum_A": max(values),
        }
        for seed, values in sorted(random_by_seed.items(), key=lambda item: int(item[0]))
    }
    payload = {
        "schema": "kernel-analyzer-phi-carrier-distribution-v1",
        "status": "COMPLETE_FROZEN_12_CARRIER_DISTRIBUTION",
        "manifest": str(args.manifest),
        "selection_sha256": manifest["selection_sha256"],
        "carrier_count": len(rows),
        "summary": {
            "minimum_A": min(amplitudes),
            "median_A": median(amplitudes),
            "maximum_A": max(amplitudes),
            "above_diffusive_one": sum(value > 1.0 for value in amplitudes),
            "above_historical_two": sum(value >= 2.0 for value in amplitudes),
        },
        "random_null_summary": random_summary,
        "rows": rows,
        "timing": {
            "sum_carrier_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "parallel_shard_wall_seconds": max(float(doc["elapsed_seconds"]) for doc in documents),
            "total_f_and_b_calls": sum(int(row["f_and_b_calls"]) for row in rows),
        },
        "claim_boundary": (
            "The distribution covers the 12 outcome-blind carriers in the frozen manifest. "
            "It measures independent one-parameter trajectories, not the percentage of all "
            "model parameters affected during full-parameter training."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distribution.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "distribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "carrier", "stratum", "layer_index", "coordinates",
            "coherence_amplification", "resultant_l2", "diffusive_step_scale",
            "elapsed_seconds",
        ])
        writer.writeheader()
        for row in rows:
            measurement = row["measurement"]
            writer.writerow({
                "index": row["index"], "carrier": row["carrier"],
                "stratum": row["stratum"], "layer_index": row["layer_index"],
                "coordinates": row["coordinates"],
                "coherence_amplification": measurement["coherence_amplification"],
                "resultant_l2": measurement["resultant_l2"],
                "diffusive_step_scale": measurement["diffusive_step_scale"],
                "elapsed_seconds": row["elapsed_seconds"],
            })
    try:
        import matplotlib.pyplot as plt

        labels = [row["carrier"].replace("model.layers.", "L").replace(".weight", "") for row in rows]
        figure, axis = plt.subplots(figsize=(11, 5.2))
        positions = list(range(len(rows)))
        axis.bar(positions, amplitudes, color=[
            "#c44e52" if value >= 2.0 else "#4c72b0" for value in amplitudes
        ])
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="diffusive A=1")
        axis.axhline(2.0, color="gray", linestyle=":", linewidth=1, label="historical A=2")
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        axis.set_ylabel("32-step coherence amplification A")
        axis.set_title("Phi lm_head dX persistence across frozen parameter carriers")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "distribution.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps({"status": payload["status"], **payload["summary"]}))


if __name__ == "__main__":
    main()
