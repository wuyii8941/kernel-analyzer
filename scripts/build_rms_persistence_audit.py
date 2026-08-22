#!/usr/bin/env python3
"""Compare residual magnitude with formation directionality and live persistence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HOTSPOT = ROOT / "results/property/bias_formation/hotspot_search"
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def correlations(rows: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    finite = [row for row in rows if row.get("local_rms", 0) > 0 and math.isfinite(float(row[y_key]))]
    x = [math.log10(float(row["local_rms"])) for row in finite]
    y = [float(row[y_key]) for row in finite]
    return {
        "count": len(finite),
        "pearson_log10_rms": pearson(x, y),
        "spearman_log10_rms": pearson(ranks(x), ranks(y)),
    }


def find_population_rows(task_ids: set[str]) -> dict[str, tuple[dict[str, Any], Path]]:
    found: dict[str, tuple[dict[str, Any], Path]] = {}
    roots = [
        HOTSPOT / "equivalence_confirmation",
        HOTSPOT / "multishape_confirmation",
        HOTSPOT / "semantic_region_formation",
        HOTSPOT / "multishape_screen",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                data = load(path)
            except (json.JSONDecodeError, OSError):
                continue
            entries = [data]
            entries.extend(data.get("cases", []) if isinstance(data.get("cases"), list) else [])
            for entry in entries:
                task_id = str(entry.get("binding", {}).get("task_id", entry.get("task_id", "")))
                if task_id not in task_ids:
                    continue
                populations = entry.get("populations", {})
                if not populations and isinstance(entry.get("layers"), dict):
                    populations = {"screening": entry["layers"]}
                partition = "confirmation" if "confirmation" in populations else "screening"
                local = populations.get(partition, {}).get("LOCAL_ENDPOINT")
                if not isinstance(local, dict) or int(local.get("state_count", 0)) < 4:
                    continue
                normalized = dict(entry)
                normalized["populations"] = populations
                previous = found.get(task_id)
                previous_populations = previous[0].get("populations", {}) if previous else {}
                previous_partition = (
                    "confirmation" if "confirmation" in previous_populations else "screening"
                )
                previous_count = int(
                    previous_populations.get(previous_partition, {}).get(
                        "LOCAL_ENDPOINT", {}
                    ).get("state_count", 0)
                )
                if previous is None or int(local["state_count"]) > previous_count:
                    found[task_id] = (normalized, path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "rms_persistence")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atlas = load(HOTSPOT / "backward_rescreen_atlas.json")
    pool = [
        row for row in atlas["rows"]
        if row.get("confirmation_available") and float(row.get("confirmation_local_ratio", 0)) > 0
        and row.get("outcome") != "PROMOTED"
    ]
    # Match the frozen denominator used by the screen-negative audit.
    task_ids = {str(row["task_id"]) for row in pool}
    populations = find_population_rows(task_ids)
    formation_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in pool:
        task_id = str(row["task_id"])
        if task_id not in populations:
            missing.append(task_id)
            continue
        data, path = populations[task_id]
        population_map = data["populations"]
        partition = "confirmation" if "confirmation" in population_map else "screening"
        local = population_map[partition]["LOCAL_ENDPOINT"]
        coordinates = int(local["coordinate_count"])
        formation_rows.append({
            "task_id": task_id,
            "model": row["model"],
            "sequence_length": row["sequence_length"],
            "family": row["family"],
            "local_rms": math.sqrt(float(local["average_state_energy"]) / coordinates),
            "formation_cross_state_ratio": float(local["cross_state_ratio"]),
            "formation_status": local["status"],
            "magnitude_state_count": int(local["state_count"]),
            "magnitude_partition": partition,
            "artifact": str(path.relative_to(ROOT)),
        })

    trajectory_rows: list[dict[str, Any]] = []
    for path in sorted((BASE / "consequence").glob("*.json")):
        data = load(path)
        if data.get("status") != "COMPLETE" or int(data.get("step_count", 0)) != 32:
            continue
        levels = data["statistics"]["levels"]
        coordinates = int(data["carrier_coordinates"])
        local = levels["local"]
        trajectory_rows.append({
            "case_id": data["case_id"],
            "architecture": data["architecture"],
            "carrier": data["carrier"],
            "local_rms": math.sqrt(float(local["energy"]) / (32 * coordinates)),
            "local_amplification": float(local["coherence_amplification"]),
            "actual_amplification": float(levels["actual"]["coherence_amplification"]),
            "feedback_amplification": float(levels["feedback"]["coherence_amplification"]),
            "artifact": str(path.relative_to(ROOT)),
        })

    payload = {
        "schema": "kernel-analyzer-rms-persistence-audit-v1",
        "status": "COMPLETE" if len(formation_rows) == 32 and len(trajectory_rows) == 12 else "PARTIAL",
        "formation_population": {
            "eligible": len(pool), "matched": len(formation_rows), "missing_task_ids": missing,
            "correlation": correlations(formation_rows, "formation_cross_state_ratio"),
            "rows": formation_rows,
            "claim_boundary": "16-state open-loop formation directionality; not live trajectory persistence.",
        },
        "live_trajectory_sample": {
            "count": len(trajectory_rows),
            "local_correlation": correlations(trajectory_rows, "local_amplification"),
            "actual_correlation": correlations(trajectory_rows, "actual_amplification"),
            "rows": trajectory_rows,
            "claim_boundary": "32-step four-counterfactual screen-negative audit sample.",
        },
    }
    (args.output_dir / "rms_persistence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, rows in (("formation32.csv", formation_rows), ("trajectory12.csv", trajectory_rows)):
        if not rows:
            continue
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].scatter(
            [row["local_rms"] for row in formation_rows],
            [row["formation_cross_state_ratio"] for row in formation_rows],
            alpha=0.75,
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("local residual RMS")
        axes[0].set_ylabel("open-loop cross-state ratio")
        axes[0].set_title("32 reachable nonzero candidates")
        axes[1].scatter(
            [row["local_rms"] for row in trajectory_rows],
            [row["local_amplification"] for row in trajectory_rows],
            label="local A", alpha=0.8,
        )
        axes[1].scatter(
            [row["local_rms"] for row in trajectory_rows],
            [row["actual_amplification"] for row in trajectory_rows],
            label="actual A", marker="x", alpha=0.8,
        )
        axes[1].set_xscale("log")
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_xlabel("local effective-update RMS")
        axes[1].set_ylabel("32-step amplification A")
        axes[1].set_title("12-case live consequence sample")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "rms_persistence.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps({
        "status": payload["status"], "formation": len(formation_rows),
        "trajectory": len(trajectory_rows), "output": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
