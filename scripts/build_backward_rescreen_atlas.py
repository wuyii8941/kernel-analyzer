#!/usr/bin/env python3
"""Join backward F+B short screens with independent confirmation outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"


def ratio(layer: dict) -> float:
    return float(layer["cross_state_ratio"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output-prefix", type=Path, default=BASE / "backward_rescreen_atlas")
    args = parser.parse_args()

    equivalence_path = args.base / "multishape_backward_equivalence.json"
    equivalence = json.loads(equivalence_path.read_text())
    # Cell IDs are generated presentation identifiers and can change when the
    # denominator is rebuilt.  Historical screen artifacts remain valid, so
    # join scientific metadata by the stable model/shape/task coordinate.
    model_alias = {"phi": "phi4", "deepseek": "deepseek8b", "deepseek8": "deepseek8b"}
    cell_by_coordinate = {
        (
            str(cell["model"]),
            int(cell["sequence_length"]),
            str(cell["representative"]["task_id"]),
        ): cell
        for cell in equivalence["cells"]
    }
    cells_by_id = {str(cell["cell_id"]): cell for cell in equivalence["cells"]}
    for member in equivalence["membership"]:
        model, shape, task_id = str(member["member_id"]).split(":", 2)
        sequence_length = int(shape.removeprefix("seq"))
        cell_by_coordinate.setdefault(
            (model, sequence_length, task_id), cells_by_id[str(member["cell_id"])]
        )

    confirmations: dict[str, dict] = {}
    for root_name in ("equivalence_confirmation", "multishape_confirmation"):
        for path in (args.base / root_name).rglob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("schema") == "kernel-analyzer-bias-formation-certificate-v2_1":
                confirmations[str(payload["case_id"])] = payload

    screens: list[tuple[str, int, Path]] = []
    seq64 = args.base / "equivalence_screen"
    for path in seq64.glob("*/screening_gram.json"):
        screens.append((path.parent.name, 64, path))
    multishape = args.base / "multishape_screen"
    for path in multishape.glob("*_seq*/screening_gram.json"):
        name = path.parent.name
        match = re.match(r"^(?P<model>.+)_seq(?P<length>\d+)(?:_.+)?$", name)
        if match is None:
            continue
        screens.append((match.group("model"), int(match.group("length")), path))

    rows = []
    for model, sequence_length, path in sorted(screens):
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            canonical_model = model_alias.get(model, model)
            cell = cell_by_coordinate.get(
                (canonical_model, sequence_length, str(case["task_id"])), {}
            )
            local = ratio(case["layers"]["LOCAL_ENDPOINT"])
            gradient = ratio(case["layers"]["PARAMETER_GRADIENT"])
            confirmation = confirmations.get(str(case["case_id"]))
            row = {
                "model": canonical_model,
                "sequence_length": sequence_length,
                "case_id": str(case["case_id"]),
                "task_id": str(case["task_id"]),
                "carrier": str(case["carrier"]),
                "family": cell.get("family"),
                "depth_stratum": cell.get("depth_stratum"),
                "capture_boundary": cell.get("capture_boundary"),
                "member_count": cell.get("member_count"),
                "screen_local_ratio": local,
                "screen_gradient_ratio": gradient,
                "screen_transport_gain": gradient - local,
                "confirmation_available": confirmation is not None,
                "first_confirmed_bias_stage": None,
                "calibration_local_ratio": None,
                "calibration_gradient_ratio": None,
                "confirmation_local_ratio": None,
                "confirmation_gradient_ratio": None,
                "confirmation_gradient_status": None,
                "outcome": "NOT_PROMOTED",
            }
            if confirmation is not None:
                populations = confirmation["populations"]
                row.update({
                    "first_confirmed_bias_stage": confirmation["first_confirmed_bias_stage"],
                    "calibration_local_ratio": ratio(populations["calibration"]["LOCAL_ENDPOINT"]),
                    "calibration_gradient_ratio": ratio(populations["calibration"]["PARAMETER_GRADIENT"]),
                    "confirmation_local_ratio": ratio(populations["confirmation"]["LOCAL_ENDPOINT"]),
                    "confirmation_gradient_ratio": ratio(populations["confirmation"]["PARAMETER_GRADIENT"]),
                    "confirmation_gradient_status": populations["confirmation"]["PARAMETER_GRADIENT"]["status"],
                })
                if confirmation["first_confirmed_bias_stage"]:
                    row["outcome"] = "CONFIRMED_BIAS"
                elif gradient > 0 and row["confirmation_gradient_ratio"] < 0:
                    row["outcome"] = "DIRECTION_REVERSAL"
                elif row["calibration_gradient_ratio"] > 0 and row["confirmation_gradient_ratio"] > 0:
                    row["outcome"] = "SAME_SIGN_UNRESOLVED_OR_CENTERED"
                else:
                    row["outcome"] = "NOT_REPRODUCED"
            rows.append(row)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps({
        "schema": "kernel-analyzer-backward-rescreen-atlas-v1",
        "rows": rows,
        "summary": {
            "screened": len(rows),
            "confirmed": sum(row["confirmation_available"] for row in rows),
            "confirmed_bias": sum(row["outcome"] == "CONFIRMED_BIAS" for row in rows),
            "direction_reversal": sum(row["outcome"] == "DIRECTION_REVERSAL" for row in rows),
            "same_sign_unresolved_or_centered": sum(
                row["outcome"] == "SAME_SIGN_UNRESOLVED_OR_CENTERED" for row in rows),
        },
    }, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
