#!/usr/bin/env python3
"""Recompute what the retained reduction cases establish about mean bias.

This is an analysis of existing measurements. It preserves their fixed-suite
scope and does not turn counts of positive projections into a mean test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_current_root_cause_closure import heldout_direction_from_gram

OUT = ROOT / "results/property/root_cause_closure_v1/reduction_mean_bias_audit_v1.json"


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def analyze() -> dict:
    granite_path = "results/property/training_numerical_analysis_v2/granite_expert_order_confirmation_v2/raw.json"
    granite = read(granite_path)
    assert granite["state_ids"] == granite["calibration_state_ids"] + granite["confirmation_state_ids"]
    granite_stages = {}
    for stage in ("LOCAL", "PARAMETER_GRADIENT", "PARAMETER_WRITE"):
        geometries = granite["stages"][stage]
        geometry = "EXACT" if "EXACT" in geometries else next(iter(geometries))
        granite_stages[stage] = {
            "geometry": geometry,
            **heldout_direction_from_gram(geometries[geometry]["profile"]),
        }
    rows = granite["original_coordinate_statistics"]["PARAMETER_WRITE"][16:]
    total_rms = math.sqrt(math.fsum(r["effect_energy"] for r in rows)
                          / math.fsum(r["repair_energy"] for r in rows))

    write_path = "results/property/liger_fp32_chunk_order_v1/length64_parameter_write_confirmation_v1.json"
    write = read(write_path)
    liger_stages = {}
    for length in (64, 256):
        path = f"results/property/liger_fp32_chunk_order_v1/length{length}_confirmation.json"
        data = read(path)
        liger_stages[str(length)] = {
            "source": path,
            "PARAMETER_GRADIENT": heldout_direction_from_gram(data["profiles"]["PARAMETER_GRADIENT"]),
            "PROPOSED_ADAMW_UPDATE": heldout_direction_from_gram(data["profiles"]["ADAMW_UPDATE"]),
            "schedule_prediction_positive_count": data["source_prediction"]["confirmation_positive_count"],
            "schedule_prediction_scope": "GRADIENT_PROJECTION_ON_CALIBRATION_SCHEDULE_MEAN_DIRECTION",
        }
    # The older same-bank gradient measurement used the correct dW reference.
    older = read(liger_stages["64"]["source"])
    matching_effect = (
        write["state_ids"] == older["state_ids"]
        and write["profiles"]["PARAMETER_GRADIENT"]["suite"]["joint_gram"]["effect_effect"]
        == older["profiles"]["PARAMETER_GRADIENT"]["suite"]["joint_gram"]["effect_effect"]
    )
    population_results = {}
    for length in (64, 256):
        path = ROOT / f"results/property/liger_fp32_chunk_order_v1/length{length}_population_mean_v1.json"
        if path.is_file():
            data = read(str(path.relative_to(ROOT)))
            population_results[str(length)] = {
                "source": str(path.relative_to(ROOT)),
                "confirmation": data["confirmation"],
                "population": data["population"],
                "calibration_count": data["calibration_count"],
                "confirmation_count": data["confirmation_count"],
            }
    granite_population_path = ROOT / "results/property/granite_expert_order_population_v1/result.json"
    granite_population = read(str(granite_population_path.relative_to(ROOT))) if granite_population_path.is_file() else None
    return {
        "schema": "kernel-analyzer-reduction-mean-bias-audit-v1",
        "status": "EXISTING_FIXED_SUITE_EVIDENCE_REANALYZED",
        "new_gpu_measurements": False,
        "natural_population_mean_bias_established": False,
        "declared_empirical_bank_population_mean_bias_established": bool(population_results or granite_population),
        "zero_population_mean_established": False,
        "granite": {
            "source": granite_path,
            "state_count": len(granite["state_ids"]),
            "stages": granite_stages,
            "confirmation_original_coordinate_write_rms": total_rms,
            "confirmation_nonzero_write_states": sum(r["effect_energy"] > 0 for r in rows),
            "interpretation": (
                "The historical 16-state fixed-suite write effects were orthogonal to the "
                "historical calibration direction and therefore did not establish a mean. "
                "A separate 32-calibration/64-confirmation run below now tests a frozen "
                "direction on iid-with-replacement draws from the declared Granite empirical "
                "bank; its positive one-sided bound supports a scoped projected mean, not a "
                "natural-population claim."
            ),
        },
        "liger": {
            "gradient_and_proposed_update": liger_stages,
            "actual_parameter_write": {
                "source": write_path,
                "geometry": "PERIODIC_LINEAR_SUMMARY_8192",
                "original_coordinate_write_rms_all_32_states": write["fixed_suite_total_write_rms"],
                **heldout_direction_from_gram(write["profiles"]["PARAMETER_WRITE"]),
            },
            "legacy_analysis_errors": {
                "independence": "Fixed-bank identifiers were supplied as independent units. Historical population labels and p-values are not accepted as population inference.",
                "gradient_reference": "The write-v1 runner used reverse hidden gradient for the dW reference summary. That gradient profile's normalization, aligned and residual fields are invalid; the actual parameter-write statistics are unaffected.",
                "same_bank_gradient_effect_gram_matches_correct_reference_run": matching_effect,
                "correct_gradient_reference_source": liger_stages["64"]["source"],
            },
            "interpretation": "Positive held-out projections are fixed-bank directional evidence. The newer original-coordinate population runs provide scoped empirical-bank mean tests; they do not establish a natural-pretraining population claim.",
            "population_mean_confirmations": population_results,
        },
        "granite_population_mean_confirmation": (
            {
                "source": str(granite_population_path.relative_to(ROOT)),
                "parameter_write": granite_population["parameter_write"],
                "population": granite_population["population"],
                "calibration_count": granite_population["calibration_count"],
                "confirmation_count": granite_population["confirmation_count"],
            }
            if granite_population is not None else None
        ),
        "needed_for_population_claim": [
            "Specify the input/state distribution and the measured stage.",
            "Freeze a direction using development data, then independently sample confirmation units without outcome-based selection.",
            "Measure projection magnitudes and apply a valid mean test under declared assumptions and multiplicity rules.",
            "Assess accumulated training consequences separately from the one-step mean.",
        ],
    }


def main() -> None:
    report = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": report["status"]}))


if __name__ == "__main__":
    main()
