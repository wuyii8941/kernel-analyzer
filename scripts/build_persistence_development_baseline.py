#!/usr/bin/env python3
"""Build the preregistered development baseline without imputing missing levels."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/persistence_v1/development_baseline.json"


def load(path: str) -> dict[str, Any]:
    target = ROOT / path
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def seup(case_id: str, path: str) -> dict[str, Any]:
    evaluation = load(path)["evaluation"]
    local_energy = float(evaluation["local_energy"])
    feedback_energy = float(evaluation["feedback_energy"])
    return {
        "case_id": case_id,
        "horizon": int(evaluation["evaluation_steps"]),
        "levels": {
            "local_effective_update": {
                "coherence_amplification": float(evaluation["local_accumulation_l2"])
                / max(local_energy**0.5, 1e-30),
                "source": path,
            },
            "feedback": {
                "coherence_amplification": float(evaluation["feedback_accumulation_l2"])
                / max(feedback_energy**0.5, 1e-30),
                "source": path,
            },
            "actual_drift_increment": {
                "status": "MISSING_ACTUAL_ENERGY_REQUIRES_RERUN",
            },
            "local_endpoint": {"status": "MISSING_ORDERED_TRAJECTORY_VECTOR"},
            "parameter_gradient": {"status": "MISSING_ORDERED_TRAJECTORY_VECTOR"},
        },
        "complete_five_level_signature": False,
    }


def ordered(case_id: str, path: str) -> dict[str, Any]:
    source = load(path)
    summaries = source["summaries"]
    return {
        "case_id": case_id,
        "horizon": int(source["protocol"]["steps"]),
        "levels": {
            "local_effective_update": summaries["local"],
            "feedback": summaries["feedback"],
            "actual_drift_increment": summaries["actual_drift_increment"],
            "local_endpoint": {"status": "MISSING_ORDERED_TRAJECTORY_VECTOR"},
            "parameter_gradient": {"status": "MISSING_ORDERED_TRAJECTORY_VECTOR"},
        },
        "alignment": {
            "local_final_drift_cosine": summaries["local_final_drift_cosine"],
            "feedback_final_drift_cosine": summaries["feedback_final_drift_cosine"],
        },
        "complete_five_level_signature": False,
        "source": path,
    }


def main() -> None:
    rows = [
        seup("liger_fused_ce", "results/property/seup_mainline/liger_seup.json"),
        seup("phi4_seq64_lmhead_dx", "results/property/seup_mainline/phi_seup.json"),
        seup("qwen_saved_p_seq128", "results/property/seup_mainline/qwen_softmax_seup.json"),
        seup("qwen_bmm_seq64", "results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz"),
        ordered(
            "qwen128_vproj_mm",
            "results/coverage/cases/qwen128_vproj_rounding_persistence.json",
        ),
        ordered(
            "qwen3vl_silu_layer0",
            "results/coverage/cases/qwen3vl_layer0_silu_persistence_recurrence.json",
        ),
    ]
    payload = {
        "schema": "kernel-analyzer-persistence-development-baseline-v1",
        "status": "PARTIAL_EXISTING_ARTIFACT_AUDIT",
        "claim_boundary": (
            "This table is an inventory, not a property verdict. Missing levels are never "
            "imputed; horizons are reported explicitly and are not compared as effect sizes."
        ),
        "cases": rows,
        "rerun_required": [
            row["case_id"] for row in rows if not row["complete_five_level_signature"]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
