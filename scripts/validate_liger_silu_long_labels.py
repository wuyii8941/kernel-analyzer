#!/usr/bin/env python3
"""Validate the two headline long-horizon labels without touching the talk.

This is deliberately a small evidence check, not a new statistical analysis:
it prevents the 4096-step direct and feedback cases from being accidentally
reported with their old 32-step labels.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECHECK = ROOT / "results/property/declared_persistent_4096/liger_silu_recheck.json"
AUDIT = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
OUT = ROOT / "results/property/declared_persistent_4096/liger_silu_validation.json"


def main() -> None:
    recheck = json.loads(RECHECK.read_text())
    audit = json.loads(AUDIT.read_text())
    rows = {row["case"]: row for row in audit["rows"]}
    checked = {}

    liger = recheck["cases"][0]
    if liger["final_label"] != "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT":
        raise RuntimeError("Liger is not labelled as a long persistent-bias case")
    if liger["direct"]["A4096_exact_full_coordinate"] <= 2.0:
        raise RuntimeError("Liger long direct direction is unexpectedly weak")
    if "last_512_loss_gap_mean" not in liger["paired_consequence"]:
        raise RuntimeError("Liger paired loss split is missing")
    if rows["Liger fused CE"]["final_label"] != liger["final_label"]:
        raise RuntimeError("audit and recheck disagree for Liger")
    checked["Liger fused CE"] = {
        "final_label": liger["final_label"],
        "A4096": liger["direct"]["A4096_exact_full_coordinate"],
        "paired_loss_observed": True,
        "interpretation": "direct bias component survives the long horizon",
    }

    silu = recheck["cases"][1]
    if silu["final_label"] != "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT":
        raise RuntimeError("SiLU is not labelled as feedback-sustained long bias")
    local = silu["direct"]
    feedback = silu["feedback"]
    if local["local_A4096"] >= 1.1:
        raise RuntimeError("SiLU local component unexpectedly passes direct gate")
    if feedback["A4096"] <= 2.0:
        raise RuntimeError("SiLU feedback component is not long directional")
    if "last_512_loss_gap_mean" not in silu["paired_consequence"]:
        raise RuntimeError("SiLU paired loss split is missing")
    if rows["Qwen3-VL SiLU"]["final_label"] != silu["final_label"]:
        raise RuntimeError("audit and recheck disagree for SiLU")
    checked["Qwen3-VL SiLU"] = {
        "final_label": silu["final_label"],
        "local_A4096": local["local_A4096"],
        "feedback_A4096": feedback["A4096"],
        "paired_loss_observed": True,
        "interpretation": "direct source is diffuse; feedback bias survives the long horizon",
    }

    payload = {
        "schema": "liger-silu-long-label-validation-v1",
        "status": "PASS",
        "source_artifacts": [str(RECHECK.relative_to(ROOT)), str(AUDIT.relative_to(ROOT))],
        "checked": checked,
        "claim_boundary": (
            "This check validates the two 4096-step labels only. It does not claim "
            "full-parameter training, convergence to different final losses, or a "
            "universal property."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
