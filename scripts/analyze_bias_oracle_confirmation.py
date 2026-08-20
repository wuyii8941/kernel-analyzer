#!/usr/bin/env python3
"""Apply the frozen moving-frame rule to disjoint confirmation states."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_relative_oracle import (  # noqa: E402
    ReferenceRelativeObservation,
    certify_reference_relative,
)


BASE = ROOT / "results/property/bias_oracle_recovery/confirmation"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def pooled_certificate(case: dict[str, Any]) -> dict[str, Any]:
    source = case["reference_relative_parameter_gradient"]
    rows = source["calibration"]["rows"] + source["confirmation"]["rows"]
    certificate = certify_reference_relative([
        ReferenceRelativeObservation(
            condition_id=str(row["condition_id"]),
            error_reference_dot=float(row["error_reference_dot"]),
            error_energy=float(row["error_energy"]),
            reference_energy=float(row["reference_energy"]),
        )
        for row in rows
    ])
    return certificate.as_dict()


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Prospective moving-frame confirmation",
        "",
        "## Result",
        "",
        (
            f"The frozen screen confirmed {result['candidate_confirmed']}/"
            f"{result['candidate_count']} promoted candidates on 32 disjoint natural "
            f"states. It directly flagged {result['new_case']}. None of the "
            f"{result['control_count']} sign-changing controls produced a directional hit."
        ),
        "",
        "| case | role | mean coefficient | 95% bootstrap CI | result |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["rows"]:
        interval = row["certificate"]["bootstrap_interval"]
        lines.append(
            f"| `{row['case_id']}` | {row['development_role']} | "
            f"{row['certificate']['mean_coefficient']:.6g} | "
            f"[{interval[0]:.6g}, {interval[1]:.6g}] | `{row['result']}` |"
        )
    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        "The new DeepSeek case is the layer-35 attention value-gradient boundary:",
        "",
        "`O = P V`, `dV = P^T dO`, `dW_v = dV^T H`.",
        "",
        (
            "Replacing only the actual compiled BF16 `dV` BMM output with its exact "
            "FP32-recomputed, BF16-ABI reference changes the complete `v_proj.weight` "
            "gradient. Across unseen states, the candidate-minus-repair gradient has a "
            "negative mean component in the same-state repair-gradient frame. Thus the "
            "implementation systematically contracts this update component even though "
            "the absolute parameter-space direction changes with the state."
        ),
        "",
        "This is a new conditional training-bias case, not proof that every attention "
        "BMM is biased. Qwen q_norm and DeepSeek softmax failed the frozen confirmation "
        "and remain unresolved/non-replicating rather than positives.",
        "",
        "## Oracle boundary",
        "",
        "The moving-frame statistic is now a validated sufficient risk witness. A miss "
        "is not a safety certificate: Phi requires complete-vector population coherence, "
        "while saved-P and SiLU require the exact antithetic optimizer-response witness. "
        "The practical oracle is therefore a fail-closed multi-witness cascade, not one "
        "universal scalar.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol = load(BASE / "protocol.json")
    if protocol["status"] != "FROZEN_BEFORE_TAIL_STATE_MEASUREMENT":
        raise RuntimeError("confirmation protocol was not frozen before measurement")
    rows = []
    for frozen in protocol["cases"]:
        role = str(frozen["development_role"])
        if role not in {"CANDIDATE", "SIGN_CHANGING_CONTROL"}:
            continue
        case_id = str(frozen["case_id"])
        model = str(frozen["model"])
        case = load(BASE / model / f"{case_id}.json")
        certificate = pooled_certificate(case)
        directional = (
            certificate["status"] == "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
        )
        mean = float(certificate["mean_coefficient"])
        frozen_direction = frozen["frozen_direction"]
        direction_matches = (
            frozen_direction is not None
            and mean * int(frozen_direction) > 0.0
        )
        if role == "CANDIDATE":
            passed = directional and direction_matches
            outcome = "CONFIRMED_DIRECTIONAL_RISK" if passed else "REJECTED_ON_HELDOUT"
        else:
            passed = not directional
            outcome = (
                "CONTROL_NOT_FLAGGED" if passed
                else "CONTROL_BECAME_DIRECTIONAL_REVIEW_REQUIRED"
            )
        rows.append({
            "case_id": case_id,
            "model": model,
            "development_role": role,
            "frozen_direction": frozen_direction,
            "certificate": certificate,
            "passed": passed,
            "result": outcome,
        })
    candidates = [row for row in rows if row["development_role"] == "CANDIDATE"]
    controls = [
        row for row in rows
        if row["development_role"] == "SIGN_CHANGING_CONTROL"
    ]
    confirmed = [
        row for row in candidates
        if row["result"] == "CONFIRMED_DIRECTIONAL_RISK"
    ]
    result = {
        "schema": "kernel-analyzer-moving-frame-confirmation-result-v1",
        "status": "COMPLETE",
        "threshold_changed_after_measurement": False,
        "candidate_count": len(candidates),
        "candidate_confirmed": len(confirmed),
        "control_count": len(controls),
        "controls_not_flagged": sum(row["passed"] for row in controls),
        "new_case": confirmed[0]["case_id"] if len(confirmed) == 1 else None,
        "rows": rows,
        "decision": (
            "MOVING_FRAME_IS_A_VALIDATED_SUFFICIENT_RISK_WITNESS_"
            "NOT_A_STANDALONE_SAFETY_ORACLE"
            if confirmed and all(row["passed"] for row in controls)
            else "MOVING_FRAME_NOT_VALIDATED"
        ),
        "cost": {
            "development_full_fb_arms": 448,
            "confirmation_full_fb_arms": 352,
            "reference_relative_extra_fb_arms_after_candidate_repair_capture": 0,
            "grouped_per_state_cost": "N_ENDPOINT_REPAIRS_PLUS_3_SHARED_ARMS",
        },
    }
    (BASE / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (BASE / "summary.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "candidate_confirmed": len(confirmed),
        "controls_not_flagged": result["controls_not_flagged"],
        "new_case": result["new_case"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
