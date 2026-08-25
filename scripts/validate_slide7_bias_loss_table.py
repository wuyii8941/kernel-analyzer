#!/usr/bin/env python3
"""Validate every row in Slide 7's bias-plus-loss table.

The check keeps two questions separate:

1. Is there independent evidence that the implementation contrast has bias?
2. Did the paired long run record a loss split?

A loss split alone is never accepted as bias evidence.  The resulting JSON is
the machine-readable certificate behind the second table in Slide 7.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
PARITY = ROOT / "results/property/joint_bias_formation_v1/mu_parity_decomposition.json"
TALK = ROOT / "docs/talk_beyond_tolerance.md"
OUT = ROOT / "results/property/declared_persistent_4096/slide7_bias_loss_validation.json"

INCLUDED_LABELS = {
    "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT",
    "AGGREGATE_LONG_BIAS_WITH_PAIRED_LOSS_SPLIT_LATE_WINDOW_NOT_EXPORTED",
    "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT",
    "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE",
}

EXPECTED_CASES = {
    "Liger fused CE",
    "Phi lm_head dX",
    "Qwen lm_head dX",
    "Llama lm_head dX",
    "Ministral lm_head dX",
    "Qwen64 v_proj",
    "Qwen v_proj",
    "Mamba in_proj",
    "saved-P",
    "Qwen3-VL SiLU",
}


def load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def conditional_bias_certificate(path: Path, arm: str = "JOINT") -> dict:
    payload = load(path)
    if isinstance(payload.get("arms"), dict):
        summary = payload["arms"][arm]["aggregate"]
    else:
        summary = payload["conditional_debias_summary"][arm]
    roles = summary["roles"]
    return {
        "artifact": str(path.relative_to(ROOT)),
        "condition_count": summary["condition_count"],
        "local_biased": roles["candidate_local_effect_removed"]["status_counts"].get("CONDITIONAL_BIAS", 0),
        "gradient_biased": roles["candidate_gradient_effect_removed"]["status_counts"].get("CONDITIONAL_BIAS", 0),
        "adamw_biased": roles["candidate_adamw_zero_update_effect_removed"]["status_counts"].get("CONDITIONAL_BIAS", 0),
        "repair_local_centered": roles["repair_local_residual"]["status_counts"].get("CONDITIONAL_CENTERED", 0),
    }


def formation_certificate(case: str, row: dict, parity: dict) -> dict:
    long = row["long_direct"]
    if case in {"Liger fused CE", "Phi lm_head dX", "Qwen lm_head dX"}:
        require(long.get("long_direct") == "ROBUST", f"{case}: long direct label is not robust")
        require(long["A4096"] > long["null95"], f"{case}: A4096 does not exceed its null")
        require(long.get("late_windows_above_one", long.get("late_windows_above_own_null", 0)) >= 48,
                f"{case}: fewer than 75% of late windows retain direction")
        return {"kind": "4096_STEP_DIRECT_BIAS", "A4096": long["A4096"], "null95": long["null95"]}

    if case in {"Llama lm_head dX", "Ministral lm_head dX"}:
        require(long.get("long_direct") == "ROBUST", f"{case}: aggregate direct label is not robust")
        require(long["A4096"] > long["null95"] and long["p"] <= 0.05,
                f"{case}: aggregate 4096-step direct evidence fails")
        return {
            "kind": "4096_STEP_AGGREGATE_DIRECT_BIAS_LATE_WINDOWS_NOT_EXPORTED",
            "A4096": long["A4096"],
            "null95": long["null95"],
            "p": long["p"],
        }

    if case == "Qwen3-VL SiLU":
        p = parity["qwen3vl_silu_seq160"]
        require(p["even_over_natural_resultant"] > 0, "SiLU: response-even component is zero")
        require(long.get("long_direct") == "FEEDBACK_SUSTAINED", "SiLU: feedback is not sustained")
        require(long["projection_A4096"] > long["null95"], "SiLU: feedback projection does not exceed null")
        require(long["late_windows_above_one"] / long["late_windows"] >= 0.75,
                "SiLU: fewer than 75% of late feedback windows retain direction")
        return {
            "kind": "ANTITHETIC_RESPONSE_BIAS_PLUS_4096_STEP_FEEDBACK_BIAS",
            "response_even_over_natural": p["even_over_natural_resultant"],
            "feedback_projection_A4096": long["projection_A4096"],
            "feedback_null95": long["null95"],
        }

    if case == "Qwen64 v_proj":
        cert = conditional_bias_certificate(ROOT / "results/property/conditional_debias/qwen64_vproj.json")
    elif case == "Qwen v_proj":
        cert = conditional_bias_certificate(ROOT / "results/property/conditional_debias/qwen128_vproj.json", "ROUNDING_ONLY")
    elif case == "Mamba in_proj":
        cert = conditional_bias_certificate(ROOT / "results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz")
    elif case == "saved-P":
        p = parity["qwen_saved_p_seq128"]
        require(p["even_over_natural_resultant"] > 0, "saved-P: response-even component is zero")
        require(long.get("long_direct") == "NOT_ROBUST", "saved-P: expected a nonpersistent direct component")
        return {
            "kind": "ANTITHETIC_RESPONSE_BIAS_NOT_LONG_DIRECT",
            "artifact": str(PARITY.relative_to(ROOT)),
            "response_even_over_natural": p["even_over_natural_resultant"],
        }
    else:
        raise RuntimeError(f"No independent formation check is declared for {case}")

    require(cert["condition_count"] == 16, f"{case}: expected 16 fixed conditions")
    require(cert["local_biased"] == 16, f"{case}: local bias was not confirmed in all conditions")
    require(cert["adamw_biased"] == 16, f"{case}: AdamW bias was not confirmed in all conditions")
    require(cert["repair_local_centered"] == 16, f"{case}: repair residual was not centered in all conditions")
    require(long.get("long_direct") == "NOT_ROBUST", f"{case}: expected a nonpersistent long direct component")
    cert["kind"] = "CONDITIONAL_BIAS_NOT_LONG_DIRECT"
    return cert


def main() -> None:
    audit = load(AUDIT)
    rows = {row["case"]: row for row in audit["rows"] if row["final_label"] in INCLUDED_LABELS}
    require(set(rows) == EXPECTED_CASES, f"Slide 7 set mismatch: {sorted(set(rows) ^ EXPECTED_CASES)}")

    parity_payload = load(PARITY)
    parity = {row["case_id"]: row for row in parity_payload["cases"]}
    talk = TALK.read_text()
    require("下面这张表只保留同时满足两项的记录" in talk, "Slide 7 second-table introduction is missing")

    checked = []
    for case in sorted(rows):
        row = rows[case]
        long_path = ROOT / row["long_direct"]["artifact"]
        require(long_path.exists(), f"{case}: long artifact is missing")
        require(sha256(long_path) == row["direct_sha256"], f"{case}: long artifact hash mismatch")

        consequence = row["paired_consequence"]
        loss = consequence["loss_audit"]
        require(loss.get("recorded_steps") == 4096, f"{case}: paired loss does not contain 4096 rows")
        require(loss.get("any_period_split") is True, f"{case}: paired loss never splits")
        require(loss.get("split_step_count", 0) > 0, f"{case}: no split step was recorded")
        require(loss.get("max_abs_gap", 0.0) > loss.get("split_threshold", 1e-8),
                f"{case}: max loss gap does not exceed the frozen threshold")

        formation = formation_certificate(case, row, parity)
        require(row["model"] in talk and row["operator_or_region"].split()[0] in talk,
                f"{case}: no corresponding model/operator text found in the talk")
        checked.append({
            "case": case,
            "model": row["model"],
            "operator_or_region": row["operator_or_region"],
            "final_label": row["final_label"],
            "bias_evidence": formation,
            "loss_evidence": {
                "recorded_steps": loss["recorded_steps"],
                "split_step_count": loss["split_step_count"],
                "max_abs_gap": loss["max_abs_gap"],
                "split_threshold": loss["split_threshold"],
            },
            "long_artifact": row["long_direct"]["artifact"],
            "long_sha256": row["direct_sha256"],
            "paired_artifact": consequence["artifact"],
            "paired_sha256": row.get("paired_sha256"),
        })

    counts = {
        "bias_and_loss_rows": len(checked),
        "late_window_direct_persistent": sum(r["final_label"] == "PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT" for r in checked),
        "aggregate_direct_missing_late_windows": sum(r["final_label"] == "AGGREGATE_LONG_BIAS_WITH_PAIRED_LOSS_SPLIT_LATE_WINDOW_NOT_EXPORTED" for r in checked),
        "feedback_persistent": sum(r["final_label"] == "FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT" for r in checked),
        "bias_and_loss_but_not_long_direct": sum(r["final_label"] == "LONG_LOSS_SPLIT_WITHOUT_DIRECT_PERSISTENCE" for r in checked),
    }
    payload = {
        "schema": "slide7-bias-plus-loss-validation-v1",
        "status": "PASS",
        "counts": counts,
        "checked": checked,
        "claim_boundary": (
            "All rows have independent bias evidence and a 4096-record paired loss split. "
            "Only the first three direct rows, two aggregate direct rows, and the SiLU feedback row "
            "have long directional evidence; the remaining four are consequence controls and are not "
            "persistent-bias cases. A loss split is not evidence of different converged optima."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "counts": counts, "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
