#!/usr/bin/env python3
"""Build a compact, verdict-blind map of F+B bias-mechanism candidates.

This is a discovery summary, not a classifier.  It contrasts short-screen
geometry with disjoint confirmation and keeps operator name, old T1--T4, and
SEUP verdicts out of candidate selection.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"
ATLAS = BASE / "backward_rescreen_atlas.json"
PHI = ROOT / "results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json"
OUT_JSON = BASE / "bias_mechanism_candidate_map.json"
OUT_MD = BASE / "bias_mechanism_candidate_map.md"


def population_row(payload: dict, partition: str, layer: str) -> dict:
    return payload["populations"][partition][layer]


def direct_screen_rows(base: Path) -> list[dict]:
    """Convert newly-bound direct screens to the atlas row shape.

    These screens deliberately have no confirmation verdict.  They are added
    only as discovery candidates; a four-state ratio can never become a case
    without a disjoint 16+16 run.
    """
    rows = []
    screen_paths = list((base / "multishape_screen").glob("*/screening_gram.json"))
    screen_paths += list((base / "semantic_region_screen").glob("*/screening_gram.json"))
    for path in sorted(screen_paths):
        match = re.match(r"^(qwen|phi4|mamba|deepseek8b)_seq(\d+)(?:_.+)?$", path.parent.name)
        if match is None:
            continue
        model = match.group(1)
        sequence_length = int(match.group(2))
        payload = json.loads(path.read_text())
        for case in payload.get("cases", []):
            layers = case.get("layers", {})
            local = layers.get("LOCAL_ENDPOINT", {})
            gradient = layers.get("PARAMETER_GRADIENT", {})
            if "cross_state_ratio" not in local or "cross_state_ratio" not in gradient:
                continue
            rows.append({
                "model": model,
                "sequence_length": sequence_length,
                "case_id": case.get("case_id", f"newly-bound:{model}:{sequence_length}:{case['task_id']}"),
                "task_id": case["task_id"],
                "family": case.get("family", "STATE_SPACE_RECURRENT_BACKWARD"),
                "carrier": case.get("carrier"),
                "screen_local_ratio": float(local["cross_state_ratio"]),
                "screen_gradient_ratio": float(gradient["cross_state_ratio"]),
                "screen_transport_gain": float(gradient["cross_state_ratio"] - local["cross_state_ratio"]),
                "confirmation_available": False,
                "confirmation_outcome": "NOT_PROMOTED",
                "confirmation_gradient_ratio": None,
            })
    return rows


def main() -> None:
    atlas = json.loads(ATLAS.read_text())
    rows = list(atlas["rows"])
    # The static atlas is intentionally immutable; append only newly-bound
    # screens that were not present when it was built.
    seen = {(row["model"], int(row["sequence_length"]), row["task_id"]) for row in rows}
    for row in direct_screen_rows(BASE):
        key = (row["model"], int(row["sequence_length"]), row["task_id"])
        if key not in seen:
            rows.append(row)
            seen.add(key)

    # A 0.1 ratio is only a cheap promotion threshold.  It is deliberately not
    # called BIASED; a scientific case still needs disjoint 16+16 populations.
    candidates = []
    for row in rows:
        local = float(row["screen_local_ratio"])
        gradient = float(row["screen_gradient_ratio"])
        if gradient < 0.1:
            continue
        signature = "SOURCE_DIRECTIONAL" if local >= 0.1 else "TRANSPORT_AMPLIFIED"
        candidates.append({
            "model": row["model"],
            "sequence_length": row["sequence_length"],
            "case_id": row["case_id"],
            "task_id": row["task_id"],
            "family": row.get("family"),
            "carrier": row["carrier"],
            "screen_signature": signature,
            "screen_local_ratio": local,
            "screen_gradient_ratio": gradient,
            "screen_transport_gain": gradient - local,
            "confirmation_available": row["confirmation_available"],
            "confirmation_outcome": row.get("outcome", row.get("confirmation_outcome", "NOT_PROMOTED")),
            "confirmation_gradient_ratio": row["confirmation_gradient_ratio"],
        })
    candidates.sort(key=lambda row: row["screen_gradient_ratio"], reverse=True)

    family_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        family_rows[(str(row["model"]), str(row.get("family")))].append(row)
    family_summary = []
    for (model, family), members in sorted(family_rows.items()):
        promoted = [row for row in members if float(row["screen_gradient_ratio"]) >= 0.1]
        confirmed = [row for row in promoted if row["confirmation_available"]]
        family_summary.append({
            "model": model,
            "family": family,
            "screened_task_coordinate_rows": len(members),
            "short_screen_candidates": len(promoted),
            "confirmed_candidates": len(confirmed),
            "confirmed_bias_cases": sum(row["outcome"] == "CONFIRMED_BIAS" for row in confirmed),
            "max_screen_gradient_ratio": max(float(row["screen_gradient_ratio"]) for row in members),
        })

    phi = json.loads(PHI.read_text())
    phi_anchor = {
        "case_id": phi["case_id"],
        "model": "phi",
        "sequence_length": 64,
        "first_confirmed_bias_stage": phi["first_confirmed_bias_stage"],
        "calibration_local_ratio": population_row(phi, "calibration", "LOCAL_ENDPOINT")["cross_state_ratio"],
        "calibration_gradient_ratio": population_row(phi, "calibration", "PARAMETER_GRADIENT")["cross_state_ratio"],
        "confirmation_local_ratio": population_row(phi, "confirmation", "LOCAL_ENDPOINT")["cross_state_ratio"],
        "confirmation_gradient_ratio": population_row(phi, "confirmation", "PARAMETER_GRADIENT")["cross_state_ratio"],
    }

    result = {
        "schema": "kernel-analyzer-bias-mechanism-candidate-map-v2",
        "status": "DISCOVERY_IN_PROGRESS",
        "selection_uses_old_t1_t4_or_seup": False,
        "complete_forward_backward_unit_required": True,
        # These rows combine several discovery ledgers and are not the frozen
        # semantic-cell denominator.  Keep the name explicit so this count is
        # never mistaken for unique F+B coverage.
        "screened_task_coordinate_rows": len(rows),
        "short_screen_candidate_count": len(candidates),
        "confirmed_new_case_count": sum(
            row["confirmation_outcome"] == "CONFIRMED_BIAS" for row in candidates
        ),
        "known_sensitivity_anchor": phi_anchor,
        "candidates": candidates,
        "family_summary": family_summary,
        "current_feature_hypothesis": (
            "A local difference becomes a bias candidate only when it survives the next "
            "representation boundary and its actual backward transport produces cross-state "
            "directional parameter-gradient geometry. Endpoint magnitude, operator identity, "
            "and proximity to the loss are insufficient."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Bias mechanism candidate map",
        "",
        "This is an F+B discovery map, not a scientific verdict. Short-screen ratios only",
        "select cases for disjoint confirmation; old T1--T4 and SEUP labels are not inputs.",
        "",
        "## Confirmed sensitivity anchor",
        "",
        f"- Phi seq64 `lm_head dX`: `{phi_anchor['first_confirmed_bias_stage']}`.",
        f"- Local ratio: {phi_anchor['calibration_local_ratio']:.4g} / "
        f"{phi_anchor['confirmation_local_ratio']:.4g} (calibration / confirmation).",
        f"- Gradient ratio: {phi_anchor['calibration_gradient_ratio']:.4g} / "
        f"{phi_anchor['confirmation_gradient_ratio']:.4g}.",
        "",
        "## Current candidate funnel",
        "",
        f"- Screened task-coordinate rows currently present: {len(rows)}.",
        f"- Engineering candidates with gradient ratio >= 0.1: {len(candidates)}.",
        f"- Newly confirmed stable bias cases in this rescreen: "
        f"{result['confirmed_new_case_count']}.",
        "",
        "## Leading unconfirmed or rejected signals",
        "",
        "| model | seq | family | signature | local | gradient | confirmation |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in candidates[:20]:
        lines.append(
            f"| {row['model']} | {row['sequence_length']} | {row['family']} | "
            f"{row['screen_signature']} | {row['screen_local_ratio']:.3g} | "
            f"{row['screen_gradient_ratio']:.3g} | {row['confirmation_outcome']} |"
        )
    lines += [
        "",
        "## Mechanistic contrast emerging from the screen",
        "",
        "- Internal Qwen/DeepSeek normalization reduction differences are erased at the next",
        "  reduction/BF16 writeback boundary.",
        "- Qwen and Mamba loss-head differences reach parameter gradients but remain centered",
        "  across states and shapes.",
        "- Phi seq64, unlike Phi seq128, combines boundary survival with directional backward",
        "  transport. Shape/reduction geometry is therefore part of the case, not merely the",
        "  `mm` operator name.",
        "- A stable feature claim still requires additional confirmed positives.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({
        "json": str(OUT_JSON), "md": str(OUT_MD),
        "screened_task_coordinate_rows": len(rows), "candidates": len(candidates),
        "new_confirmed": result["confirmed_new_case_count"],
    }))


if __name__ == "__main__":
    main()
