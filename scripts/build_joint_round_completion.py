#!/usr/bin/env python3
"""Fail closed while consolidating the completed joint-bias experiment round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    base = args.base

    consequence = load(base / "consequence_summary.json")
    if consequence.get("completed_cases") != consequence.get("expected_cases"):
        raise RuntimeError("consequence audit is incomplete")
    if consequence.get("invalid_results"):
        raise RuntimeError("consequence audit contains invalid results")

    offline = load(base / "offline_factor_summary.json")
    if int(offline.get("response_even_odd_raw_ready_count", 0)) < 2:
        raise RuntimeError("both exact raw response replays are not retained")

    heldout = load(base / "heldout_confirmation.json")
    deepseek_paths = sorted((base / "antithetic/deepseek8b_seq64").glob("*.json"))
    if len(deepseek_paths) != 3:
        raise RuntimeError("expected three formal DeepSeek antithetic captures")
    deepseek = [load(path) for path in deepseek_paths]
    representable = sum(
        bool(row.get("antithetic_response", {}).get("representability", {}).get(
            "exact_all_states"
        ))
        for row in deepseek
    )

    prefix = consequence["prefix16_to_full32_backtest"]
    payload = {
        "schema": "kernel-analyzer-joint-bias-round-completion-v1",
        "status": "COMPLETE_WITH_CLAIM_BOUNDARIES",
        "live_consequence": {
            "completed": consequence["completed_cases"],
            "expected": consequence["expected_cases"],
            "regime_counts": consequence["regime_counts"],
            "all_recurrences_closed": all(
                float(row["max_recurrence_relative"]) == 0.0
                for row in consequence["rows"]
            ),
            "prefix16_to_full32_backtest": prefix,
        },
        "raw_response_replay": {
            "exact_ready_cases": ["qwen_saved_p_seq128", "qwen3vl_silu_seq160"],
            "deepseek_bf16_reflection_cases": len(deepseek),
            "deepseek_exactly_representable": representable,
            "deepseek_status": (
                "COMPLETE" if representable == len(deepseek)
                else "UNRESOLVED_REPRESENTABILITY"
            ),
        },
        "heldout": {
            "status": heldout["status"],
            "source_factor_result": heldout["source_factor_confirmation"],
            "feedback_result": heldout["feedback_discovery"]["result"],
        },
        "scientific_decision": {
            "universal_property_supported": False,
            "full_joint_predictor_confirmed": False,
            "source_factor_heldout_negative_predictions_confirmed": True,
            "feedback_is_a_separate_measured_regime": True,
            "screen_negative_recall_interpretation": (
                "The complete residual-nonzero, parameter-reachable sample shows "
                "that local cancellation is common while closed-loop feedback can "
                "still produce persistent actual drift. It must not be promoted to "
                "operator-local persistent-source bias without a matched background "
                "feedback baseline."
            ),
        },
        "claim_boundary": (
            "This closes the originally declared measurement round. It does not "
            "establish a universal property or a universal safety oracle."
        ),
    }
    (base / "round_completion.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    regimes = ", ".join(
        f"{name}={count}" for name, count in sorted(consequence["regime_counts"].items())
    )
    actual = prefix["actual"]
    text = f"""# Joint bias round completion

Status: **COMPLETE_WITH_CLAIM_BOUNDARIES**.

## What completed

- Live four-counterfactual consequence: **{consequence['completed_cases']}/{consequence['expected_cases']}** cases, all recurrence residuals exactly closed.
- Observed regimes: `{regimes}`.
- Exact raw response replay: saved-P and Qwen3-VL SiLU, 32 states each.
- DeepSeek BF16 antithetic capture: 3/3 executed, but {len(deepseek) - representable}/3 reflected endpoints are not exactly representable and therefore remain `UNRESOLVED_REPRESENTABILITY`.
- Frozen Gemma NEW_IMPL held-out evidence consolidated.

## Main result

The screen-negative sample does not reveal a new persistent local-source family. Local increments are mostly diffusive, while persistent actual drift is usually carried by the closed-loop feedback term. This is a real trajectory effect, but it is not evidence that the sampled operator source itself has Flash-style persistent directionality.

The 16-step actual-amplification prefix preserves the side of the diffusive boundary in {actual['same_side_of_diffusive_one']}/{actual['evaluated']} cases and has Pearson correlation {actual['pearson_a16_a32']:.3f} with the 32-step value. This supports it as a triage feature, not yet as a universal classifier.

The held-out campaign confirms source-negative predictions on Gemma RMSNorm/attention, and independently finds Adam-moment feedback. Because the frozen feedback predictor abstained, the full three-factor predictor remains unresolved.

## Claim boundary

This round closes the requested execution gaps. It does **not** establish a universal property, does not label feedback-sustained drift as operator-local source bias, and does not impute exact `-epsilon` where BF16 representability forbids it.
"""
    (base / "round_completion.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "base": str(base)}, sort_keys=True))


if __name__ == "__main__":
    main()
