#!/usr/bin/env python3
"""Freeze the development-only property decisions before held-out use.

This is a *scope and measurement* freeze, not a claim that one property is
universal.  It deliberately keeps formation candidates separate from the
short-trajectory consequence screen and records overlap evidence for the
low-dimensional concentration proxy.  Missing measurements never become
negative labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results/property/bias_property_search/development_property_profile.json"
OUT = ROOT / "results/property/bias_property_search"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _concentration_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        confirmation = case.get("transport_concentration", {}).get("confirmation", {})
        layer = confirmation.get("PARAMETER_GRADIENT", {})
        proxy = layer.get("transport_concentration", {})
        top1 = _finite(proxy.get("top_eigen_fraction"))
        rank = _finite(proxy.get("effective_rank_participation"))
        if top1 is None or rank is None:
            continue
        rows.append({"case_id": case["case_id"], "top1": top1, "effective_rank": rank})
    return rows


def build(profile: dict[str, Any]) -> dict[str, Any]:
    cases = list(profile.get("cases", []))
    concentration = _concentration_rows(cases)
    controls = [
        row for row in cases if row.get("development_role") == "CENTERED_CONTROL"
    ]
    positive_roles = {
        "KNOWN_PERSISTENCE_ANCHOR",
        "KNOWN_FORMATION_AND_PERSISTENCE",
        "SEMANTIC_REGION_PERSISTENCE",
    }
    positive_ids = [
        row["case_id"] for row in cases if row.get("development_role") in positive_roles
    ]
    centered_control_count = len(controls)
    source_control_count = sum(
        row.get("source_asymmetry", {}).get("status") == "CENTERED_LOCAL_CONTROL"
        for row in controls
    )
    transport_control_count = sum(
        row.get("source_transport_coupling", {}).get("status")
        not in {None, "UNMEASURED", "UNMEASURED_OR_NOT_APPLICABLE"}
        for row in controls
    )
    stability_control_count = sum(
        row.get("carrier_stability", {}).get("status")
        in {"MEASURED_TRAJECTORY_STABLE", "MEASURED_TRAJECTORY_NOT_STABLE"}
        for row in controls
    )
    return {
        "schema": "kernel-analyzer-development-property-freeze-v1",
        "status": "FROZEN_BEFORE_HELDOUT",
        "freeze_scope": "source_persistent_and_transport_formation_only",
        "development_profile": "results/property/bias_property_search/development_property_profile.json",
        "historical_verdicts_used_as_predictor_inputs": False,
        "out_of_domain": [
            "feedback_sustained_drift",
            "discrete_routing_or_graph_divergence",
            "missing_or_unresolved_vjp_boundary",
            "optimizer_only_rectification_without_source_or_transport_capture",
        ],
        "development_population": {
            "persistent_or_formation_reference_ids": positive_ids,
            "centered_control_count": len(controls),
            "centered_control_ids": [row["case_id"] for row in controls],
            "concentration_measurement_count": len(concentration),
        },
        "development_separation_audit": {
            "source_asymmetry": {
                "known_positive_count": sum(
                    str(row.get("source_asymmetry", {}).get("status", "")).startswith("SUPPORTED")
                    for row in cases if row.get("development_role") != "CENTERED_CONTROL"
                ),
                "centered_control_count": source_control_count,
                "centered_control_total": centered_control_count,
                "admitted_to_formation_predictor": (
                    source_control_count == centered_control_count
                ),
            },
            "source_transport_coupling": {
                "measured_control_count": transport_control_count,
                "centered_control_total": centered_control_count,
                "admitted_to_formation_predictor": (
                    transport_control_count == centered_control_count
                ),
            },
            "transport_concentration": {
                "admitted_to_standalone_predictor": False,
                "reason": "overlapping known/control ranges",
            },
            "carrier_stability": {
                "measured_control_count": stability_control_count,
                "centered_control_total": centered_control_count,
                "admitted_to_short_screen_confirmation": (
                    stability_control_count == centered_control_count
                ),
            },
        },
        "properties": {
            "source_asymmetry": {
                "decision": "RETAIN_CONDITIONAL_FORMATION_BRANCH",
                "role": "source_formation_prior",
                "required_measurements": [
                    "declared semantic orbit or source decomposition",
                    "conditional signed mean with frozen reference margin",
                    "variance/norm preservation for any intervention",
                ],
                "positive_rule": "source conditional mean is above its frozen margin and the exact boundary is finite",
                "negative_rule": "source conditional mean is inside the frozen equivalence margin",
                "missing_rule": "missing orbit or source atoms yields ABSTAIN, never SAFE",
                "not_a_global_claim": True,
            },
            "source_transport_coupling": {
                "decision": "RETAIN_CASE_LEVEL_FORMATION_BRANCH",
                "role": "transport_formation_test",
                "required_measurements": [
                    "complete declared VJP transport boundary",
                    "residual and transport marginals",
                    "paired permutation intervention and sham",
                ],
                "positive_rule": "natural pairing is directional and matched pairing permutation suppresses it while preserving declared marginals",
                "negative_rule": "pairing permutation does not suppress the directional component",
                "missing_rule": "incomplete transport or non-marginal-preserving repair yields ABSTAIN",
                "not_a_global_claim": True,
            },
            "transport_concentration": {
                "decision": "RETAIN_AS_SUPPORTING_FEATURE_ONLY",
                "role": "screen_priority_tie_breaker",
                "required_measurements": ["complete cross-state Gram or streamed equivalent"],
                "forbidden_use": "must not be a standalone positive/negative verdict",
                "reason": "development controls overlap the known-case top-eigen and effective-rank ranges",
                "observed_rows": concentration,
            },
            "carrier_stability": {
                "decision": "RETAIN_AS_SHORT_TRAJECTORY_CONFIRMATION",
                "role": "persistence_consequence_gate",
                "required_measurements": [
                    "ordered reference states",
                    "effective-update residual vector per state",
                    "shared signed CountSketch protocol",
                ],
                "positive_rule": "short screen exceeds its frozen sign-flip null with positive short-lag correlation and late prefix growth",
                "negative_rule": "null-like screen is not a proof of safety; it only avoids expensive confirmation",
                "missing_rule": "incomplete vector path yields ABSTAIN",
            },
        },
        "oracle_input_whitelist": {
            "formation": [
                "source_asymmetry_measurements",
                "source_transport_coupling_measurements",
                "declared semantic shape/contract metadata",
            ],
            "short_screen": [
                "ordered reference state IDs",
                "effective-update residual vectors or declared sketch handles",
                "fixed CountSketch seed and dimension",
                "sign-flip null seed and draw count",
            ],
            "forbidden": [
                "T1-T4 verdicts",
                "SEUP verdicts",
                "final drift or loss trajectory",
                "case names or historical labels",
                "held-out outcome values",
            ],
        },
        "heldout_gate": {
            "freeze_required_before_reveal": True,
            "short_screen_positive_requires_exact_confirmation": True,
            "centered_controls_remain_in_denominator": True,
            "no_property_is_claimed_universal": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    payload = build(_load(args.profile))
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / "property_freeze_v1.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Development property freeze v1",
        "",
        "Status: `FROZEN_BEFORE_HELDOUT`. This freezes scope and measurement rules; it does not claim a universal property.",
        "",
        "| property | decision | role |",
        "|---|---|---|",
    ]
    for name, spec in payload["properties"].items():
        lines.append(f"| `{name}` | `{spec['decision']}` | {spec['role']} |")
    lines += [
        "",
        "Concentration is explicitly supporting-only because its development ranges overlap centered controls.",
        "Source asymmetry is the only formation candidate currently admitted by a complete centered-control comparison.",
        "Source--transport has no measured control intervention, and carrier stability has no measured control trajectory; both remain explicitly unvalidated.",
        "Carrier stability is measured by the shared short ordered-vector screen and is not inferred from formation labels.",
        "Feedback-sustained and unresolved boundaries abstain rather than becoming negative examples.",
    ]
    (args.output_root / "property_freeze_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(target.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
