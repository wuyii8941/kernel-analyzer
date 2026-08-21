#!/usr/bin/env python3
"""Audit whether each development property separates controls before oracle use.

This is deliberately a mechanical audit.  It does not promote a property just
because it is measured on a known case: a property must have both positive
development evidence and a measured centered-control comparison.  Missing
control measurements remain missing and are reported as a blocker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "results/property/bias_property_search/development_property_profile.json"
DEFAULT_OUT = ROOT / "results/property/bias_property_search"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(cases: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        status = str(case.get(key, {}).get("status", "UNMEASURED"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def audit(profile: dict[str, Any]) -> dict[str, Any]:
    cases = list(profile.get("cases", []))
    controls = [case for case in cases if case.get("development_role") == "CENTERED_CONTROL"]
    known = [case for case in cases if case.get("development_role") != "CENTERED_CONTROL"]

    source_controls = [
        case for case in controls
        if case.get("source_asymmetry", {}).get("status") == "CENTERED_LOCAL_CONTROL"
    ]
    source_positive = [
        case for case in known
        if str(case.get("source_asymmetry", {}).get("status", "")).startswith("SUPPORTED")
    ]
    transport_positive = [
        case for case in known
        if str(case.get("source_transport_coupling", {}).get("status", "")).startswith("SUPPORTED")
    ]
    transport_controls = [
        case for case in controls
        if case.get("source_transport_coupling", {}).get("status")
        not in {None, "UNMEASURED", "UNMEASURED_OR_NOT_APPLICABLE"}
    ]
    concentration_controls = [
        case for case in controls
        if case.get("transport_concentration", {}).get("status") == "MEASURED"
    ]
    stability_positive = [
        case for case in known
        if case.get("carrier_stability", {}).get("status") == "MEASURED_TRAJECTORY_STABLE"
    ]
    stability_controls = [
        case for case in controls
        if case.get("carrier_stability", {}).get("status")
        in {"MEASURED_TRAJECTORY_STABLE", "MEASURED_TRAJECTORY_NOT_STABLE"}
    ]

    properties = {
        "source_asymmetry": {
            "known_positive_count": len(source_positive),
            "centered_control_count": len(source_controls),
            "control_total": len(controls),
            "separates_development": bool(source_positive and len(source_controls) == len(controls)),
            "oracle_eligibility": "CANDIDATE_FORMATION_INPUT" if source_positive and len(source_controls) == len(controls) else "ABSTAIN_UNVALIDATED",
        },
        "source_transport_coupling": {
            "known_positive_count": len(transport_positive),
            "measured_control_count": len(transport_controls),
            "control_total": len(controls),
            "separates_development": bool(transport_positive and len(transport_controls) == len(controls)),
            "oracle_eligibility": "CANDIDATE_FORMATION_INPUT" if transport_positive and len(transport_controls) == len(controls) else "CASE_LEVEL_ONLY_UNVALIDATED",
        },
        "transport_concentration": {
            "known_measured_count": sum(case.get("transport_concentration", {}).get("status") == "MEASURED" for case in known),
            "measured_control_count": len(concentration_controls),
            "control_total": len(controls),
            "separates_development": False,
            "oracle_eligibility": "SUPPORTING_FEATURE_ONLY",
            "reason": "known and centered-control ranges overlap; no standalone verdict",
        },
        "carrier_stability": {
            "known_positive_count": len(stability_positive),
            "measured_control_count": len(stability_controls),
            "control_total": len(controls),
            "separates_development": bool(stability_positive and len(stability_controls) == len(controls)),
            "oracle_eligibility": "SHORT_SCREEN_UNVALIDATED" if not len(stability_controls) == len(controls) else "SHORT_SCREEN_CANDIDATE",
        },
    }
    return {
        "schema": "kernel-analyzer-development-property-separation-audit-v1",
        "status": "COMPLETE_WITH_CONTROL_GAPS" if any(
            value["oracle_eligibility"] in {"ABSTAIN_UNVALIDATED", "CASE_LEVEL_ONLY_UNVALIDATED", "SHORT_SCREEN_UNVALIDATED"}
            for value in properties.values()
        ) else "COMPLETE",
        "profile": str(DEFAULT_PROFILE.relative_to(ROOT)),
        "development_case_count": len(cases),
        "centered_control_count": len(controls),
        "known_case_count": len(known),
        "property_status_counts": {
            key: _status_counts(cases, key)
            for key in ("source_asymmetry", "source_transport_coupling", "transport_concentration", "carrier_stability")
        },
        "properties": properties,
        "rule": "Only a property with positive evidence and complete measured centered controls may enter a held-out predictor; missing control measurements are not negative labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = audit(_load(args.profile))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "development_property_separation_audit_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Development property separation audit",
        "",
        f"Status: `{payload['status']}`.",
        "",
        "A property is not admitted to a held-out predictor merely because it has a positive case; centered controls must be measured under the same property definition.",
        "",
        "| property | known positives | measured controls | separation | oracle eligibility |",
        "|---|---:|---:|---|---|",
    ]
    for name, value in payload["properties"].items():
        positives = value.get("known_positive_count", value.get("known_measured_count", 0))
        controls_count = value.get("centered_control_count", value.get("measured_control_count", 0))
        lines.append(
            f"| `{name}` | {positives} | {controls_count} | "
            f"{value['separates_development']} | `{value['oracle_eligibility']}` |"
        )
    lines += [
        "",
        "Concentration is supporting-only by design.  Source--transport and carrier stability remain explicitly unvalidated against the five scope-extension controls until their required measurements are captured.",
    ]
    (args.output_root / "development_property_separation_audit_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
