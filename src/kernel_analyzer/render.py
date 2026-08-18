"""Compact human-readable mathematical and case dossiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import CaseCertificate, TIERS


def render_mathematics(path: Path, templates: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Mathematical forward/backward templates", ""]
    for row in sorted(templates, key=lambda value: str(
            value.get("template_id", value.get("rule_id", "")))):
        derivation = row.get("derivation", row)
        identifier = row.get("template_id", row.get("rule_id", "UNRESOLVED"))
        lines.extend([
            "## `%s`" % identifier,
            "",
            "Overload: `%s`" % row.get("overload", derivation.get("overload", "")),
            "",
            "Forward:", "",
            str(derivation.get("exact_forward_map", derivation.get("forward_map", "UNRESOLVED"))),
            "", "Actual analytic VJP:", "",
            str(derivation.get("exact_vjp_map", derivation.get("vjp_map", "UNRESOLVED"))),
            "", "Finite arithmetic:", "",
            str(derivation.get("finite_arithmetic_realization",
                               derivation.get("finite_arithmetic", "UNRESOLVED"))),
            "", "Error relation:", "",
            (str(derivation.get("forward_error_relation", "")) + " " +
             str(derivation.get("vjp_error_relation",
                                derivation.get("error_relation", "")))).strip(),
            "",
        ])
    path.write_text("\n".join(lines) + "\n")


def render_case(path: Path, certificate: CaseCertificate) -> None:
    lines = [
        "# Directional-bias case", "",
        "- Case: `%s`" % certificate.case_id,
        "- F+B unit: `%s`" % certificate.proof_unit_id,
        "- Candidate: `%s`" % certificate.candidate_id,
        "- Natural: `%s`" % certificate.natural,
        "- Classification: `%s`" % certificate.classification,
        "",
    ]
    for tier in TIERS:
        evidence = certificate.tiers[tier]
        lines.extend([
            "## %s" % tier, "",
            "Status: `%s`" % evidence.status, "",
            "Reason: %s" % (evidence.reason or "—"), "",
            "Evidence: `%s`" % dict(evidence.evidence), "",
        ])
    path.write_text("\n".join(lines) + "\n")
