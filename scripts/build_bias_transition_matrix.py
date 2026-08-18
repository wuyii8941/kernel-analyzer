#!/usr/bin/env python3
"""Reduce v2.1 formation certificates into the Bias Formation Map.

Only v2.1 open-loop certificates are accepted as formation evidence.  Missing
certificates remain PENDING; historical T1--T4/SEUP artifacts are never read
by this reducer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/property/bias_formation/protocol.json"
DEFAULT_OUTPUT = ROOT / "results/property/bias_formation/bias_transition_matrix.csv"

LAYERS = (
    ("LOCAL_ENDPOINT", "local"),
    ("PARAMETER_GRADIENT", "gradient"),
    ("EFFECTIVE_UPDATE", "update"),
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("certificate must be an object")
    return value


def _status(cert: dict[str, Any], layer: str) -> str:
    populations = cert.get("populations")
    if not isinstance(populations, dict):
        return "UNRESOLVED"
    confirmation = populations.get("confirmation")
    if not isinstance(confirmation, dict):
        return "UNRESOLVED"
    # v2.1 population keys preserve the enum spelling (for example
    # ``LOCAL_ENDPOINT_status``).  Do not lower-case this lookup: doing so
    # silently turns every measured layer into UNRESOLVED/PENDING.
    value = confirmation.get(layer + "_status")
    if value == "BIASED":
        return "{}{}_BIAS".format("LOCAL" if layer == "LOCAL_ENDPOINT" else "GRADIENT" if layer == "PARAMETER_GRADIENT" else "UPDATE", "")
    if value == "CENTERED":
        return "{}{}_CENTERED".format("LOCAL" if layer == "LOCAL_ENDPOINT" else "GRADIENT" if layer == "PARAMETER_GRADIENT" else "UPDATE", "")
    if value == "CANCELING_STRUCTURE":
        return "{}{}_CANCELING".format("LOCAL" if layer == "LOCAL_ENDPOINT" else "GRADIENT" if layer == "PARAMETER_GRADIENT" else "UPDATE", "")
    return "UNRESOLVED"


def _validate_certificate(cert: dict[str, Any]) -> None:
    if cert.get("schema") != "kernel-analyzer-bias-formation-certificate-v2_1":
        raise ValueError("only v2.1 formation certificates are accepted")
    if cert.get("measurement_kind") != "candidate_repair_ground_truth":
        raise ValueError("certificate is not a formation ground-truth measurement")
    if cert.get("uses_historical_verdicts") is not False or cert.get("verdict_blind") is not True:
        raise ValueError("certificate provenance is not verdict-blind")
    if cert.get("trajectory_drift_in_formation") is not False:
        raise ValueError("trajectory drift cannot be used as formation evidence")


def _mechanism(local: str, gradient: str, update: str, certificate: dict[str, Any] | None) -> str:
    if certificate is None:
        return "PENDING_MEASUREMENT"
    if "UNRESOLVED" in (local, gradient, update):
        return "UNRESOLVED"
    if local == "LOCAL_BIAS":
        return "SOURCE_CANDIDATE"
    if local == "LOCAL_CENTERED" and gradient == "GRADIENT_BIAS":
        return "TRANSPORT_OR_CONTRACT_CANDIDATE"
    if local == "LOCAL_CENTERED" and gradient == "GRADIENT_CENTERED" and update == "UPDATE_BIAS":
        return "OPTIMIZER_CANDIDATE"
    if all(value in {"LOCAL_CENTERED", "GRADIENT_CENTERED", "UPDATE_CENTERED"} for value in (local, gradient, update)):
        return "VARIANCE_ONLY_CANDIDATE"
    return "UNRESOLVED"


def build(certificate_paths: dict[str, Path] | None = None) -> list[dict[str, str]]:
    protocol = _load(PROTOCOL)
    rows: list[dict[str, str]] = []
    for case in protocol.get("cases", []):
        case_id = str(case["case_id"])
        path = (certificate_paths or {}).get(case_id)
        certificate = _load(path) if path is not None else None
        if certificate is not None:
            _validate_certificate(certificate)
        statuses = [_status(certificate, layer) if certificate is not None else "PENDING" for layer, _ in LAYERS]
        local, gradient, update = statuses
        rows.append({
            "case": case_id,
            "local": local,
            "gradient": gradient,
            "update": update,
            "mechanism_candidate": _mechanism(local, gradient, update, certificate),
            "certificate_status": "" if certificate is None else str(certificate.get("status", "UNRESOLVED")),
            "certificate_path": "" if path is None else str(path),
            "formation_label_source": "v2_1_open_loop_certificate" if certificate is not None else "NOT_MEASURED",
            "claim_boundary": "Transition only; mechanism requires its declared intervention and no T4/SEUP label is used.",
        })
    return rows


def write(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "local", "gradient", "update", "mechanism_candidate", "certificate_status", "certificate_path", "formation_label_source", "claim_boundary"]
    temporary = output.with_name("." + output.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    for binding in args.certificate:
        if "=" not in binding:
            raise SystemExit("--certificate expects CASE=PATH")
        case, path = binding.split("=", 1)
        paths[case] = Path(path)
    rows = build(paths)
    write(rows, args.output)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "pending": sum(r["local"] == "PENDING" for r in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
