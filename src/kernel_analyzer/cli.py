"""Command-line interface for reproducible automated analyses."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from .api import AnalysisSpec, TIERS
from .artifacts import digest, read_json
from .runner import Analyzer
from .tcmp_campaign import (
    TCMPDisposition,
    audit_denominator,
    benjamini_hochberg,
    exact_sign_flip_statistics,
    holm_rejections,
)


def load_spec(path: Path) -> AnalysisSpec:
    module_spec = importlib.util.spec_from_file_location("kernel_analyzer_user_spec", str(path))
    if module_spec is None or module_spec.loader is None:
        raise ValueError("cannot import analysis spec: %s" % path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, "build_spec", None)
    if not callable(factory):
        raise ValueError("spec module must define build_spec()")
    value = factory()
    if not isinstance(value, AnalysisSpec):
        raise TypeError("build_spec() must return AnalysisSpec")
    return value


def verify_run(run_dir: Path) -> Mapping[str, Any]:
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    expected = report.pop("report_sha256")
    if digest(report) != expected:
        raise ValueError("report self-digest differs")
    proof = read_json(run_dir / "proof_units.json.gz")
    units = proof["proof_units"]
    ids = [row.get("unit_id", row.get("proof_unit_id")) for row in units]
    if len(ids) != len(set(ids)):
        raise ValueError("proof-unit IDs are not unique")
    for path in sorted((run_dir / "candidates").glob("*.json.gz")):
        data = read_json(path)
        for certificate in data["certificates"]:
            blocked = False
            for tier in TIERS:
                status = certificate["tiers"][tier]["status"]
                if status == "PASS" and blocked:
                    raise ValueError("out-of-order pass in %s" % certificate["case_id"])
                blocked = blocked or status != "PASS"
    return {
        "status": "VALID",
        "run_id": report["run_id"],
        "proof_unit_count": len(units),
        "complete_case_count": len(report["case_certificates"]),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="kernel-analyzer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "resume"):
        command = sub.add_parser(name)
        command.add_argument("spec", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("run_dir", type=Path)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("run_dir", type=Path)
    tcmp = sub.add_parser("tcmp")
    tcmp_sub = tcmp.add_subparsers(dest="tcmp_command", required=True)
    inventory = tcmp_sub.add_parser("inventory")
    inventory.add_argument("campaign", type=Path)
    screen = tcmp_sub.add_parser("screen")
    screen.add_argument("input", type=Path)
    confirm = tcmp_sub.add_parser("confirm")
    confirm.add_argument("input", type=Path)
    audit = tcmp_sub.add_parser("audit")
    audit.add_argument("ledger", type=Path)
    audit.add_argument("dispositions", type=Path)
    args = parser.parse_args(argv)

    if args.command in {"analyze", "resume"}:
        spec = load_spec(args.spec)
        report = Analyzer().analyze(spec, resume=args.command == "resume")
        print(json.dumps(report.as_dict(), sort_keys=True))
    elif args.command == "verify":
        print(json.dumps(verify_run(args.run_dir), sort_keys=True))
    elif args.command == "coverage":
        report = json.loads((args.run_dir / "report.json").read_text())
        print(json.dumps({
            "run_id": report["run_id"],
            "proof_units": report["proof_unit_count"],
            "unresolved": report["unresolved_proof_units"],
            "candidates": report["candidate_summaries"],
        }, indent=2, sort_keys=True))
    elif args.tcmp_command == "inventory":
        campaign = json.loads(args.campaign.read_text())
        cells = campaign.get("cells", [])
        ids = [row.get("cell_id") for row in cells]
        if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("campaign has missing or duplicate cell IDs")
        print(json.dumps({
            "status": "VALID_CAMPAIGN_INVENTORY",
            "campaign_id": campaign.get("campaign_id"),
            "cells": len(cells),
            "expansion_cells": sum(row.get("phase") == "EXPANSION" for row in cells),
            "heldout_cells": sum(row.get("phase") == "HELDOUT_CONFIRMATION" for row in cells),
        }, sort_keys=True))
    elif args.tcmp_command in {"screen", "confirm"}:
        payload = json.loads(args.input.read_text())
        rows = payload.get("rows", [])
        statistics = {row["unit_id"]: exact_sign_flip_statistics(row["gram"]) for row in rows}
        p_values = {key: value["one_sided_p"] for key, value in statistics.items()}
        selected = (
            benjamini_hochberg(p_values, 0.10)
            if args.tcmp_command == "screen"
            else holm_rejections(p_values, 0.05)
        )
        print(json.dumps({
            "status": "COMPLETE",
            "mode": args.tcmp_command.upper(),
            "statistics": statistics,
            "selected": selected,
        }, sort_keys=True))
    else:
        ledger = read_json(args.ledger)
        payload = read_json(args.dispositions)
        rows = [TCMPDisposition(**row) for row in payload["rows"]]
        print(json.dumps(audit_denominator(ledger["invocation_ids"], rows), sort_keys=True))


if __name__ == "__main__":
    main()
