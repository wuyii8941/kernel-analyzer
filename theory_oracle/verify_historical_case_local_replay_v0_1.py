#!/usr/bin/env python
"""Independent audit for the blind historical-case local-replay slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_matches_eager(report: dict[str, Any], manifest_override: Path | None = None) -> bool | None:
    """Recheck the frozen reference for reports written by older runners.

    Early reports did not persist ``eager_matches_reference_artifact`` inside
    ``complete_witness``.  We do not silently accept that omission: when the
    case manifest is available, reload its reference tensor and compare the
    same content hash used by the runner.
    """

    witness = report["complete_witness"]
    if "eager_matches_reference_artifact" in witness:
        return bool(witness["eager_matches_reference_artifact"])
    manifest_path = str(manifest_override) if manifest_override else report.get("case_manifest", {}).get("path")
    if not manifest_path:
        return None
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        reference_path = Path(manifest["reference"]["path"])
        if not reference_path.exists():
            return None
        import torch

        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        raw = reference.detach().contiguous().numpy().tobytes()
        import hashlib

        reference_hash = hashlib.sha256(raw).hexdigest()
        return reference_hash == witness["eager"]["sha256"]
    except Exception:
        return None


def provenance_tokens(report: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for stage in report["provenance"].values():
        for artifact in stage["artifacts"]:
            tokens.update(artifact.get("source_nodes", []))
            tokens.update(artifact.get("original_aten", []))
            tokens.update(artifact.get("kernel_paths", []))
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    errors: list[str] = []
    if first.get("schema_version") != "forkcert.historical_case_local_replay.v0.1":
        errors.append("unexpected first schema")
    if second.get("schema_version") != first.get("schema_version"):
        errors.append("independent reports use different schemas")
    if first.get("environment") != second.get("environment"):
        errors.append("independent environments differ")
    for report in (first, second):
        witness = report["complete_witness"]
        local = report["local_replay"]
        gates = report["gates"]
        if not witness["complete_witness"] if "complete_witness" in witness else False:
            errors.append("malformed witness")
        if not witness["compiled_repeat_exact"]:
            errors.append("compiled repeat is not exact")
        if witness["eager_vs_compiled"]["max_abs"] <= 0:
            errors.append("no complete-program discrepancy")
        eager_matches = reference_matches_eager(report, args.case_manifest)
        if eager_matches is not True:
            errors.append("eager output does not match frozen reference artifact")
        if not local["boundary_inputs_exact"]:
            errors.append("same-input local replay input gate failed")
        if not local["production_observed"]:
            errors.append("local production was not observed")
        if not local["compiled_pool_exact"]:
            errors.append("pool-only control is not exact")
        if local["mediation_observed"]:
            errors.append("expected boundary mediation-negative control changed")
        if gates.get("allowed_claim_level"):
            errors.append("unexpected nested claim field")
        if report.get("allowed_claim_level") != "LOCAL_INJECTION":
            errors.append("claim level is not the preregistered local-injection level")
        if not gates["provenance_complete"]:
            errors.append("provenance gate failed")
        if gates["intervention_executed"] or gates["oracle_recomputed"]:
            errors.append("slice incorrectly claims an intervention")
        for stage, value in report["provenance"].items():
            if value.get("artifact_count", 0) <= 0:
                errors.append(f"missing provenance artifacts for {stage}")
    if first["complete_witness"] != second["complete_witness"]:
        errors.append("complete witness differs across independent processes")
    first_tokens = provenance_tokens(first)
    second_tokens = provenance_tokens(second)
    if not any("adaptive_avg_pool2d" in token for token in first_tokens):
        errors.append("provenance lacks adaptive_avg_pool2d")
    if not any("aten.sum" in token for token in first_tokens):
        errors.append("provenance lacks aten.sum")
    if not (first_tokens & second_tokens):
        errors.append("independent provenance has no common source/kind evidence")

    audit = {
        "schema_version": "forkcert.historical_case_local_replay_audit.v0.1",
        "inputs": [str(Path(args.first).resolve()), str(Path(args.second).resolve())],
        "valid": not errors,
        "errors": errors,
        "evidence_level": "LOCAL_PRODUCER_WITH_PROVENANCE" if not errors else "INVALID",
        "allowed_claim_level": "LOCAL_INJECTION" if not errors else "INVALID",
        "production_observed": bool(first["local_replay"]["production_observed"]),
        "mediation_observed": bool(first["local_replay"]["mediation_observed"]),
        "pool_only_exact": bool(first["local_replay"]["compiled_pool_exact"]),
        "complete_repeat_exact": bool(first["complete_witness"]["compiled_repeat_exact"]),
        "limitations": [
            "no repair/injection or non-target context comparison",
            "local suffix replay establishes a producer candidate, not a unique root cause",
            "historical patch remains hidden for later external scoring",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
