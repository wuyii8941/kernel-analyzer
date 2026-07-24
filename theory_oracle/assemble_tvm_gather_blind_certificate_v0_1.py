"""Build a fail-closed pre-reveal certificate from buggy Gather replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.operator_evidence import EvidenceGates, allowed_claim_level, validate_evidence_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--repeat", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    a, b = (json.loads(Path(path).read_text()) for path in (args.baseline, args.repeat))
    stable = a["reference"] == b["reference"] and a["compiled"]["output"] == b["compiled"]["output"]
    witness = bool(not a["compiled"]["exact_vs_reference"] and stable and a["positive_control"]["exact_vs_reference"])
    gates = EvidenceGates(
        complete_witness=witness,
        same_input_local_replay=False,
        local_discrepancy_reproducible=False,
        provenance_complete=False,
        candidate_realization_preserved=False,
        intervention_executed=False,
        oracle_recomputed=False,
        non_target_context_invariant=False,
        lower_level_replay=False,
        first_bad_stage_isolated=False,
        null_controls_valid=stable,
    )
    report = {
        "schema_version": "forkcert.operator-evidence.v0.1",
        "case_identity": {"case_id": a["case_id"], "blind_status": a["blind_status"], "tvm_root": a["tvm_root"], "tvm_version": a["tvm_version"], "target": a["target"]},
        "region_inventory": [{"region_id": "onnx_gather_to_relax_take", "source": a["provenance"]["onnx_node"], "compiled_region": "Relax take -> TIR take", "replayability": "artifact inspection"}],
        "local_replay": {"same_input_numeric_replay": "UNINSTANTIATED", "provenance_artifact": a["provenance"]},
        "provenance": a["provenance"],
        "intervention": {"executed": False, "reason": "pre-reveal certificate stops after observation"},
        "oracle": {"declared_endpoint": a["semantic_spec"], "buggy": a["compiled"], "positive_control": a["positive_control"]},
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "limitations": ["fixed revision/patch/root-cause discussion excluded", "no same-input local replay", "no repair or mediation", "kernel identity/context invariance not captured"],
    }
    errors = validate_evidence_report(report)
    if errors:
        raise SystemExit("invalid certificate: " + "; ".join(errors))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "allowed_claim_level": report["allowed_claim_level"], "witness": witness}, sort_keys=True))


if __name__ == "__main__":
    main()
