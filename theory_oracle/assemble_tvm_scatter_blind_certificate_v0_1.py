"""Assemble the pre-reveal certificate for the TVM ScatterElements case.

Only buggy-run artifacts are accepted.  The fixed checkout is intentionally
not an input to this command.  This makes the certificate suitable for a
patch-excluded replay even though the candidate itself is not a fully blind
benchmark (the fix was read during screening).
"""

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
    baseline = json.loads(Path(args.baseline).read_text())
    repeat = json.loads(Path(args.repeat).read_text())

    stable_reference = baseline["reference"] == repeat["reference"]
    stable_compiled = baseline["compiled"]["output"] == repeat["compiled"]["output"]
    witness = (
        baseline["compiled"]["exact_vs_reference"] is False
        and stable_reference
        and stable_compiled
    )
    # The frontend/TIR mismatch is an IR-level observation.  It is not a
    # same-input numeric replay of two local callables, so producer evidence
    # remains explicitly uninstantiated.
    gates = EvidenceGates(
        complete_witness=witness,
        same_input_local_replay=False,
        local_discrepancy_reproducible=False,
        provenance_complete=False,
        candidate_realization_preserved=False,
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=False,
        lower_level_replay=False,
        first_bad_stage_isolated=False,
        # Two independent baseline runs are the no-op/repeat control for this
        # CPU case.  A numeric local-replay control is still missing.
        null_controls_valid=True,
    )
    report = {
        "schema_version": "forkcert.operator-evidence.v0.1",
        "case_identity": {
            "case_id": baseline["case_id"],
            "blind_status": baseline["blind_status"],
            "tvm_root": baseline["tvm_root"],
            "tvm_version": baseline["tvm_version"],
            "target": baseline["target"],
        },
        "region_inventory": [
            {
                "region_id": "onnx_to_relax_scatter_elements",
                "source": baseline["provenance"]["onnx_node"],
                "compiled_region": baseline["provenance"]["compiled_region"],
                "replayability": "IR boundary only",
            },
            {
                "region_id": "relax_to_tir_scatter_elements",
                "source": "relax.scatter_elements",
                "compiled_region": "TIR prim_func scatter_elements",
                "replayability": "artifact inspection only",
            },
        ],
        "local_replay": {
            "same_input_numeric_replay": "UNINSTANTIATED",
            "ir_boundary_observation": baseline["production"],
            "tir_structural_signal": baseline["tir_mechanism_signal"],
        },
        "provenance": baseline["provenance"],
        "intervention": {
            **baseline["intervention"],
            "before": baseline["compiled"],
            "after": baseline["repaired"],
            "repeat_control": {
                "reference_stable": stable_reference,
                "compiled_stable": stable_compiled,
            },
        },
        "oracle": {
            "declared_endpoint": baseline["semantic_spec"],
            "compiled": baseline["compiled"],
            "repair": baseline["repaired"],
        },
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "limitations": [
            "candidate fix was read during screening; this is not a fully blind benchmark",
            "IR observation is not same-input numeric local production evidence",
            "non-target compiler context is not invariant after IR rebuild",
            "kernel identity and source-line provenance are not captured",
            "no correctness claim beyond the declared ONNX Runtime contract",
        ],
        "mechanism_hypothesis": {
            "text": "the ONNX reduction attribute is not preserved at the Relax boundary",
            "status": "hypothesis_supported_by_IR_and_endpoint_intervention_not_root_cause",
        },
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
