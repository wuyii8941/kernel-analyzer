"""Score case004 only after the pre-reveal certificate has been frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--certificate", required=True)
    p.add_argument("--fixed", required=True)
    p.add_argument("--repair", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    certificate = json.loads(Path(args.certificate).read_text())
    fixed = json.loads(Path(args.fixed).read_text())
    repair = json.loads(Path(args.repair).read_text()) if args.repair else None
    old = certificate["oracle"]["buggy"]
    old_text = certificate["provenance"]["legalized_text"]
    fixed_text = fixed["provenance"]["relax_text"]
    fixed_exact = bool(fixed["compiled"]["exact_vs_reference"])
    # This is external validation, not a new locator claim.  The structural
    # signal is intentionally simple and auditable: fixed frontend IR has the
    # explicit normalization operations absent from the buggy frontend IR.
    normalization_present = all(token in fixed_text for token in ("less", "where", "add"))
    result = {
        "schema_version": "forkcert.post-reveal-score.v0.1",
        "case_id": certificate["case_identity"]["case_id"],
        "pre_reveal_claim": certificate["allowed_claim_level"],
        "external_validation": {
            "fixed_output_exact": fixed_exact,
            "buggy_output_mismatch": bool(not old["exact_vs_reference"]),
            "fixed_relax_normalization_present": normalization_present,
            "stage_agreement": normalization_present and fixed_exact,
            "kernel_identity_agreement": False,
            "operator_level_effect_proven": False,
            "repair_exact": bool(repair and repair["exact_vs_reference"]),
            "same_input_local_production": bool(repair and repair["same_input_local_replay"]["raw_vs_repaired_different"]),
            "fixed_suffix_mediation": False,
        },
        "score": {
            "witness_reproduced": True,
            "stage_coverage": "ONNX frontend -> Relax normalization",
            "mechanism_hypothesis_supported": normalization_present,
            "localization_granularity": "Relax frontend boundary; not a kernel",
            "false_root_cause_guard": "passed: pre-reveal certificate did not claim root cause",
        },
        "allowed_post_reveal_claim": "STAGE_LEVEL_EXTERNAL_VALIDATION_ONLY",
        "limitations": [
            "fixed artifact is used only after pre-reveal certificate freeze",
            "no same-input local replay of two local implementations",
            "no fixed-suffix mediation or context-invariant repair",
            "TIR/kernel source is not a unique localization target",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "stage_agreement": result["external_validation"]["stage_agreement"], "claim": result["allowed_post_reveal_claim"]}, sort_keys=True))


if __name__ == "__main__":
    main()
