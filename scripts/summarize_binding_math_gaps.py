#!/usr/bin/env python3
"""Inventory failed proof obligations, not numerical bias verdicts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from scripts.run_training_numerical_v2 import save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("math", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with gzip.open(args.math, "rt") as handle:
        data = json.load(handle)
    pending = [u for u in data["units"] if not (u.get("composite_vjp_proof") or {}).get("passed")]
    counts = Counter((tuple(u["forward_program"]), tuple(u["actual_vjp_program"])) for u in pending)
    save_new(args.output, {
        "status": "MATHEMATICAL_BINDING_INCOMPLETE" if pending else "UNIT_PROOFS_COMPLETE_CHECK_REMAINING_GATES",
        "source_sha256": hashlib.sha256(args.math.read_bytes()).hexdigest(),
        "missing_formula_targets": data["missing_formula_targets"],
        "failed_gates": [key for key, value in data["gates"].items() if not value],
        "unproved_units": len(pending),
        "program_patterns": [{"forward": list(f), "backward": list(b), "count": n}
                             for (f, b), n in sorted(counts.items())],
        "interpretation": "Missing checker derivations, not proof that the implementation is mathematically wrong or free of bias.",
        "next_action": "Add general exact-input/constant/shape-checked derivations and mutation tests; do not bypass the global proof gate.",
    })


if __name__ == "__main__":
    main()
