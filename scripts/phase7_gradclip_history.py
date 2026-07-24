#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.stats import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Trainer global gradient norms around a clip threshold.")
    parser.add_argument("--trainer-state", required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--out", default="results/phase7_gradclip_ref_history.json")
    args = parser.parse_args()

    state = json.loads(Path(args.trainer_state).read_text(encoding="utf-8"))
    rows = [row for row in state["log_history"] if "grad_norm" in row]
    norms = [float(row["grad_norm"]) for row in rows]
    margins = [abs(value - args.threshold) for value in norms]
    positive = [value for value in norms if value > 0.0]
    closest = min(range(len(norms)), key=lambda index: margins[index])
    result = {
        "schema_version": "forkcert.phase7.gradclip_history.v1",
        "source": args.trainer_state,
        "path": "training HF eager path (Trainer logged pre-clip global norm)",
        "threshold": args.threshold,
        "steps": len(norms),
        "trigger_steps": sum(value > args.threshold for value in norms),
        "nontrigger_steps": sum(value <= args.threshold for value in norms),
        "zero_norm_steps": sum(value == 0.0 for value in norms),
        "minimum_positive_norm": min(positive) if positive else None,
        "margin_min": margins[closest],
        "closest_log_index": closest + 1,
        "closest_norm": norms[closest],
        "margin_p1": percentile(margins, 1),
        "margin_p5": percentile(margins, 5),
        "margin_p50": percentile(margins, 50),
        "near_boundary_counts": {
            str(value): sum(margin < value for margin in margins)
            for value in [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
        },
        "scope": (
            "Demand-side coverage for the eager training path only; it does not replace a paired backend scan "
            "or an analytic legal bound."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
