#!/usr/bin/env python3
"""Migrate bounded v2.1 open-loop artifacts to explicit inconclusive status.

Older captures used ``UNRESOLVED_INSUFFICIENT_STATES`` for both genuinely
short populations and populations with enough states whose confidence interval
straddled the frozen margins.  This migration only changes the latter label;
it never recomputes a verdict or changes a Gram statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OLD = "UNRESOLVED_INSUFFICIENT_STATES"
NEW = "UNRESOLVED_INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    changed = 0
    for path in sorted(args.root.glob("**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        populations = payload.get("populations", {})
        local_changed = False
        for partition in ("calibration", "confirmation"):
            for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
                block = populations.get(partition, {}).get(layer)
                if not isinstance(block, dict):
                    continue
                if block.get("status") != OLD:
                    continue
                if int(block.get("state_count", 0)) < 16:
                    continue
                block["status"] = NEW
                populations[partition][layer + "_status"] = NEW
                block["status_note"] = (
                    "16 states were collected; the frozen interval was inconclusive."
                )
                local_changed = True
        if local_changed:
            statuses = [
                populations[partition].get(layer + "_status")
                for partition in ("calibration", "confirmation")
                for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
            ]
            payload["status"] = NEW if NEW in statuses else payload.get("status")
            payload["status_migration"] = {
                "from": OLD,
                "to": NEW,
                "reason": "enough states, unresolved frozen margin",
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            changed += 1
    print(json.dumps({"changed": changed, "root": str(args.root)}, sort_keys=True))


if __name__ == "__main__":
    main()
