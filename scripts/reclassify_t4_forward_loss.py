#!/usr/bin/env python3
"""Move the non-required forward-loss diagnostic out of strict T4 gates."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = ROOT / "results/coverage/cases/trajectory"
    changed = []
    for path in sorted(root.glob("qwen_seq*_r1/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        gates = payload.get("gates", {})
        if "forward_loss_unchanged" not in gates:
            continue
        previous = payload.get("status")
        diagnostic = bool(gates.pop("forward_loss_unchanged"))
        payload.setdefault("diagnostics", {})["forward_loss_unchanged"] = diagnostic
        payload["previous_status_before_protocol_correction"] = previous
        payload["protocol_correction"] = (
            "forward loss equality is descriptive, not one of the five required "
            "Flash-style gates in docs/bias_protocol.md"
        )
        payload["status"] = (
            "PASS_T4_PAIRED_ACCUMULATION"
            if all(gates.values()) else "FAIL_DIRECTIONAL_ACCUMULATION"
        )
        payload.pop("result_sha256", None)
        payload["result_sha256"] = canonical(payload)
        temporary = path.with_name("." + path.name + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        temporary.replace(path)
        changed.append({"path": str(path), "before": previous,
                        "after": payload["status"]})
    print(json.dumps({"changed": changed}, sort_keys=True))


if __name__ == "__main__":
    main()
