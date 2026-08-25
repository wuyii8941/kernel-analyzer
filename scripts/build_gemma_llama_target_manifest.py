#!/usr/bin/env python3
"""Freeze a small, result-blind Gemma/Llama target replay set.

The existing pattern screen only says which exact implementation identities
look directional.  This helper binds those identities to a concrete generated
forward/backward region and an endpoint in the frozen campaign.  It does not
call a verdict positive: every row still needs a legal parameter-reachable
replay and the same long-horizon consequence test used by the existing cases.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    # One row per distinct operation/phase family, selected only from the
    # frozen screen ordering.  We deliberately include both forward and
    # backward examples and do not filter by model-name or expected outcome.
    specs = [
        {
            "model": "Gemma-4 E2B",
            "architecture": "gemma4",
            "model_path": "/data1/tzh/models/google/gemma-4-E2B",
            "input_bank": "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json",
            "consequence_bank": "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json",
            "screen": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/pattern_screen.json",
            "census": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/implementation_census.json",
            "campaign": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/runtime_release/campaign.json.gz",
            "carrier": "model.language_model.per_layer_model_projection.weight",
            "case_prefix": "gemma4_scan",
            "max_targets": 3,
        },
        {
            "model": "Llama-3.2-3B",
            "architecture": "generic",
            "model_path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
            "input_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128.json",
            "consequence_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128_trajectory4096.json",
            "screen": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/pattern_screen.json",
            "census": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/implementation_census.json",
            "campaign": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/campaign.json.gz",
            "carrier": "model.norm.weight",
            "case_prefix": "llama32_scan",
            "max_targets": 3,
        },
    ]

    rows = []
    for spec in specs:
        screen = json.loads((root / spec["screen"]).read_text())["rows"]
        census = json.loads((root / spec["census"]).read_text())["implementations"]
        identities = {row["exact_implementation_id"]: row for row in census}
        with gzip.open(root / spec["campaign"], "rt", encoding="utf-8") as handle:
            campaign = json.load(handle)["rows"]
        chosen = []
        seen = set()
        for screen_row in sorted(
            screen, key=lambda row: (-float(row.get("amplification", 0.0)), float(row.get("p_value", 1.0)))
        ):
            identity = identities.get(screen_row.get("representative_exact_implementation_id"))
            if identity is None:
                continue
            exact = identity["identity"]["exact_payload"]
            operation = exact.get("operation")
            phase = exact.get("phase")
            endpoint = screen_row.get("endpoint")
            matches = [
                row for row in campaign
                if row.get("phase") == phase
                and row.get("symbol") == operation
                and endpoint in row.get("output_names", [])
            ]
            if not matches or (phase, operation, endpoint) in seen:
                continue
            target = matches[0]
            seen.add((phase, operation, endpoint))
            rows.append({
                "case_id": f"{spec['case_prefix']}_{len(chosen):02d}",
                "model": spec["model"],
                "architecture": spec["architecture"],
                "model_path": spec["model_path"],
                "input_bank": spec["input_bank"],
                "consequence_bank": spec["consequence_bank"],
                "carrier": spec["carrier"],
                "target_region": target["region_id"],
                "target_endpoint": endpoint,
                "target_symbol": target["symbol"],
                "source_screen": spec["screen"],
                "screen_exact_implementation_id": screen_row["representative_exact_implementation_id"],
                "screen_amplification": screen_row.get("amplification"),
                "screen_p_value": screen_row.get("p_value"),
                "screen_rms_mean": screen_row.get("rms_mean"),
                "binding_rule": "same phase+generated symbol+screened endpoint; first frozen campaign region",
                "status": "PENDING_PARAMETER_REACHABLE_REPLAY",
            })
            chosen.append(target)
            if len(chosen) >= int(spec["max_targets"]):
                break

    payload = {
        "schema": "kernel-analyzer-gemma-llama-target-replay-manifest-v1",
        "status": "FROZEN_TARGET_REPLAY_SET",
        "selection_rule": "highest-amplification rows from the frozen pattern screen, one per phase/operation/endpoint family, before any target replay result is read",
        "rows": rows,
    }
    payload["manifest_sha256"] = sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
