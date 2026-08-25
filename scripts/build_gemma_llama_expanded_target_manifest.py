#!/usr/bin/env python3
"""Build a replay manifest for every nonzero Gemma/Llama scan row.

The pattern scan is not a training verdict.  This file only binds each
nonzero, result-blind scan row to a fresh-compile symbol family and records
rows for which no matching generated program exists.  The replay runner is
still responsible for parameter reachability, short screening, and the
4096-step outcome check.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def family(symbol: str | None) -> str:
    return re.sub(r"_\d+$", "", symbol or "")


SPECS = [
    {
        "model": "Llama-3.2-3B",
        "architecture": "generic",
        "model_path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "scan": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/pattern_screen.json",
        "campaign": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/campaign.json.gz",
        "input_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128.json",
        "consequence_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128_trajectory4096.json",
        "carrier": "model.embed_tokens.weight",
        "prefix": "llama32_text128_scan",
    },
    {
        "model": "Gemma-4 E2B",
        "architecture": "gemma4",
        "model_path": "/data1/tzh/models/google/gemma-4-E2B",
        "scan": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/pattern_screen.json",
        "eligibility": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/eligibility_freeze.json",
        "campaign": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/runtime_release/campaign.json.gz",
        "input_bank": "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json",
        "consequence_bank": "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json",
        "carrier": "model.language_model.per_layer_model_projection.weight",
        "prefix": "gemma4_text128_scan",
    },
    {
        "model": "Llama-3.2-3B (text512)",
        "architecture": "generic",
        "model_path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "scan": "results/property/declared_persistent_4096/llama32_3b_text512_pattern_screen.json",
        "campaign": "results/property/tcmp_allop_v1/heldout/llama32_3b_text512/campaign.json.gz",
        "input_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text512.json",
        "consequence_bank": "results/property/declared_persistent_4096/input_banks/llama32_3b_text512_trajectory4096_cycled.json",
        "carrier": "model.lm_head.weight",
        "prefix": "llama32_text512_scan",
    },
]


def load_scan(spec: dict) -> list[dict]:
    rows = json.loads((ROOT / spec["scan"]).read_text())["rows"]
    # Gemma's eligibility freeze is the authoritative residual-nonzero filter.
    # Llama's earlier freeze was not emitted, so use the same nonzero rule
    # applied by the published operator-scan audit.
    if spec.get("eligibility"):
        eligibility = json.loads((ROOT / spec["eligibility"]).read_text())
        allowed = {
            (r.get("implementation_pattern_id"), r.get("endpoint"), r.get("phase"))
            for r in eligibility.get("all_rows", [])
            if r.get("eligibility") == "ELIGIBLE_NONZERO_NEW_IMPL"
        }
        rows = [
            r for r in rows
            if (r.get("implementation_pattern_id"), r.get("endpoint"), r.get("phase")) in allowed
        ]
    else:
        rows = [r for r in rows if float(r.get("amplification", 0.0)) > 0.0]
    # Keep the frozen scan ordering.  Do not rank after looking at replay data.
    return rows


def main() -> None:
    output = ROOT / "results/property/declared_persistent_4096/operator_scan_target_manifest.json"
    rows: list[dict] = []
    blocked: list[dict] = []
    for spec in SPECS:
        scan_rows = load_scan(spec)
        with gzip.open(ROOT / spec["campaign"], "rt", encoding="utf-8") as handle:
            campaign = json.load(handle)["rows"]
        # Match by phase, implementation symbol family, and the captured
        # endpoint.  A symbol family is deliberately used because a fresh
        # compile may assign a different numeric suffix to the same program.
        for index, scan in enumerate(scan_rows):
            matches = [
                c for c in campaign
                if c.get("phase") == scan.get("phase")
                and family(c.get("symbol")) == family(scan.get("operation"))
                and scan.get("endpoint") in c.get("output_names", [])
            ]
            base = {
                "model": spec["model"],
                "architecture": spec["architecture"],
                "model_path": spec["model_path"],
                "input_bank": spec["input_bank"],
                "consequence_bank": spec["consequence_bank"],
                "carrier": spec["carrier"],
                "target_endpoint": scan.get("endpoint"),
                "target_symbol": scan.get("operation"),
                "target_phase": scan.get("phase"),
                "source_screen": spec["scan"],
                "implementation_pattern_id": scan.get("implementation_pattern_id"),
                "screen_exact_implementation_id": scan.get("representative_exact_implementation_id"),
                "screen_amplification": scan.get("amplification"),
                "screen_p_value": scan.get("p_value"),
                "screen_rms_mean": scan.get("rms_mean"),
            }
            if not matches:
                blocked.append({
                    **base,
                    "status": "UNRESOLVED_NO_MATCHING_GENERATED_PROGRAM",
                    "reason": "The frozen campaign has no same-phase, same-symbol-family endpoint; no target was fabricated.",
                })
                continue
            # Use the historical region as an optional hint.  The runner also
            # checks symbol family after fresh compilation, so an ordinal
            # change cannot silently target an unrelated fused program.
            target = matches[0]
            rows.append({
                **base,
                "case_id": f"{spec['prefix']}_{len(rows):04d}",
                # A phase-only hint intentionally prevents an ordinal from a
                # prior compile from being treated as an exact region.  The
                # runner then resolves the symbol family and endpoint in the
                # fresh campaign.
                "target_region": str(target.get("phase", scan.get("phase"))),
                "status": "PENDING_PARAMETER_REACHABLE_REPLAY",
                "binding_rule": "same phase + generated symbol family + captured endpoint; fresh compile rechecks family",
            })
    payload = {
        "schema": "kernel-analyzer-expanded-gemma-llama-target-manifest-v1",
        "status": "FROZEN_NONZERO_SCAN_REPLAY_SET",
        "selection_rule": "all frozen residual-nonzero scan rows from Gemma/Llama; no row is called negative before legal replay",
        "rows": rows,
        "blocked_rows": blocked,
        "counts": {"replay_targets": len(rows), "blocked_no_matching_program": len(blocked)},
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    blocked_path = output.with_name("operator_scan_replay_blocked.json")
    blocked_path.write_text(json.dumps({
        "schema": "kernel-analyzer-expanded-gemma-llama-replay-blocked-v1",
        "claim_boundary": "Blocked rows are unresolved, never negative.",
        "rows": blocked,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
