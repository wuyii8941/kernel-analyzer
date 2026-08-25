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


# Exact external-call identities do not retain module paths.  These labels are
# therefore tied to the deterministic first callsite selected by the replay,
# and were verified against the generated source kept in the runtime release.
# Keeping the mapping here prevents the evidence table from falling back to a
# vague "extern_kernels.mm" label when the actual training location is known.
EXTERNAL_FIRST_CALLSITE_LOCATIONS = {
    "5477521afded413c06fbad69c94b7011e380f2449a116424e9b42a2e586de5a2": (
        "layer-0 attention q_proj forward GEMM"
    ),
}


def family(symbol: str | None) -> str:
    return re.sub(r"_\d+$", "", symbol or "")


def carrier_for(spec: dict, scan: dict) -> str:
    """Choose a parameter that keeps the screened backward reachable.

    Gemma's original screen was compiled with the complete backward graph.
    The replay deliberately trains one carrier at a time.  Using the vision
    projection for every language-model backward silently removes attention
    and normalization parameter-gradient programs from the specialized graph.
    Bind those two generated families to a declared layer-0 parameter on the
    same mathematical path; the runner still requires the exact phase and
    generated symbol family before applying a repair.
    """
    if spec.get("architecture") != "gemma4":
        return spec["carrier"]
    symbol = family(scan.get("operation"))
    endpoint = scan.get("endpoint")
    if "arange_bmm_cat_cos" in symbol and scan.get("phase") == "BACKWARD":
        return (
            "model.language_model.layers.0.self_attn.k_proj.weight"
            if endpoint == "out_ptr2"
            else "model.language_model.layers.0.self_attn.q_proj.weight"
        )
    if "pow_sum_view" in symbol and scan.get("phase") == "BACKWARD":
        return "model.language_model.layers.0.input_layernorm.weight"
    return spec["carrier"]


def external_carrier_for(spec: dict, exact_payload: dict) -> str:
    """Choose a declared carrier that remains reachable from an external call.

    These names are hypotheses checked by the replay's nonzero-effect gate.
    A zero result stays unresolved and is never converted into a negative.
    """
    if not spec["model"].startswith("Llama"):
        return spec["carrier"]
    phase = exact_payload.get("phase")
    operation = exact_payload.get("operation")
    contracts = exact_payload.get("operand_contracts", {})
    output = contracts.get("kw:out") or contracts.get("output") or {}
    shape = list(output.get("shape", []))
    if phase != "BACKWARD":
        return "model.norm.weight"
    if operation == "extern_kernels.addmm":
        return "model.embed_tokens.weight"
    if operation == "extern_kernels.bmm":
        return "model.layers.0.input_layernorm.weight"
    if operation == "extern_kernels.mm" and len(shape) == 2:
        if shape[0] in {128, 512} and shape[1] == 3072:
            return "model.layers.0.input_layernorm.weight"
        if sorted(shape) == [3072, 8192]:
            return "model.layers.27.mlp.down_proj.weight"
    return "model.layers.0.input_layernorm.weight"


SPECS = [
    {
        "model": "Llama-3.2-3B",
        "architecture": "generic",
        "model_path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "scan": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/pattern_screen.json",
        "campaign": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/campaign.json.gz",
        "implementation_census": "results/property/tcmp_allop_v1/heldout/llama32_3b_text128/implementation_census.json",
        "input_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128.json",
        "consequence_bank": "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128_trajectory4096.json",
        # Keep the default carrier small enough for a fresh compiled replay;
        # the queue can rebind to the tied embedding only when this carrier
        # has zero energy.
        "carrier": "model.norm.weight",
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
        "implementation_census": "results/property/declared_persistent_4096/llama32_3b_text512_runtime_release/implementation_census.json",
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
    existing_ids = {}
    if output.exists():
        existing = json.loads(output.read_text())
        for row in existing.get("rows", []):
            key = (
                row.get("model"), row.get("implementation_pattern_id"),
                row.get("target_endpoint"), row.get("target_phase"),
            )
            existing_ids[key] = row.get("case_id")

    def case_id_for(spec: dict, scan: dict, kind: str) -> str:
        key = (
            spec["model"], scan.get("implementation_pattern_id"),
            scan.get("endpoint"), scan.get("phase"),
        )
        if existing_ids.get(key):
            return existing_ids[key]
        digest = str(scan.get("representative_exact_implementation_id") or "unresolved")[:12]
        return f"{spec['prefix']}_{kind.lower()}_{digest}"

    rows: list[dict] = []
    blocked: list[dict] = []
    for spec in SPECS:
        scan_rows = load_scan(spec)
        with gzip.open(ROOT / spec["campaign"], "rt", encoding="utf-8") as handle:
            campaign = json.load(handle)["rows"]
        census_by_exact = {}
        if spec.get("implementation_census"):
            census = json.loads((ROOT / spec["implementation_census"]).read_text())
            census_by_exact = {
                row["exact_implementation_id"]: row
                for row in census.get("implementations", [])
            }
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
                "carrier": carrier_for(spec, scan),
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
            exact_row = census_by_exact.get(scan.get("representative_exact_implementation_id"))
            exact_payload = (exact_row or {}).get("identity", {}).get("exact_payload", {})
            if str(scan.get("operation", "")).startswith("extern_kernels."):
                if not exact_payload or exact_payload.get("implementation_kind") != "EXTERNAL_OR_LIBRARY":
                    blocked.append({
                        **base,
                        "status": "UNRESOLVED_EXTERNAL_EXACT_CONTRACT_MISSING",
                        "reason": "The frozen exact external-call ABI is absent; no runtime target was fabricated.",
                    })
                    continue
                rows.append({
                    **base,
                    "case_id": case_id_for(spec, scan, "extern"),
                    "target_kind": "EXTERN",
                    "external_census": spec["implementation_census"],
                    "external_exact_id": scan.get("representative_exact_implementation_id"),
                    "semantic_location": EXTERNAL_FIRST_CALLSITE_LOCATIONS.get(
                        scan.get("representative_exact_implementation_id")
                    ),
                    "carrier": external_carrier_for(spec, exact_payload),
                    "retain_full_backward": scan.get("phase") == "BACKWARD",
                    "status": "PENDING_PARAMETER_REACHABLE_REPLAY",
                    "binding_rule": "frozen exact external-call ABI + fresh runtime phase/function/operand contract; deterministic first callsite representative",
                })
                continue
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
                "case_id": case_id_for(spec, scan, "triton"),
                "target_kind": "TRITON",
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
