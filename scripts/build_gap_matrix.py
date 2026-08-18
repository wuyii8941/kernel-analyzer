#!/usr/bin/env python3
"""Build the exhaustive, fail-closed invocation gap/closure matrix.

This file is deliberately a plan over the complete invocation denominator, not
an operator-family priority list.  Every invocation receives an explicit
closure action for every gate which is not already proved.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
ARCHITECTURES = ("qwen", "mamba", "moe", "phi", "deepseek8")
MODEL_KEYS = {
    "qwen": "qwen3_1p7b",
    "mamba": "mamba_130m",
    "moe": "granite_3p1_1b_a400m",
    "phi": "phi4_mini_3p8b",
    "deepseek8": "deepseek_r1_0528_qwen3_8b",
}
LEDGER_FILES = {
    "qwen": "qwen_invocation_ledger.json.gz",
    "mamba": "mamba_invocation_ledger.json.gz",
    "moe": "moe_invocation_ledger.json.gz",
    "phi": "phi4_seq64_invocation_ledger.json.gz",
    "deepseek8": "deepseek8b_seq64_invocation_ledger.json.gz",
}


def read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def qwen_fb_action(status: str) -> str:
    if "REFERENCE_ONLY" in status:
        return "qwen_prove_reference_elision_or_bind_enclosing_closed_cut"
    return "qwen_capture_runtime_tensor_edges_seq_nr_and_forward_origin"


def architecture_scope(architecture: str, overload: str, qwen_overloads: set[str]) -> str:
    if architecture == "qwen":
        return "QWEN_BASE"
    if overload in qwen_overloads:
        return "SHARED_WITH_QWEN"
    return "ARCHITECTURE_DELTA"


def row_gaps(architecture: str, row: dict[str, Any]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    fb = row["eager_aot_binding"]["status"]
    candidate = row["candidate_region_binding"]["status"]
    numerical = row["numerical_measurement"]["status"]
    verdict = row["bias_verdict"]["status"]

    if architecture == "qwen":
        if not row["mathematical_fb"]["exact_fb_origin_status"].startswith("COMPLETE_"):
            gaps.append({"gate": "EAGER_AOT_FB_IDENTITY", "action": qwen_fb_action(fb)})
        if not candidate.startswith("EXACT_"):
            action = (
                "qwen_close_candidate_dataflow_cut_after_fb_identity"
                if "BECAUSE_EAGER_AOT" in candidate
                else "qwen_propagate_aot_proof_ids_through_fx_lowering_to_generated_region"
            )
            gaps.append({"gate": "CANDIDATE_REGION_BINDING", "action": action})
    else:
        if fb != "COMPLETE_EXACT_AOT_FB_IDENTITY":
            gaps.append({
                "gate": "EAGER_AOT_FB_IDENTITY",
                "action": f"{architecture}_capture_weak_strong_aot_with_saved_tensor_edges",
            })
        if not candidate.startswith("EXACT_"):
            gaps.append({
                "gate": "CANDIDATE_REGION_BINDING",
                "action": f"{architecture}_bind_aot_proof_ids_to_both_candidate_configurations",
            })

    if not (
        numerical.startswith("MEASURED_")
        or numerical.startswith("NOT_APPLICABLE_EXACT_")
    ):
        gaps.append({
            "gate": "NUMERICAL_MEASUREMENT",
            "action": f"{architecture}_run_48_calibration_96_heldout_candidate_blind_measurement",
        })
    if not row["bias_verdict"].get("candidate_correctness_certified", False):
        gaps.append({
            "gate": "CORRECTNESS_VERDICT",
            "action": f"{architecture}_issue_bias_tail_nonfinite_verdict_from_frozen_margins",
        })
    if verdict == "EQUIVALENT" and gaps:
        raise RuntimeError(f"contradictory verdict for {row['row_id']}")
    return gaps


def main() -> None:
    bias_protocol = json.loads(
        (COVERAGE / "directional_bias_protocol.json").read_text()
    )
    model_scope = json.loads((COVERAGE / "model_scope.json").read_text())
    ledgers = {
        name: read_gzip(COVERAGE / LEDGER_FILES[name])
        for name in ARCHITECTURES
    }
    qwen_overloads = {
        row["invocation"]["overload"] for row in ledgers["qwen"]["rows"]
    }
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for architecture in ARCHITECTURES:
        for source in ledgers[architecture]["rows"]:
            invocation = source["invocation"]
            gaps = row_gaps(architecture, source)
            row = {
                "row_id": source["row_id"],
                "architecture": architecture,
                "model": MODEL_KEYS[architecture],
                "scope": model_scope["models"][MODEL_KEYS[architecture]]["scope"],
                "phase": invocation["phase"],
                "overload": invocation["overload"],
                "architecture_scope": architecture_scope(
                    architecture, invocation["overload"], qwen_overloads
                ),
                "source_row_sha256": source["row_sha256"],
                "missing_gates": gaps,
                "complete": not gaps,
            }
            verdict_status = source["bias_verdict"]["status"]
            if verdict_status == "EQUIVALENT_EXACT_SEMANTIC_ELISION":
                t1_status = "NOT_APPLICABLE_EQUIVALENT"
            elif source["numerical_measurement"]["status"].startswith("MEASURED_"):
                t1_status = "PENDING_PROTOCOL_V2_LOCAL_ADJUDICATION"
            else:
                t1_status = "BLOCKED_BY_NUMERICAL_MEASUREMENT"
            row["directional_bias_pipeline"] = {
                "protocol_sha256": bias_protocol["protocol_sha256"],
                "T1_LOCAL": {"status": t1_status},
                "T2_CAUSAL": {
                    "status": "BLOCKED_UNTIL_T1_POSITIVE",
                    "required_declarations": [
                        "intervention_type", "exact_semantic_endpoint",
                        "all_directly_reachable_parameter_carriers",
                        "matched_negative_control",
                    ],
                },
                "T3_RAW": {
                    "status": "BLOCKED_UNTIL_T2_POSITIVE",
                    "required_declarations": [
                        "independent_confirmation_state_ids",
                        "all_carrier_coordinates_or_exact_partition",
                        "multiplicity_correction_family",
                    ],
                },
                "T3_RELATIVE": {
                    "status": "BLOCKED_UNTIL_T2_POSITIVE",
                    "required_declarations": [
                        "reference_gradient_endpoint", "complete_carrier_coordinates",
                        "independent_confirmation_state_ids", "multiplicity_correction_family",
                    ],
                },
                "T3_FACTOR": {
                    "status": "BLOCKED_UNTIL_T2_POSITIVE",
                    "required_declarations": [
                        "analytic_error_coefficient", "reference_only_carrier",
                        "independent_confirmation_state_ids", "multiplicity_correction_family",
                    ],
                },
                "T4_ACCUMULATION": {"status": "BLOCKED_UNTIL_T3_POSITIVE"},
            }
            row["row_sha256"] = digest(row)
            rows.append(row)
            grouped[(
                architecture,
                invocation["phase"],
                invocation["overload"],
                tuple(gap["action"] for gap in gaps),
            )].append(source["row_id"])

    expected = sum(len(ledger["rows"]) for ledger in ledgers.values())
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise RuntimeError("gap matrix does not preserve the exact invocation denominator")

    groups = []
    for (architecture, phase, overload, actions), ids in sorted(grouped.items()):
        groups.append({
            "architecture": architecture,
            "phase": phase,
            "overload": overload,
            "invocations": len(ids),
            "invocation_ids_sha256": digest(sorted(ids)),
            "actions": list(actions),
        })

    architecture_summary = {}
    for architecture in ARCHITECTURES:
        subset = [row for row in rows if row["architecture"] == architecture]
        scope_counts = Counter(row["architecture_scope"] for row in subset)
        gate_counts = Counter(
            gap["gate"] for row in subset for gap in row["missing_gates"]
        )
        architecture_summary[architecture] = {
            "invocations": len(subset),
            "unique_overloads": len({row["overload"] for row in subset}),
            "architecture_scope_counts": dict(sorted(scope_counts.items())),
            "missing_gate_invocation_counts": dict(sorted(gate_counts.items())),
            "fully_complete_invocations": sum(row["complete"] for row in subset),
        }

    active_rows = [row for row in rows if row["scope"] == "FULL_STEP"]
    payload = {
        "schema": "kernel-analyzer-exhaustive-gap-matrix-v3",
        "status": "PARTIAL_FAIL_CLOSED",
        "unit": "one actual dispatcher invocation in one complete loss forward/backward step",
        "denominator": expected,
        "active_full_step_denominator": len(active_rows),
        "retained_paused_denominator": expected - len(active_rows),
        "architecture_summary": architecture_summary,
        "candidate_configurations": {
            "qwen": ["bf16_eager", "bf16_inductor_full_step"],
            "mamba": ["official_fused", "compiled_explicit_recurrence"],
            "moe": ["PAUSED_OUT_OF_SCOPE_RETAINED_EVIDENCE"],
            "phi": ["bf16_eager", "bf16_inductor_full_step"],
            "deepseek8": ["bf16_eager", "bf16_inductor_sharded_full_step"],
        },
        "state_protocol": {
            "reference_calibration_states": 48,
            "heldout_candidate_states": 96,
            "candidate_blind_margin_freeze": True,
        },
        "directional_bias_protocol_sha256": bias_protocol["protocol_sha256"],
        "model_scope_sha256": model_scope["scope_sha256"],
        "closure_order": [
            "EAGER_AOT_FB_IDENTITY",
            "CANDIDATE_REGION_BINDING",
            "NUMERICAL_MEASUREMENT",
            "CORRECTNESS_VERDICT",
        ],
        "groups": groups,
        "rows": rows,
        "claim_boundary": (
            "This matrix enumerates every current invocation and its required closure work. "
            "It is not a completion claim and it does not collapse invocations into families."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "gap_matrix.json.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "denominator": expected,
        "groups": len(groups),
        "sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
