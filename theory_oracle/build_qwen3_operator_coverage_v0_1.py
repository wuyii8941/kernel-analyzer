#!/usr/bin/env python
"""Build a denominator-explicit Qwen3 operator coverage ledger."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


CATEGORY = {
    "mm": "linear_algebra",
    "bmm": "attention_linear_algebra",
    "mean": "reduction_norm",
    "sum": "reduction",
    "amax": "reduction",
    "rsqrt": "norm_pointwise",
    "pow": "norm_or_backward_pointwise",
    "embedding": "embedding",
    "exp": "softmax_or_activation",
    "div": "softmax_or_activation",
    "sigmoid": "activation_backward",
    "fma": "pointwise_backward",
    "add": "pointwise_or_residual",
    "sub": "pointwise",
    "mul": "pointwise_or_gating",
    "cos": "rotary",
    "sin": "rotary",
    "view": "shape_layout",
    "permute": "shape_layout",
    "transpose": "shape_layout",
    "expand": "shape_layout",
    "slice": "indexing_layout",
    "slice_scatter": "indexing_backward",
    "clone": "layout_materialization",
    "convert_element_type": "cast_precision",
    "logical_not": "mask_logic",
    "where": "mask_logic",
    "eq": "mask_logic",
    "any": "mask_reduction",
    "bitwise_and": "mask_logic",
    "index": "indexing",
    "iota": "index_generation",
    "full": "tensor_construction",
    "cat": "shape_layout",
    "unsqueeze": "shape_layout",
    "squeeze": "shape_layout",
    "neg": "pointwise",
    "le": "comparison",
    "ge": "comparison",
    "lt": "comparison",
    "ne": "comparison",
    "scalar_tensor": "tensor_construction",
    "unsafe_masked_index_put_accumulate": "gradient_accumulation",
}


def target_base(target: str) -> str:
    parts = target.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else target


def short_name(target: str) -> str:
    return target.split(".")[1] if "." in target else target


def parse_targets(path: Path) -> collections.Counter[str]:
    text = path.read_text()
    return collections.Counter(
        re.findall(r"torch\.ops\.((?:aten|prims)\.[A-Za-z0-9_\.]+)\(", text)
    )


def row_for_target(
    domain: str,
    target: str,
    count: int,
    memberships: dict[str, list[str]],
) -> dict[str, Any]:
    base = target_base(target)
    name = short_name(target)
    kernels = memberships.get(base, [])
    evidence = "MAPPED_NOT_INTERVENED" if kernels else "OBSERVED_ONLY"
    partial = None
    if domain == "model_forward" and target == "aten.mm.default":
        partial = "one of 197 Linear invocations (final lm_head) has VALID_NULL_EFFECT"
    elif domain == "model_forward" and name in {
        "pow", "mean", "rsqrt", "mul", "add", "convert_element_type"
    }:
        partial = "one composite final-RMSNorm invocation has VALID_NULL_EFFECT; the target type is not fully covered"
    return {
        "domain": domain,
        "target": target,
        "count": count,
        "category": CATEGORY.get(name, "other"),
        "kernel_membership": kernels,
        "coverage_state": evidence,
        "partial_evidence": partial,
        "fully_causally_covered": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    inventory = Path(args.inventory_dir).resolve()
    result = json.loads((inventory / "result.json").read_text())
    summary = json.loads((inventory / "kernel_summary.json").read_text())
    if result["status"] != "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY":
        raise ValueError("invalid source inventory")
    if summary["status"] != "VALID_DESCRIPTIVE_SUMMARY":
        raise ValueError("invalid source kernel summary")

    memberships: dict[str, list[str]] = collections.defaultdict(list)
    for family in summary["kernel_families"]:
        for target in family["original_aten"]:
            memberships[target].append(family["name"])
    memberships["aten.mm"].append("extern:mm")
    memberships["aten.bmm"].append("extern:bmm")
    for key in memberships:
        memberships[key] = sorted(set(memberships[key]))

    trace_root = inventory / "inductor_trace/torchinductor"
    forward = parse_targets(trace_root / "model__1_forward_4.1/fx_graph_readable.py")
    backward = parse_targets(trace_root / "model__1_backward_5.2/fx_graph_readable.py")
    atomic_rows = [
        row_for_target("model_forward", target, count, memberships)
        for target, count in sorted(forward.items())
    ] + [
        row_for_target("model_backward", target, count, memberships)
        for target, count in sorted(backward.items())
    ]

    semantic_families = [
        {"family": "Linear", "forward_invocations": 197, "valid_invocations": 1, "state": "PARTIAL", "missing": "196 q/k/v/o/gate/up/down projections"},
        {"family": "RMSNorm", "forward_invocations": 113, "valid_invocations": 1, "state": "PARTIAL", "missing": "112 input/post-attention/q/k norm invocations"},
        {"family": "attention_bmm", "forward_invocations": 56, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "attention_softmax", "forward_invocations": 28, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "rotary_embedding", "forward_invocations": 28, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "MLP_SiLU", "forward_invocations": 28, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "MLP_gate_multiply", "forward_invocations": 28, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "residual_add", "forward_invocations": 56, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "token_embedding", "forward_invocations": 1, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
        {"family": "causal_mask_construction", "forward_invocations": 1, "valid_invocations": 0, "state": "MAPPED_NOT_INTERVENED"},
    ]
    declared_forward_invocations = sum(row["forward_invocations"] for row in semantic_families)
    valid_forward_invocations = sum(row["valid_invocations"] for row in semantic_families)

    non_model_domains = [
        {"domain": "scorer_postprocess", "units": ["gather", "logsumexp", "stack", "subtract", "squeeze"], "descriptive": "SOURCE_INSPECTED", "causal": "UNINSTANTIATED_PROPAGATION"},
        {"domain": "grpo_loss_and_event", "units": ["exp_ratio", "clamp", "minimum", "mask", "reductions", "clip_predicate"], "descriptive": "SOURCE_INSPECTED", "causal": "PARTIAL_BRANCH_REPAIR_ONLY"},
        {"domain": "model_backward", "units": sorted(backward), "descriptive": "GRAPH_INVENTORIED", "causal": "UNINSTANTIATED"},
        {"domain": "gradient_control", "units": ["finite_check", "unscale", "global_norm", "clip_coefficient", "gradient_scale"], "descriptive": "PATH_EXECUTED", "causal": "UNINSTANTIATED"},
        {"domain": "optimizer_amp_scheduler", "units": ["fused_AdamW", "GradScaler_update", "AMP_skip", "scheduler_step"], "descriptive": "PATH_EXECUTED", "causal": "UNINSTANTIATED"},
    ]

    payload = {
        "schema_version": "forkcert.qwen3-operator-coverage.v0.1",
        "scope": "Qwen3-0.6B GRPO frozen step-29 training transition",
        "source_inventory_status": result["status"],
        "coverage_verdicts": {
            "frozen_forward_descriptive": "COMPLETE",
            "frozen_forward_operator_causal": "INCOMPLETE",
            "frozen_training_step_causal": "INCOMPLETE",
            "population_operator_causal": "UNINSTANTIATED",
            "discrepancy_oracle_definition": "NOT_INVALIDATED_BY_COVERAGE_GAP",
        },
        "metrics": {
            "forward_atomic_target_types_observed": len(forward),
            "forward_atomic_calls_observed": sum(forward.values()),
            "backward_atomic_target_types_observed": len(backward),
            "backward_atomic_calls_observed": sum(backward.values()),
            "declared_high_level_forward_invocations": declared_forward_invocations,
            "validly_intervened_high_level_forward_invocations": valid_forward_invocations,
            "valid_intervention_fraction": valid_forward_invocations / declared_forward_invocations,
            "fully_covered_high_level_families": sum(row["state"] not in {"PARTIAL", "MAPPED_NOT_INTERVENED"} for row in semantic_families),
            "declared_high_level_families": len(semantic_families),
        },
        "semantic_forward_families": semantic_families,
        "atomic_graph_ledger": atomic_rows,
        "non_model_domains": non_model_domains,
        "invalid_evidence": {
            "decoder_layer_27_split": "INVALID_TREATMENT; candidate endpoint changed and is excluded from coverage credit"
        },
        "completion_rule": "each invocation or justified equivalence class must advance beyond observation/mapping with treatment-integrity and state-distribution evidence",
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3 operator coverage ledger v0.1",
        "",
        "## Current verdict",
        "",
        "| Claim | Verdict |",
        "|---|---|",
    ]
    for key, value in payload["coverage_verdicts"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "The frozen forward realization is descriptively inventoried, but no high-level operator family is fully causally covered. Only final `lm_head` and final RMSNorm have valid intervention evidence.",
        "",
        "## High-level forward denominator",
        "",
        "| Family | Invocations | Valid interventions | State | Missing |",
        "|---|---:|---:|---|---|",
    ]
    for row in semantic_families:
        lines.append(
            f"| {row['family']} | {row['forward_invocations']} | {row['valid_invocations']} | {row['state']} | {row.get('missing', '')} |"
        )
    lines += [
        "",
        f"Valid invocation coverage: {valid_forward_invocations}/{declared_forward_invocations} ({100 * valid_forward_invocations / declared_forward_invocations:.3f}%).",
        "",
        "## Training domains not causally covered",
        "",
        "| Domain | Descriptive evidence | Causal state |",
        "|---|---|---|",
    ]
    for row in non_model_domains:
        lines.append(f"| {row['domain']} | {row['descriptive']} | {row['causal']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Oracle measurement validity and operator-analysis completeness are separate. The current coverage gap does not undo the B/H/N/U or selected-state Oracle definition, but it prohibits a complete operator-root-cause claim.",
        "",
        "Raw ATen/prims type and invocation details are in the paired JSON ledger.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
