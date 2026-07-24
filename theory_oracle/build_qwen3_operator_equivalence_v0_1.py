#!/usr/bin/env python
"""Build candidate operator-equivalence and intervention denominators for Qwen3.

The output deliberately grants no equivalence credit.  It turns the coarse
family denominator into semantic-role and generated-treatment denominators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYERS = list(range(28))


def repeated(
    class_id: str,
    family: str,
    semantic_role: str,
    *,
    layers: list[int] = LAYERS,
    fusion_context: str,
    existing_valid_layers: list[int] | None = None,
) -> dict[str, Any]:
    existing_valid_layers = existing_valid_layers or []
    candidates = [layers[0], layers[len(layers) // 2], layers[-1]]
    return {
        "class_id": class_id,
        "family": family,
        "semantic_role": semantic_role,
        "invocations": len(layers),
        "layers": layers,
        "fusion_context": fusion_context,
        "minimum_transport_representatives": candidates,
        "existing_valid_layers": existing_valid_layers,
        "valid_invocations": len(existing_valid_layers),
        "equivalence_status": "PROPOSED_UNVALIDATED",
        "coverage_state": "MAPPED_NOT_INTERVENED",
    }


def singleton(
    class_id: str,
    family: str,
    semantic_role: str,
    *,
    fusion_context: str,
    valid: bool = False,
) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "family": family,
        "semantic_role": semantic_role,
        "invocations": 1,
        "layers": None,
        "fusion_context": fusion_context,
        "minimum_transport_representatives": ["singleton"],
        "existing_valid_layers": ["singleton"] if valid else [],
        "valid_invocations": 1 if valid else 0,
        "equivalence_status": "NOT_APPLICABLE_SINGLETON",
        "coverage_state": "VALID_NULL_EFFECT" if valid else "MAPPED_NOT_INTERVENED",
    }


def build_semantic_classes() -> list[dict[str, Any]]:
    mm = "extern:mm plus role-specific cast/layout consumers"
    rows = [
        repeated("linear.q_proj", "Linear", "attention query projection", fusion_context=mm),
        repeated("linear.k_proj", "Linear", "attention key projection", fusion_context=mm),
        repeated("linear.v_proj", "Linear", "attention value projection", fusion_context=mm),
        repeated("linear.o_proj", "Linear", "attention output projection", fusion_context=mm),
        repeated("linear.gate_proj", "Linear", "MLP gate projection", fusion_context=mm),
        repeated("linear.up_proj", "Linear", "MLP up projection", fusion_context=mm),
        repeated("linear.down_proj", "Linear", "MLP down projection", fusion_context=mm),
        singleton("linear.lm_head", "Linear", "vocabulary projection", fusion_context="extern:mm plus logits conversion/slice", valid=True),
        singleton("norm.input.layer0", "RMSNorm", "layer-0 input normalization", fusion_context="embedding fused with input RMSNorm"),
        repeated(
            "norm.input.layers1_27",
            "RMSNorm",
            "decoder input normalization after previous-layer residual",
            layers=list(range(1, 28)),
            fusion_context="cross-layer previous residual/add fused with next input RMSNorm",
        ),
        repeated("norm.post_attention", "RMSNorm", "post-attention normalization", fusion_context="attention residual fused with RMSNorm and downstream cast"),
        repeated("norm.q_norm", "RMSNorm", "query normalization", fusion_context="q projection, RMSNorm and rotary application fused"),
        repeated("norm.k_norm", "RMSNorm", "key normalization", fusion_context="k projection, RMSNorm, rotary and layout fused"),
        singleton("norm.final", "RMSNorm", "final hidden-state normalization", fusion_context="last residual plus final RMSNorm", valid=True),
        repeated("attention.qk_bmm", "attention_bmm", "query-key score product", fusion_context="extern:bmm with fused q/k producers and softmax consumer"),
        repeated("attention.pv_bmm", "attention_bmm", "probability-value product", fusion_context="extern:bmm with softmax/value producers and layout consumers"),
        repeated("attention.softmax", "attention_softmax", "masked safe softmax", fusion_context="mask add and online safe-softmax reduction fused"),
        repeated("attention.rotary", "rotary_embedding", "paired query/key rotary application", fusion_context="q/k RMSNorm, trig, slicing and pointwise application fused"),
        repeated("mlp.silu", "MLP_SiLU", "gate activation", fusion_context="SiLU fused with gate multiplication"),
        repeated("mlp.gate_multiply", "MLP_gate_multiply", "activated-gate times up projection", fusion_context="SiLU fused with gate multiplication"),
        repeated("residual.attention", "residual_add", "attention residual connection", fusion_context="residual add fused with post-attention RMSNorm"),
        repeated("residual.mlp", "residual_add", "MLP residual connection", fusion_context="residual add fused across decoder-layer boundary"),
        singleton("embedding.token", "token_embedding", "token lookup", fusion_context="embedding fused with layer-0 input RMSNorm"),
        singleton("mask.causal", "causal_mask_construction", "causal/padding mask construction", fusion_context="index/comparison/where construction plus per-layer mask consumption"),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--kernel-summary", required=True)
    parser.add_argument("--barrier-result")
    parser.add_argument("--named-batch-result")
    parser.add_argument("--functional-batch-result")
    parser.add_argument("--sdpa-decomposition-result")
    parser.add_argument("--causal-mask-result")
    parser.add_argument("--candidate-kernel15-result")
    parser.add_argument("--candidate-kernel-result", action="append", default=[])
    parser.add_argument("--candidate-kernel-campaign-result")
    parser.add_argument("--candidate-kernel-campaign-audit")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    coverage = json.loads(Path(args.coverage_json).read_text())
    kernels = json.loads(Path(args.kernel_summary).read_text())
    semantic = build_semantic_classes()
    barrier = None
    if args.barrier_result:
        barrier = json.loads(Path(args.barrier_result).read_text())
        if barrier["status"] != "VALID_BARRIER_CONDITIONED_OPERATOR_EFFECT":
            raise ValueError("barrier result is not valid barrier-conditioned evidence")
        target = next(row for row in semantic if row["class_id"] == "norm.input.layers1_27")
        target["barrier_conditioned_layers"] = [27]
        target["coverage_state"] = "BARRIER_CONDITIONED"
    if args.named_batch_result:
        batch = json.loads(Path(args.named_batch_result).read_text())
        if batch["status"] != "VALID_BARRIER_CONDITIONED_BATCH":
            raise ValueError("named batch is not valid barrier-conditioned evidence")
        class_index = {row["class_id"]: row for row in semantic}
        for target_id in batch["targets"]:
            parts = target_id.split(".")
            if target_id == "embedding.token":
                class_id, layer = "embedding.token", "singleton"
            elif parts[0] == "linear":
                class_id, layer = ".".join(parts[:2]), int(parts[2].removeprefix("layer"))
            elif target_id == "norm.input.layer0":
                class_id, layer = "norm.input.layer0", 0
            elif target_id.startswith("norm.input.layer"):
                class_id, layer = "norm.input.layers1_27", int(parts[2].removeprefix("layer"))
            else:
                class_id, layer = ".".join(parts[:2]), int(parts[2].removeprefix("layer"))
            row = class_index[class_id]
            observed = set(row.get("barrier_conditioned_layers", []))
            observed.add(layer)
            row["barrier_conditioned_layers"] = sorted(observed, key=str)
            row["coverage_state"] = "BARRIER_CONDITIONED"
    composite_sdpa_invocations = 0
    sdpa_decomposition_status = "UNINSTANTIATED"
    if args.functional_batch_result:
        batch = json.loads(Path(args.functional_batch_result).read_text())
        if batch["status"] != "VALID_BARRIER_CONDITIONED_BATCH":
            raise ValueError("functional batch is not valid barrier-conditioned evidence")
        class_index = {row["class_id"]: row for row in semantic}
        direct_prefixes = {
            "attention.rotary": "attention.rotary",
            "mlp.silu": "mlp.silu",
            "mlp.gate_multiply": "mlp.gate_multiply",
            "residual.attention": "residual.attention",
            "residual.mlp": "residual.mlp",
        }
        for target_id in batch["targets"]:
            prefix, layer_text = target_id.rsplit(".layer", 1)
            layer = int(layer_text)
            if prefix == "attention.sdpa":
                composite_sdpa_invocations += 1
                for class_id in ("attention.qk_bmm", "attention.softmax", "attention.pv_bmm"):
                    row = class_index[class_id]
                    observed = set(row.get("composite_sdpa_evidence_layers", []))
                    observed.add(layer)
                    row["composite_sdpa_evidence_layers"] = sorted(observed)
                continue
            row = class_index[direct_prefixes[prefix]]
            observed = set(row.get("barrier_conditioned_layers", []))
            observed.add(layer)
            row["barrier_conditioned_layers"] = sorted(observed)
            row["coverage_state"] = "BARRIER_CONDITIONED"
    if args.sdpa_decomposition_result:
        decomposition = json.loads(Path(args.sdpa_decomposition_result).read_text())
        sdpa_decomposition_status = decomposition["status"]
        if sdpa_decomposition_status != "VALID_INVALIDATION_REFERENCE_RECONSTRUCTION_CHANGED":
            raise ValueError("unexpected SDPA decomposition result status")
        for class_id in ("attention.qk_bmm", "attention.softmax", "attention.pv_bmm"):
            row = next(row for row in semantic if row["class_id"] == class_id)
            row["coverage_state"] = "INVALID_TREATMENT"
    causal_mask_status = "UNINSTANTIATED"
    if args.causal_mask_result:
        mask_result = json.loads(Path(args.causal_mask_result).read_text())
        causal_mask_status = mask_result["status"]
        if causal_mask_status != "INVALID_TREATMENT":
            raise ValueError("unexpected causal-mask result status")
        mask_row = next(row for row in semantic if row["class_id"] == "mask.causal")
        mask_row["coverage_state"] = "INVALID_TREATMENT"
    candidate_kernel_evidence: dict[str, dict[str, Any]] = {}

    def add_candidate_kernel_result(kernel_result: dict[str, Any]) -> None:
        family = kernel_result["kernel_family"]
        rows = kernel_result["repairs"]
        evidence = {
            "mode": "shared_path_reexecution" if family.startswith("extern:") else "repair",
            "repairs": len(kernel_result["selected_call_indices"]),
            "nonzero": 0,
            "null": 0,
            "closer": 0,
            "farther": 0,
            "direction_unmeasured": 0,
        }
        for selected in kernel_result["selected_call_indices"]:
            row = rows[str(selected)]
            effect = row.get("candidate_to_repair", row.get("effect", {})).get("l2")
            if effect is None:
                raise ValueError(f"missing repair effect for {family} call {selected}")
            evidence["null" if effect == 0.0 else "nonzero"] += 1
            direction = row.get("direction")
            if not direction:
                evidence["direction_unmeasured"] += 1
            elif direction["l2_distance_change"] < 0.0:
                evidence["closer"] += 1
            elif direction["l2_distance_change"] > 0.0:
                evidence["farther"] += 1
        previous = candidate_kernel_evidence.get(family)
        if previous is not None and previous != evidence:
            raise ValueError(f"conflicting repair evidence for {family}")
        candidate_kernel_evidence[family] = evidence

    if args.candidate_kernel15_result:
        kernel_result = json.loads(Path(args.candidate_kernel15_result).read_text())
        if kernel_result["status"] != "VALID_ORIGINAL_CANDIDATE_KERNEL_REPAIR":
            raise ValueError("candidate kernel-15 result is not valid")
        add_candidate_kernel_result(kernel_result)
    for result_path in args.candidate_kernel_result:
        kernel_result = json.loads(Path(result_path).read_text())
        if not kernel_result.get("status", "").startswith("VALID_ORIGINAL_CANDIDATE_"):
            raise ValueError(f"candidate kernel result is not valid: {result_path}")
        add_candidate_kernel_result(kernel_result)
    if bool(args.candidate_kernel_campaign_result) != bool(args.candidate_kernel_campaign_audit):
        raise ValueError("campaign result and audit must be supplied together")
    if args.candidate_kernel_campaign_result:
        campaign = json.loads(Path(args.candidate_kernel_campaign_result).read_text())
        campaign_audit = json.loads(Path(args.candidate_kernel_campaign_audit).read_text())
        if campaign_audit["status"] != "VALID_FAIL_CLOSED_CAMPAIGN_AUDIT":
            raise ValueError("candidate kernel campaign audit is invalid")
        for family, audit_row in campaign_audit["families"].items():
            if audit_row["status"] != "VALID_FAMILY_AUDIT":
                continue
            family_row = campaign["families"][family]
            add_candidate_kernel_result({
                "kernel_family": family,
                "selected_call_indices": [0],
                "repairs": {"0": family_row},
            })
    semantic_invocations = sum(row["invocations"] for row in semantic)
    valid_invocations = sum(row["valid_invocations"] for row in semantic)
    barrier_conditioned_invocations = sum(len(row.get("barrier_conditioned_layers", [])) for row in semantic)
    expected = coverage["metrics"]["declared_high_level_forward_invocations"]
    if semantic_invocations != expected:
        raise ValueError(f"semantic denominator mismatch: {semantic_invocations} != {expected}")

    triton_candidate_evidence = {
        family: evidence
        for family, evidence in candidate_kernel_evidence.items()
        if not family.startswith("extern:")
    }
    external_candidate_evidence = {
        family: evidence
        for family, evidence in candidate_kernel_evidence.items()
        if family.startswith("extern:")
    }

    kernel_classes = []
    for family in kernels["kernel_families"]:
        evidence = candidate_kernel_evidence.get(family["name"])
        if evidence:
            treatment_state = (
                f"PARTIAL_ORIGINAL_CANDIDATE_REPAIR_{evidence['repairs']}_OF_{family['call_count']}"
                f"__{evidence['nonzero']}_EFFECT_{evidence['null']}_NULL"
            )
        elif family["name"] in {
            "triton_per_fused__unsafe_view_add_mean_pow_rsqrt_16",
            "triton_poi_fused__to_copy__unsafe_view_19",
            "triton_poi_fused__to_copy__unsafe_view_add_mul_slice_18",
            "triton_poi_fused__to_copy_t_17",
        }:
            treatment_state = "PARTIAL_MODULE_LEVEL_NULL_EFFECT"
        else:
            treatment_state = "MAPPED_NOT_INTERVENED"
        kernel_classes.append({
            "treatment_id": family["name"],
            "kind": "generated_triton_family",
            "calls": family["call_count"],
            "constituent_aten": family["original_aten"],
            "coverage_state": treatment_state,
            "fully_covered": False,
            "original_candidate_evidence": evidence,
        })
    for name, calls in sorted(kernels["extern_kernel_call_counts"].items()):
        treatment_id = f"extern:{name}"
        evidence = external_candidate_evidence.get(treatment_id)
        if evidence:
            treatment_state = (
                f"PARTIAL_SHARED_PATH_REEXECUTION_{evidence['repairs']}_OF_{calls}"
                f"__{evidence['nonzero']}_EFFECT_{evidence['null']}_NULL"
            )
        elif name == "mm":
            treatment_state = "PARTIAL_MODULE_LEVEL_NULL_EFFECT"
        else:
            treatment_state = "MAPPED_NOT_INTERVENED"
        kernel_classes.append({
            "treatment_id": treatment_id,
            "kind": "external_kernel_family",
            "calls": calls,
            "constituent_aten": [f"aten.{name}"],
            "coverage_state": treatment_state,
            "fully_covered": False,
            "original_candidate_evidence": evidence,
        })

    repeated_classes = [row for row in semantic if row["equivalence_status"] == "PROPOSED_UNVALIDATED"]
    minimum_representatives = sum(len(row["minimum_transport_representatives"]) for row in semantic)
    invalid_representative_attempts = 9 + (1 if causal_mask_status == "INVALID_TREATMENT" else 0)
    representative_attempts = valid_invocations + barrier_conditioned_invocations + invalid_representative_attempts
    payload = {
        "schema_version": "forkcert.qwen3-operator-equivalence.v0.1",
        "scope": coverage["scope"],
        "semantics": {
            "invocation_exhaustive_mode": "repair/injection every declared invocation; 536 forward invocation units",
            "equivalence_mode": "replace invocation coverage only after role/fusion/state transport gates pass",
            "no_credit_rule": "candidate equivalence classes grant zero coverage until validated",
        },
        "metrics": {
            "semantic_role_classes": len(semantic),
            "repeated_candidate_equivalence_classes": len(repeated_classes),
            "singleton_classes": len(semantic) - len(repeated_classes),
            "semantic_invocations": semantic_invocations,
            "existing_valid_invocations": valid_invocations,
            "barrier_conditioned_invocations": barrier_conditioned_invocations,
            "composite_sdpa_barrier_invocations": composite_sdpa_invocations,
            "sdpa_decomposition_primitive_coverage": 0,
            "minimum_representative_invocations_if_transport_succeeds": minimum_representatives,
            "representative_attempts": representative_attempts,
            "usable_representative_interventions": valid_invocations + barrier_conditioned_invocations,
            "invalid_representative_treatments": invalid_representative_attempts,
            "original_candidate_representative_interventions": valid_invocations,
            "generated_treatment_families": len(kernel_classes),
            "fully_covered_treatment_families": sum(row["fully_covered"] for row in kernel_classes),
            "original_candidate_generated_kernel_repairs": sum(row["repairs"] for row in triton_candidate_evidence.values()),
            "original_candidate_generated_kernel_nonzero_effects": sum(row["nonzero"] for row in triton_candidate_evidence.values()),
            "original_candidate_generated_kernel_null_effects": sum(row["null"] for row in triton_candidate_evidence.values()),
            "original_candidate_generated_kernel_repairs_closer_to_eager": sum(row["closer"] for row in triton_candidate_evidence.values()),
            "original_candidate_generated_kernel_repairs_farther_from_eager": sum(row["farther"] for row in triton_candidate_evidence.values()),
            "original_candidate_generated_kernel_families_partially_covered": len(triton_candidate_evidence),
            "original_candidate_external_reexecutions": sum(row["repairs"] for row in external_candidate_evidence.values()),
            "original_candidate_external_reexecution_nonzero_effects": sum(row["nonzero"] for row in external_candidate_evidence.values()),
            "original_candidate_external_reexecution_null_effects": sum(row["null"] for row in external_candidate_evidence.values()),
            "original_candidate_external_families_partially_covered": len(external_candidate_evidence),
        },
        "semantic_role_classes": semantic,
        "generated_treatment_classes": kernel_classes,
        "transport_gate": [
            "same semantic role, shape family, dtype and compiler configuration",
            "same declared fusion-context class",
            "valid repair and injection treatment integrity",
            "early/middle/late representatives for repeated classes",
            "effect compatibility across the declared matched-state distribution",
            "null, non-null and invalid results all retained",
        ],
        "non_semantic_atomic_policy": {
            "units": ["cast", "view/layout", "clone/materialization", "index/slice", "mask construction"],
            "rule": "do not discard as bookkeeping; cover through generated treatment family or a separately valid primitive intervention",
            "current_state": "MAPPED_OR_OBSERVED_ONLY",
        },
        "verdict": "CANDIDATE_EQUIVALENCE_PARTITION_COMPLETE_FOR_FROZEN_FORWARD; TRANSPORT_UNVALIDATED; CAUSAL_COVERAGE_INCOMPLETE",
        "invalid_treatments": {
            "sdpa_qk_softmax_pv_source_reconstruction": sdpa_decomposition_status,
            "causal_mask_none_tensor_continuation": causal_mask_status
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3 operator equivalence ledger v0.1",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "The classes below are candidate experimental strata, not proven equivalence classes. They receive no causal-coverage credit until the transport gate passes.",
        "",
        "## Semantic-role denominator",
        "",
        "| Class | Family | Role | Calls | Original-valid | State | Candidate representatives | Fusion context |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for row in semantic:
        reps = ",".join(map(str, row["minimum_transport_representatives"]))
        lines.append(f"| `{row['class_id']}` | {row['family']} | {row['semantic_role']} | {row['invocations']} | {row['valid_invocations']} | {row['coverage_state']} | {reps} | {row['fusion_context']} |")
    lines += [
        "",
        f"The denominator is {semantic_invocations} forward invocations in {len(semantic)} semantic-role/context classes. If all transport assumptions succeed, at least {minimum_representatives} representative invocations are still required; {valid_invocations} original-candidate-valid singleton interventions and {barrier_conditioned_invocations} non-transported barrier-conditioned interventions currently count in their separate categories.",
        f"Additionally, {composite_sdpa_invocations} fixed-boundary SDPA invocations have joint composite evidence. They receive no separate qk-bmm, softmax, or pv-bmm coverage credit.",
        "",
        "## Generated-treatment denominator",
        "",
        "| Treatment | Kind | Calls | State |",
        "|---|---|---:|---|",
    ]
    for row in kernel_classes:
        lines.append(f"| `{row['treatment_id']}` | {row['kind']} | {row['calls']} | {row['coverage_state']} |")
    lines += [
        "",
        "No generated treatment family is fully covered. A valid module-level null effect may partially constrain several generated kernels, but it cannot identify which constituent kernel or primitive is null.",
        "",
        "## Structural and precision operations",
        "",
        "Casts, views/layout operations, clones, indexing and mask construction remain in the denominator. They must be covered through a generated treatment family or a valid primitive intervention; they are not silently treated as harmless bookkeeping.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
