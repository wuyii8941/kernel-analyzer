#!/usr/bin/env python
"""Independently audit original-compiled eager-boundary mediation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disagreement_coordinates(left: list[list[bool]], right: list[list[bool]]) -> list[list[int]]:
    return [
        [sample, token]
        for sample, (left_row, right_row) in enumerate(zip(left, right, strict=True))
        for token, (left_value, right_value) in enumerate(
            zip(left_row, right_row, strict=True)
        )
        if left_value != right_value
    ]


def validate(
    report: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    gate: dict[str, Any],
    baseline_eager: dict[str, Any],
    baseline_compiled: dict[str, Any],
) -> list[str]:
    errors = []
    if report.get("valid") is not True or report.get("status") != "VALID":
        errors.append("runner report is not valid")
    failed = [name for name, passed in report.get("gates", {}).items() if passed is not True]
    if failed:
        errors.append(f"failed runner gates: {failed}")
    if gate.get("forward_kernel_inventory_eligible") is not True:
        errors.append("forward observability/provenance gate is not eligible")
    contract = json.loads(Path(manifest["realization_contract"]).read_text())
    anchors = report.get("anchors", {})
    if anchors.get("eager") != [contract["reference_scorer_sha256"]] * 2:
        errors.append("eager anchor mismatch")
    if anchors.get("candidate") != [contract["candidate_scorer_sha256"]] * 2:
        errors.append("candidate anchor mismatch")
    if anchors.get("restored") != anchors.get("candidate"):
        errors.append("restored candidate mismatch")

    expected_forks = disagreement_coordinates(
        baseline_eager["semantic"]["clip_decisions"],
        baseline_compiled["semantic"]["clip_decisions"],
    )
    if report.get("baseline_endpoint", {}).get("fork_coordinates") != expected_forks:
        errors.append("baseline fork coordinates differ from independent transitions")
    expected_kernel = manifest["generated_kernel"]
    rows = [
        row for row in inventory.get("kernels", [])
        if row.get("generated_symbol") == expected_kernel
    ]
    if len(rows) != int(manifest["expected_runtime_calls"]):
        errors.append("inventory kernel-family cardinality mismatch")
    reported_rows = report.get("kernel_family", {}).get("provenance_rows", [])
    if [row.get("kernel_id") for row in reported_rows] != [row.get("kernel_id") for row in rows]:
        errors.append("reported provenance rows differ from inventory")
    intermediate_symbol = manifest.get("intermediate_generated_kernel")
    if intermediate_symbol:
        intermediate_rows = [
            row
            for row in inventory.get("kernels", [])
            if row.get("generated_symbol") == intermediate_symbol
        ]
        reported_intermediate = report.get("kernel_family", {}).get(
            "intermediate_provenance_rows", []
        )
        if len(intermediate_rows) != len(reported_intermediate):
            errors.append("intermediate provenance cardinality mismatch")
        if [row.get("kernel_id") for row in reported_intermediate] != [
            row.get("kernel_id") for row in intermediate_rows
        ]:
            errors.append("reported intermediate provenance rows differ from inventory")
        # Static generated-code provenance is separate from runtime numerical
        # evidence.  It records which fused operations are present in the
        # declared kernel, without treating any one of them as causal.
        static_code_paths = {
            str(row.get("output_code_path")) for row in intermediate_rows
        }
        static_code_operations = set()
        for code_path in static_code_paths:
            path = Path(code_path)
            if not path.is_file():
                errors.append(f"intermediate generated-code path missing: {code_path}")
                continue
            code = path.read_text()
            if "tl.sum" in code:
                static_code_operations.add("reduction_sum")
            if "libdevice.rsqrt" in code:
                static_code_operations.add("rsqrt")
            if ".to(tl.float32)" in code:
                static_code_operations.add("fp32_conversion")
            if "tl.store" in code:
                static_code_operations.add("output_store")
            for row in intermediate_rows:
                if row.get("output_code_path") == code_path and row.get(
                    "output_code_sha256"
                ):
                    if sha256_file(path) != row["output_code_sha256"]:
                        errors.append(f"intermediate generated-code hash mismatch: {code_path}")
        if not {"reduction_sum", "rsqrt", "fp32_conversion", "output_store"}.issubset(
            static_code_operations
        ):
            errors.append("intermediate generated-code operation evidence incomplete")

    selected = [str(int(index)) for index in manifest["selected_call_indices"]]
    treatments = report.get("treatments", {})
    if sorted(treatments, key=int) != sorted(selected, key=int):
        errors.append("treatment set differs from manifest")
    expected_calls = int(manifest["expected_runtime_calls"])
    for treatment_id in selected:
        row = treatments.get(treatment_id, {})
        noop = row.get("noop", {})
        intervention = row.get("intervention", {})
        if noop.get("hashes") != anchors.get("candidate"):
            errors.append(f"treatment {treatment_id}: no-op differs from candidate")
        if not all(
            record == [{"calls": expected_calls, "interventions": 0}]
            for record in noop.get("call_records", [])
        ):
            errors.append(f"treatment {treatment_id}: no-op call counts invalid")
        if not all(
            record == [{"calls": expected_calls, "interventions": 1}]
            for record in intervention.get("call_records", [])
        ):
            errors.append(f"treatment {treatment_id}: intervention call counts invalid")
        records = intervention.get("boundary_records", [])
        if not (
            intervention.get("boundary_record_repeat_exact") is True
            and len(records) == 2
            and len(records[0]) == len(records[1]) == 1
            and records[0][0] == records[1][0]
        ):
            errors.append(f"treatment {treatment_id}: boundary records not exact repeats")
            continue
        record = records[0][0]
        if record.get("call_index") != int(treatment_id):
            errors.append(f"treatment {treatment_id}: intercepted wrong call")
        for required in (
            "weight_storage_identity",
            "weight_exact",
            "residual_value_transport_contract",
            "norm_value_transport_contract",
            "destination_layout_preserved",
        ):
            if record.get(required) is not True:
                errors.append(f"treatment {treatment_id}: failed {required}")
        mediation = row.get("fixed_original_suffix_mediation", {})
        if mediation.get("observed_continuous") is not True:
            errors.append(f"treatment {treatment_id}: no continuous mediation")
        event_count = int(mediation.get("off_to_on", 0)) + int(
            mediation.get("on_to_off", 0)
        )
        if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
            errors.append(f"treatment {treatment_id}: semantic counts/rate mismatch")
    expected_contextual = [
        str(int(index)) for index in manifest.get("contextual_layer_slices", [])
    ]
    contextual = report.get("contextual_layer_slices", {})
    if sorted(contextual, key=int) != sorted(expected_contextual, key=int):
        errors.append("contextual layer slice set differs from manifest")
    for layer_id in expected_contextual:
        row = contextual.get(layer_id, {})
        expected_counts = {
            "noop": {"calls": expected_calls, "entry_injections": 0, "exit_injections": 0},
            "compiled_block": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 0},
            "eager_block": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 1},
        }
        for arm_name, expected_count in expected_counts.items():
            arm = row.get(arm_name, {})
            if arm.get("repeat_exact") is not True:
                errors.append(f"layer {layer_id}: {arm_name} repeats differ")
            if not all(
                record == [expected_count] for record in arm.get("call_records", [])
            ):
                errors.append(f"layer {layer_id}: {arm_name} call counts invalid")
        if row.get("noop", {}).get("hashes") != anchors.get("candidate"):
            errors.append(f"layer {layer_id}: contextual no-op differs from candidate")
        if row.get("compiled_block", {}).get(
            "matches_independent_entry_boundary_treatment"
        ) is not True:
            errors.append(f"layer {layer_id}: entry-boundary transport mismatch")
        if row.get("eager_block", {}).get(
            "matches_independent_exit_boundary_treatment"
        ) is not True:
            errors.append(f"layer {layer_id}: exit-boundary transport mismatch")
        production = row.get("same_eager_input_composite_layer_production", {})
        if not (
            production.get("observed") is True
            and production.get("compiled_exit_records_repeat_exact") is True
            and production.get("compiled_and_eager_arms_pre_repair_exit_exact") is True
        ):
            errors.append(f"layer {layer_id}: same-input composite production invalid")
        exit_record = production.get("compiled_exit_record", {})
        if exit_record.get("call_index") != int(layer_id):
            errors.append(f"layer {layer_id}: production observed at wrong exit")
        for required in (
            "weight_storage_identity",
            "weight_exact",
            "residual_value_transport_contract",
            "norm_value_transport_contract",
            "destination_layout_preserved",
        ):
            if exit_record.get(required) is not True:
                errors.append(f"layer {layer_id}: production failed {required}")
        mediation = row.get("fixed_original_suffix_layer_mediation", {})
        if mediation.get("observed_continuous") is not True:
            errors.append(f"layer {layer_id}: no composite-layer mediation")
        event_count = int(mediation.get("off_to_on", 0)) + int(
            mediation.get("on_to_off", 0)
        )
        if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
            errors.append(f"layer {layer_id}: semantic counts/rate mismatch")

    # A subblock slice is a stricter longitudinal unit than a layer boundary:
    # it has an intermediate generated kernel (attention side) and a later
    # layer-exit kernel (MLP side).  Audit both production and mediation
    # independently; a passing intervention alone must not license either
    # local-production or unique-kernel claims.
    expected_subblocks = [
        str(int(index)) for index in manifest.get("subblock_layer_slices", [])
    ]
    subblocks = report.get("subblock_layer_slices", {})
    if sorted(subblocks, key=int) != sorted(expected_subblocks, key=int):
        errors.append("subblock layer slice set differs from manifest")
    for layer_id in expected_subblocks:
        row = subblocks.get(layer_id, {})
        arms = row.get("arms", {})
        variant_names = [
            str(name) for name in manifest.get("kernel_op_variants", [])
        ]
        live_variant_names = [
            str(name) for name in manifest.get("live_kernel_variants", [])
        ]
        expected_counts = {
            "noop": {"calls": expected_calls, "entry_injections": 0, "exit_injections": 0},
            "compiled_attention": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 0},
            "eager_attention": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 0},
            "kernel_reference": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 0},
            "eager_block": {"calls": expected_calls, "entry_injections": 1, "exit_injections": 1},
        }
        expected_counts.update(
            {
                f"kernel_variant:{name}": {
                    "calls": expected_calls,
                    "entry_injections": 1,
                    "exit_injections": 0,
                }
                for name in variant_names
            }
        )
        expected_counts.update(
            {
                f"live_kernel_variant:{name}": {
                    "calls": expected_calls,
                    "entry_injections": 1,
                    "exit_injections": 0,
                }
                for name in live_variant_names
            }
        )
        for arm_name, expected in expected_counts.items():
            arm = arms.get(arm_name, {})
            if arm.get("repeat_exact") is not True:
                errors.append(f"subblock {layer_id}: {arm_name} repeats differ")
            if arm_name == "noop" and arm.get("hashes") != anchors.get("candidate"):
                errors.append(f"subblock {layer_id}: no-op differs from candidate")
            for call_record in arm.get("call_records", []):
                boundary = call_record.get("boundary")
                intermediate = call_record.get("intermediate")
                if boundary != [expected]:
                    errors.append(f"subblock {layer_id}: {arm_name} boundary call counts invalid")
                expected_intermediate = {"calls": expected_calls + 1, "injections": 0}
                if arm_name in ("eager_attention", "kernel_reference", "eager_block") or arm_name.startswith("kernel_variant:"):
                    expected_intermediate["injections"] = 1
                if intermediate != [expected_intermediate]:
                    errors.append(f"subblock {layer_id}: {arm_name} intermediate call counts invalid")

        for arm_name in ("compiled_attention", "eager_block"):
            if arms.get(arm_name, {}).get(
                "matches_independent_entry_boundary_treatment"
                if arm_name == "compiled_attention"
                else "matches_independent_exit_boundary_treatment"
            ) is not True:
                errors.append(f"subblock {layer_id}: {arm_name} cross-artifact transport mismatch")

        boundary_records = [
            record
            for arm_name in arms
            for repeat in arms[arm_name].get("boundary_records", [])
            for record in repeat
        ]
        intermediate_records = [
            record
            for arm_name in arms
            for repeat in arms[arm_name].get("intermediate_records", [])
            for record in repeat
        ]
        for record in boundary_records:
            for required in ("weight_storage_identity", "weight_exact", "destination_layout_preserved"):
                if record.get(required) is not True:
                    errors.append(f"subblock {layer_id}: boundary failed {required}")
        for record in intermediate_records:
            for required in (
                "weight_storage_identity",
                "weight_exact",
                "attention_transport_contract",
                "post_norm_transport_contract",
                "destination_layout_preserved",
            ):
                if record.get(required) is not True:
                    errors.append(f"subblock {layer_id}: intermediate failed {required}")

        attention_production = row.get("same_eager_input_attention_region_production", {})
        if not (
            attention_production.get("observed") is True
            and attention_production.get("records_repeat_exact") is True
            and attention_production.get("compiled_and_eager_attention_arms_pre_repair_exact") is True
        ):
            errors.append(f"subblock {layer_id}: same-input attention production invalid")
        attention_record = attention_production.get("compiled_attention_record", {})
        if attention_record.get("call_index") != int(layer_id):
            errors.append(f"subblock {layer_id}: attention production observed at wrong call")

        mlp_production = row.get("same_eager_attention_input_mlp_region_production", {})
        if not (
            mlp_production.get("observed") is True
            and mlp_production.get("records_repeat_exact") is True
            and mlp_production.get("eager_attention_and_eager_block_arms_pre_repair_exit_exact") is True
        ):
            errors.append(f"subblock {layer_id}: same-input MLP production invalid")
        mlp_record = mlp_production.get("compiled_mlp_exit_record", {})
        if mlp_record.get("call_index") != int(layer_id):
            errors.append(f"subblock {layer_id}: MLP production observed at wrong call")

        kernel_production = row.get("same_input_intermediate_kernel_production", {})
        kernel_record = kernel_production.get("record", {})
        if kernel_production.get("generated_kernel") != manifest.get(
            "intermediate_generated_kernel"
        ):
            errors.append(f"subblock {layer_id}: intermediate kernel identity mismatch")
        if kernel_production.get("call_index") != int(layer_id):
            errors.append(f"subblock {layer_id}: kernel production observed at wrong call")
        for required in (
            "same_input_kernel_post_norm",
            "same_input_kernel_post_norm_production",
            "same_input_kernel_reference_post_norm_sha256",
            "same_input_kernel_input_residual_sha256",
            "same_input_kernel_input_attention_sha256",
        ):
            if required not in kernel_record:
                errors.append(f"subblock {layer_id}: missing {required}")
        if kernel_record.get("call_index") != int(layer_id):
            errors.append(f"subblock {layer_id}: kernel record call index mismatch")
        live_module_name = kernel_record.get("live_module_name")
        if live_module_name is not None and live_module_name not in report.get(
            "kernel_family", {}
        ).get("intermediate_live_module_names", []):
            errors.append(f"subblock {layer_id}: kernel live-module provenance mismatch")
        if not kernel_production.get("observed"):
            errors.append(f"subblock {layer_id}: kernel-local production not observed")
        compiled_kernel_records = arms.get("compiled_attention", {}).get(
            "intermediate_records", []
        )
        if not (
            len(compiled_kernel_records) == 2
            and all(len(repeat) == 1 for repeat in compiled_kernel_records)
        ):
            errors.append(f"subblock {layer_id}: kernel baseline records incomplete")
        else:
            for repeat in compiled_kernel_records:
                baseline_record = repeat[0]
                if baseline_record.get("compiled_attention_sha256") != baseline_record.get(
                    "post_attention_sha256"
                ):
                    errors.append(
                        f"subblock {layer_id}: kernel changed non-target attention buffer"
                    )
                if baseline_record.get("compiled_post_norm_sha256") != baseline_record.get(
                    "post_norm_sha256"
                ):
                    errors.append(
                        f"subblock {layer_id}: baseline kernel post-norm record changed before repair"
                    )
                for name in variant_names:
                    if name not in baseline_record.get("kernel_op_variant_metrics", {}):
                        errors.append(
                            f"subblock {layer_id}: missing operation variant metrics for {name}"
                        )
                    if name not in baseline_record.get("kernel_op_variant_sha256", {}):
                        errors.append(
                            f"subblock {layer_id}: missing operation variant hash for {name}"
                        )
                    if (
                        name == "reference_reduce"
                        and baseline_record.get("kernel_op_variant_sha256", {}).get(name)
                        != baseline_record.get("same_input_kernel_reference_post_norm_sha256")
                    ):
                        errors.append(
                            f"subblock {layer_id}: reference reduction does not reproduce reference post-norm"
                        )

        for mediation_name in (
            "fixed_original_suffix_attention_region_mediation",
            "fixed_original_suffix_mlp_region_mediation",
            "fixed_original_suffix_kernel_mediation",
        ):
            mediation = row.get(mediation_name, {})
            if mediation.get("observed_continuous") is not True:
                errors.append(f"subblock {layer_id}: {mediation_name} missing continuous mediation")
            event_count = int(mediation.get("off_to_on", 0)) + int(
                mediation.get("on_to_off", 0)
            )
            if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
                errors.append(f"subblock {layer_id}: {mediation_name} semantic counts/rate mismatch")
        variant_mediation = row.get("kernel_op_variant_mediation", {})
        for name in variant_names:
            mediation = variant_mediation.get(name, {})
            if mediation.get("observed_continuous") is not True:
                errors.append(
                    f"subblock {layer_id}: operation variant {name} has no continuous mediation"
                )
            event_count = int(mediation.get("off_to_on", 0)) + int(
                mediation.get("on_to_off", 0)
            )
            if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
                errors.append(
                    f"subblock {layer_id}: operation variant {name} semantic counts/rate mismatch"
                )
        live_mediation = row.get("live_kernel_variant_mediation", {})
        for name in live_variant_names:
            mediation = live_mediation.get(name, {})
            if mediation.get("observed_continuous") is not True:
                errors.append(
                    f"subblock {layer_id}: live kernel variant {name} has no continuous mediation"
                )
            event_count = int(mediation.get("off_to_on", 0)) + int(
                mediation.get("on_to_off", 0)
            )
            if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
                errors.append(
                    f"subblock {layer_id}: live kernel variant {name} semantic counts/rate mismatch"
                )
        for arm_name in (f"live_kernel_variant:{name}" for name in live_variant_names):
            for repeat in arms.get(arm_name, {}).get("intermediate_records", []):
                for record in repeat:
                    if record.get("live_kernel_variant") != arm_name.split(":", 1)[1]:
                        errors.append(
                            f"subblock {layer_id}: live variant record identity mismatch for {arm_name}"
                        )
                    if not record.get("live_kernel_variant_metadata", {}).get(
                        "modified_output_code_sha256"
                    ):
                        errors.append(
                            f"subblock {layer_id}: live variant metadata incomplete for {arm_name}"
                        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--observability-gate", required=True)
    parser.add_argument("--baseline-eager", required=True)
    parser.add_argument("--baseline-compiled", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "report": args.report,
            "manifest": args.manifest,
            "inventory": args.inventory,
            "observability_gate": args.observability_gate,
            "baseline_eager": args.baseline_eager,
            "baseline_compiled": args.baseline_compiled,
        }.items()
    }
    values = {key: json.loads(path.read_text()) for key, path in paths.items()}
    errors = validate(
        values["report"],
        values["manifest"],
        values["inventory"],
        values["observability_gate"],
        values["baseline_eager"],
        values["baseline_compiled"],
    )
    semantic_boundaries = [
        int(key)
        for key, row in values["report"].get("treatments", {}).items()
        if row.get("fixed_original_suffix_mediation", {}).get(
            "semantic_disagreement", 0.0
        ) > 0
    ]
    payload = {
        "schema_version": "forkcert.eager-layer-boundary-mediation-audit.v0.1",
        "valid": not errors,
        "verdict": "VALID" if not errors else "INVALID",
        "errors": errors,
        "evidence_level": (
            (
                "DIRECT_GENERATED_KERNEL_CODE_INTERVENTION_AND_FIXED_SUFFIX_MEDIATION"
                if values["manifest"].get("live_kernel_variants")
                else "SAME_INPUT_KERNEL_AND_SUBBLOCK_PRODUCTION_AND_FIXED_SUFFIX_MEDIATION"
                if values["manifest"].get("subblock_layer_slices")
                else "SAME_INPUT_COMPOSITE_LAYER_PRODUCTION_AND_FIXED_SUFFIX_MEDIATION"
                if values["manifest"].get("contextual_layer_slices")
                else "ORIGINAL_COMPILED_SUFFIX_LAYER_BOUNDARY_MEDIATION"
            )
            if not errors
            else "INVALID"
        ),
        "semantic_mediation_boundaries": semantic_boundaries,
        "boundary_treatments_production_claim_allowed": False,
        "contextual_composite_layer_production_claim_allowed": (
            not errors and bool(values["manifest"].get("contextual_layer_slices"))
        ),
        "subblock_production_claim_allowed": (
            not errors and bool(values["manifest"].get("subblock_layer_slices"))
        ),
        "kernel_local_production_claim_allowed": (
            not errors
            and bool(values["manifest"].get("subblock_layer_slices"))
            and all(
                row.get("same_input_intermediate_kernel_production", {}).get(
                    "observed"
                )
                is True
                for row in values["report"].get("subblock_layer_slices", {}).values()
            )
        ),
        "kernel_mediation_claim_allowed": (
            not errors and bool(values["manifest"].get("subblock_layer_slices"))
        ),
        "static_kernel_code_provenance_exact": (
            not errors
            and bool(values["manifest"].get("subblock_layer_slices"))
            and bool(values["report"].get("kernel_family", {}).get(
                "intermediate_provenance_rows"
            ))
        ),
        "reference_reduce_stage_exact": (
            not errors
            and "reference_reduce" in values["manifest"].get("kernel_op_variants", [])
            and all(
                record.get("kernel_op_variant_sha256", {}).get("reference_reduce")
                == record.get("same_input_kernel_reference_post_norm_sha256")
                for row in values["report"].get("subblock_layer_slices", {}).values()
                for repeat in row.get("arms", {}).get("compiled_attention", {}).get(
                    "intermediate_records", []
                )
                for record in repeat
            )
        ),
        "live_generated_kernel_code_intervention_exact": (
            not errors
            and bool(values["manifest"].get("live_kernel_variants"))
            and all(
                record.get("live_kernel_variant_metadata", {}).get(
                    "source_output_code_sha256"
                )
                and record.get("live_kernel_variant_metadata", {}).get(
                    "modified_output_code_sha256"
                )
                != record.get("live_kernel_variant_metadata", {}).get(
                    "source_output_code_sha256"
                )
                for row in values["report"].get("subblock_layer_slices", {}).values()
                for name in values["manifest"].get("live_kernel_variants", [])
                for repeat in row.get("arms", {}).get(
                    f"live_kernel_variant:{name}", {}
                ).get("intermediate_records", [])
                for record in repeat
            )
        ),
        "unique_kernel_root_cause_claim_allowed": False,
        "unique_operator_root_cause_claim_allowed": False,
        "contextual_composite_layers": sorted(
            int(key)
            for key in values["report"].get("contextual_layer_slices", {})
        ),
        "subblock_layers": sorted(
            int(key)
            for key in values["report"].get("subblock_layer_slices", {})
        ),
        "artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "claim_limits": (
            [
                "same-input production is licensed for the declared attention/MLP subblocks and the declared intermediate kernel post-norm output; it is not a unique constituent-op explanation",
                "kernel-only repair is mediation evidence for the declared generated kernel in this fixed context, not a unique compiler root cause",
                "subblock mediation is intervention-dependent and does not establish a unique kernel root cause",
                "the attention and MLP effects are conditioned on the declared layer-15 entry and suffix context",
                "one selected state and forward endpoint only",
                "implementation-relative, not correctness",
            ]
            if values["manifest"].get("subblock_layer_slices")
            else [
                "same-input production is licensed only for the declared composite layer, not its constituent ops or kernels",
                "composite-layer mediation is not a unique source-op root cause",
                "one selected state and forward endpoint only",
                "implementation-relative, not correctness",
            ]
            if values["manifest"].get("contextual_layer_slices")
            else [
                "boundary-state mediation, not same-input local production",
                "an influential prefix interval is not a unique source op",
                "one selected state and forward endpoint only",
                "implementation-relative, not correctness",
            ]
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["valid"] else 2)


if __name__ == "__main__":
    main()
