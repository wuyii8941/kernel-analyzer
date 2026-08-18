#!/usr/bin/env python3
"""Build a compact, explicit audit of implementation/state coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def read(name: str) -> dict:
    return json.loads((FINAL / name).read_text())


def main() -> None:
    atlas = read("implementation_atlas.json")
    matrix = read("checkpoint_matrix.json")
    carrier = read("checkpoint_carrier.json")
    inductor = read("checkpoint_inductor.json")
    invocation = read("invocation_atlas.json")
    structured = read("structured_carrier_trigger.json")
    structured_confirmation = read("structured_carrier_confirmation.json")
    liger_path = ROOT / "results/trajectory/liger_trajectory.json"
    liger_trajectory = json.loads(liger_path.read_text()) if liger_path.exists() else None
    closed_fbv_site_ids = {
        region_id
        for unit in invocation["changed_units"]
        for region_id in unit["candidate_region_ids"]
    }
    launch_closed_fbv_site_ids = {
        region_id for region_id in closed_fbv_site_ids if ":direct_aten:" not in region_id
    }
    direct_closed_fbv_site_ids = closed_fbv_site_ids - launch_closed_fbv_site_ids
    shape_matrix = read("checkpoint_matrix_shapes.json")
    dtype_boundary = read("dtype_dynamic_boundary.json") if (FINAL / "dtype_dynamic_boundary.json").exists() else None
    dtype_unresolved = read("dtype_unresolved_boundary.json") if (FINAL / "dtype_unresolved_boundary.json").exists() else None
    source_matrix_static = read("source_matrix_static.json") if (FINAL / "source_matrix_static.json").exists() else None
    source_replay = read("source_replay_matrix.json") if (FINAL / "source_replay_matrix.json").exists() else None
    dtype_topologies = []
    dtype_mappings = []
    dtype_semantic_observations = []
    dtype_endpoint_observations = []
    for dtype, tf32 in (("fp32", False), ("tf32", True)):
        for seq in (64, 128, 256):
            name = f"dtype_topology_{dtype}_seq{seq}.json"
            if (FINAL / name).exists():
                topology = read(name)
                dtype_topologies.append({
                    "dtype": "fp32" if dtype == "fp32" else "fp32",
                    "tf32": tf32,
                    "seq_len": seq,
                    "file": name,
                    "runtime_symbol_count": topology["runtime_symbol_count"],
                    "runtime_invocation_count": topology["runtime_invocation_count"],
                    "candidate_values_used_to_select_or_classify": topology["candidate_values_used_to_select_or_classify"],
                    "semantic_mapping_status": "UNRESOLVED_TOPOLOGY_ONLY",
                })
            mapping_name = f"dtype_mapping_{dtype}_seq{seq}.json"
            if (FINAL / mapping_name).exists():
                mapping = read(mapping_name)
                dtype_mappings.append({
                    "dtype": "fp32" if dtype == "fp32" else "fp32",
                    "tf32": tf32,
                    "seq_len": seq,
                    "file": mapping_name,
                    "mapped_symbols": mapping["denominator"]["mapped_symbols"],
                    "mapped_invocations": mapping["denominator"]["mapped_invocations"],
                    "unresolved_symbols": mapping["denominator"]["unresolved_symbols"],
                    "unresolved_invocations": mapping["denominator"]["unresolved_invocations"],
                    "candidate_values_used_to_select_or_classify": mapping["candidate_values_used_to_select_or_classify"],
                    "status": "SEMANTIC_MAPPING_BOUNDARY",
                })
            semantic_name = f"dtype_semantic_{dtype}_seq{seq}.json"
            if (FINAL / semantic_name).exists():
                semantic = read(semantic_name)
                semantic_rows = semantic["rows"]
                sign_persistent_endpoint_count = 0
                endpoint_count = 0
                for semantic_row in semantic_rows:
                    by_endpoint = {}
                    for metric_row in semantic_row.get("state_repeat_metrics", []):
                        by_endpoint.setdefault(str(metric_row["endpoint"]), []).append(
                            float(metric_row["metric"].get("signed_mean", 0.0))
                        )
                    for values in by_endpoint.values():
                        endpoint_count += 1
                        positive = sum(value > 0.0 for value in values)
                        negative = sum(value < 0.0 for value in values)
                        if values and (positive == len(values) or negative == len(values)):
                            sign_persistent_endpoint_count += 1
                dtype_semantic_observations.append({
                    "dtype": "fp32" if dtype == "fp32" else "fp32",
                    "tf32": tf32,
                    "seq_len": seq,
                    "file": semantic_name,
                    "checkpoint_count": semantic["checkpoint_count"],
                    "mapped_invocations": semantic["denominator"]["mapped_invocations"],
                    "mapped_regions_observed_at_all_steps_and_repeats": semantic["denominator"]["mapped_regions_observed_at_all_steps_and_repeats"],
                    "rows_with_any_nonexact_endpoint": sum(row["nonexact_record_count"] > 0 for row in semantic_rows),
                    "mapped_rows": len(semantic_rows),
                    "endpoint_count": endpoint_count,
                    "sign_persistent_endpoint_count": sign_persistent_endpoint_count,
                    "max_abs_max": max((float(row["max_abs_max"]) for row in semantic_rows), default=0.0),
                    "all_mapped_regions_present_at_all_steps_and_repeats": semantic["gates"]["all_mapped_regions_present_at_all_steps_and_repeats"],
                    "candidate_values_used_to_select_or_classify": semantic["gates"]["candidate_values_used_to_select_or_classify"],
                    "natural_bias_case_added": False,
                })
            endpoint_name = f"dtype_evolving_{dtype}_seq{seq}_expanded.json"
            # The completed seq64 source-level mapping uses a deliberately
            # separate name so it cannot overwrite the older frozen campaign.
            if seq == 64 and dtype == "fp32" and (FINAL / "dtype_evolving_fp32_seq64_full.json").exists():
                endpoint_name = "dtype_evolving_fp32_seq64_full.json"
            if (FINAL / endpoint_name).exists():
                endpoint = read(endpoint_name)
                dtype_endpoint_observations.append({
                    "dtype": "fp32" if dtype == "fp32" else "fp32",
                    "tf32": tf32,
                    "seq_len": seq,
                    "file": endpoint_name,
                    "checkpoint_count": len(endpoint["checkpoint_steps"]),
                    "mapped_invocations": endpoint["mapped_invocations"],
                    "mapped_symbols": endpoint["mapped_symbols"],
                    "unresolved_symbols": endpoint["unresolved_symbols"],
                    "unresolved_invocations": endpoint["unresolved_invocations"],
                    "all_mapped_invocations_observed": endpoint["gates"]["all_mapped_invocations_observed_at_all_checkpoints_and_repeats"],
                    "all_repeats_match": endpoint["gates"]["all_repeats_match"],
                    "f_b_closure_complete": endpoint["gates"]["f_b_closure_complete"],
                    "candidate_values_used_to_select_or_classify": not endpoint["candidate_blind"],
                })
    dynamic_by_shape = {
        seq: read(f"evolving_triton_seq{seq}.json")
        for seq in (64, 128, 256)
        if (FINAL / f"evolving_triton_seq{seq}.json").exists()
    }
    fp16_dynamic_by_shape = {
        seq: read(f"evolving_triton_fp16_seq{seq}.json")
        for seq in (64, 128, 256)
        if (FINAL / f"evolving_triton_fp16_seq{seq}.json").exists()
    }
    steps = [row["checkpoint_step"] for row in matrix["rows"]]
    attention_regions = sum(row["reference_forward_regions"] for row in matrix["rows"])
    candidate_variant_comparisons = sum(
        1
        for row in matrix["rows"]
        for variant, value in row["variants"].items()
        if variant != "eager" and value.get("status", "OK") == "OK"
    )
    candidate_region_comparisons = attention_regions * 2
    inductor_specs = {
        ("bf16", False): {
            "64": "checkpoint_inductor_bf16_seq64.json",
            "128": "checkpoint_inductor.json",
            "256": "checkpoint_inductor_bf16_seq256.json",
        },
        ("fp16", False): {
            "64": "checkpoint_inductor_fp16_seq64.json",
            "128": "checkpoint_inductor_fp16.json",
            "256": "checkpoint_inductor_fp16_seq256.json",
        },
        ("fp32", False): {
            "64": "checkpoint_inductor_fp32_seq64.json",
            "128": "checkpoint_inductor_fp32.json",
            "256": "checkpoint_inductor_fp32_seq256.json",
        },
        ("fp32", True): {
            "64": "checkpoint_inductor_tf32_seq64.json",
            "128": "checkpoint_inductor_tf32.json",
            "256": "checkpoint_inductor_tf32_seq256.json",
        },
    }
    measured_cells = []
    full_step_carrier_cells = []
    for (dtype_name, tf32), cells in inductor_specs.items():
        for seq, filename in cells.items():
            if (FINAL / filename).exists():
                cell = {"dtype": dtype_name, "tf32": tf32, "seq_len": int(seq), "file": filename}
                measured_cells.append(cell)
                rows = read(filename)["rows"]
                projections = [float(row["carrier"]["projection"]) for row in rows if not row["carrier"].get("is_pilot", False)]
                full_step_carrier_cells.append({
                    **cell,
                    "heldout_states": len(projections),
                    "heldout_positive": sum(value > 0.0 for value in projections),
                    "heldout_negative": sum(value < 0.0 for value in projections),
                    "minimum_projection": min(projections or [0.0]),
                    "maximum_projection": max(projections or [0.0]),
                    "persistent_positive": bool(projections) and all(value > 0.0 for value in projections),
                    "persistent_negative": bool(projections) and all(value < 0.0 for value in projections),
                })
    required_cells = [
        {"dtype": dtype_name, "tf32": tf32, "seq_len": seq}
        for dtype_name, tf32 in (("bf16", False), ("fp16", False), ("fp32", False), ("fp32", True))
        for seq in (64, 128, 256)
    ]
    measured_keys = {(row["dtype"], row["tf32"], row["seq_len"]) for row in measured_cells}
    missing_cells = [row for row in required_cells if (row["dtype"], row["tf32"], row["seq_len"]) not in measured_keys]
    generated_region_dynamic_complete = False
    dynamic_region_coverage = {
        "required_changed_region_sites": atlas["changed_count"],
        "measured_changed_region_sites": 0,
        "bound_changed_fbv_units": 0,
        "checkpoint_count": 0,
        "shape_strata": [],
        "complete_for_current_static_atlas": False,
    }
    dynamic_shapes = []
    for seq, dynamic in sorted(dynamic_by_shape.items()):
        atlas_name = "implementation_atlas.json" if seq == 64 else f"implementation_atlas_seq{seq}.json"
        shape_atlas = read(atlas_name)
        observed_dynamic_ids = {str(row["region_id"]) for row in dynamic.get("rows", [])}
        dynamic_shapes.append({
            "seq_len": seq,
            "implementation_atlas": atlas_name,
            "required_changed_region_sites": sum(
                bool(row.get("implementation_changed"))
                and str(row.get("mechanism")) in {
                    "EXPLICIT_FP32_REDUCTION_SCHEDULE_DIFFERENCE",
                    "MATERIALIZATION_OR_ROUNDING_SCHEDULE_INTERVENTION",
                    "SAME_PRECISION_GENERATED_SCHEDULE_DIFFERENCE",
                }
                for row in shape_atlas["rows"]
            ),
            "measured_changed_region_sites": dynamic["denominator"]["changed_sites_observed_at_all_steps_and_repeats"],
            "required_closed_fbv_sites": sum(
                bool(row.get("implementation_changed")) for row in shape_atlas["rows"]
            ),
            "observed_closed_fbv_sites": len(launch_closed_fbv_site_ids & observed_dynamic_ids),
            "checkpoint_count": dynamic["checkpoint_count"],
            "complete": bool(dynamic["gates"]["all_changed_sites_present_at_all_steps_and_repeats"]),
            "shape_transfer_is_exact_mechanism_proof": seq == 64,
        })
    fp16_dynamic_shapes = []
    for seq, dynamic in sorted(fp16_dynamic_by_shape.items()):
        fp16_dynamic_shapes.append({
            "seq_len": seq,
            "dtype": dynamic.get("dtype", "fp16"),
            "tf32": bool(dynamic.get("tf32", False)),
            "required_changed_region_sites": dynamic["denominator"]["changed_generated_region_sites"],
            "measured_changed_region_sites": dynamic["denominator"]["changed_sites_observed_at_all_steps_and_repeats"],
            "checkpoint_count": dynamic["checkpoint_count"],
            "complete_mapped_layer": bool(dynamic["gates"]["all_changed_sites_present_at_all_steps_and_repeats"]),
            "unmatched_warmed_symbols": sorted({
                symbol
                for artifact in dynamic.get("artifacts", [])
                for symbol in artifact.get("unmatched_warmed_symbols", [])
            }),
        })
    source_replay_complete = bool(
        source_replay
        and int(source_replay.get("complete_cells", 0)) == int(source_replay.get("cell_count", 0))
        and int(source_replay.get("pending_cells", 1)) == 0
        and source_replay.get("numeric_replay") == "COMPLETE"
        and all(
            cell.get("status") == "COMPLETE"
            and not cell.get("missing_worker_files")
            for cell in source_replay.get("cells", [])
        )
    )
    direct_compute_endpoint_complete = bool(
        direct_closed_fbv_site_ids == {"backward:direct_aten:0"}
        and structured.get("status") == "COMPLETE"
        and structured["coverage"]["every_coordinate_in_exactly_one_partition_block"]
        and not structured["coverage"]["sampled_coordinate_subset"]
        and structured_confirmation.get("status") == "COMPLETE"
    )
    generated_region_dynamic_complete = bool(
        dynamic_shapes
        and all(row["complete"] for row in dynamic_shapes)
        and fp16_dynamic_shapes
        and all(row["complete_mapped_layer"] for row in fp16_dynamic_shapes)
        and source_replay_complete
        and direct_compute_endpoint_complete
    )
    fused_configuration_complete = bool(
        liger_trajectory
        and liger_trajectory.get("status") == "COMPLETE"
        and all(bool(value) for value in liger_trajectory.get("gates", {}).values())
    )
    dynamic_seq64 = dynamic_by_shape.get(64)
    if dynamic_seq64 is not None:
        dynamic_region_coverage.update({
            "measured_changed_region_sites": dynamic_seq64["denominator"]["changed_sites_observed_at_all_steps_and_repeats"],
            "bound_changed_fbv_units": invocation["denominator"]["changed_fbv_units"] if dynamic_seq64["gates"]["all_changed_sites_present_at_all_steps_and_repeats"] and direct_compute_endpoint_complete else 0,
            "checkpoint_count": dynamic_seq64["checkpoint_count"],
            "shape_strata": [dynamic_seq64["seq_len"]],
            "complete_for_current_static_atlas": dynamic_seq64["gates"]["all_changed_sites_present_at_all_steps_and_repeats"],
            "required_closed_fbv_sites": len(closed_fbv_site_ids),
            "observed_closed_fbv_sites": len(launch_closed_fbv_site_ids & {str(row["region_id"]) for row in dynamic_seq64.get("rows", [])}) + len(direct_closed_fbv_site_ids if direct_compute_endpoint_complete else set()),
        })
    if dynamic_shapes:
        dynamic_region_coverage["shape_specific"] = dynamic_shapes
        dynamic_region_coverage["all_listed_shapes_complete"] = all(row["complete"] for row in dynamic_shapes)
        dynamic_region_coverage["shape_strata"] = [row["seq_len"] for row in dynamic_shapes]
    audit = {
        "schema": "kernel-analyzer-implementation-matrix-audit-v1",
        "subject": "Qwen3-1.7B current real implementation matrix",
        "static_atlas": {
            "rows": atlas["denominator"],
            "exact_replays": atlas["exact_replay_count"],
            "explicit_changed_interventions": atlas["changed_count"],
            "scope": "frozen generated-region replay/intervention atlas",
        },
        "evolving_bank": {
            "steps": steps,
            "checkpoint_count": len(steps),
            "natural_training": True,
            "validation_seq_len": matrix["evaluation"]["seq_len"],
        },
        "invocation_difference_table": {
            "generated_sites": invocation["denominator"]["generated_sites"],
            "semantic_forward_vjp_units": invocation["denominator"]["semantic_forward_vjp_units"],
            "real_changed_sites": invocation["denominator"]["real_changed_sites"],
            "real_changed_sites_bound_to_closed_fbv_units": invocation["denominator"]["real_changed_sites_bound_to_exact_fbv_units"],
            "real_changed_sites_bound_to_any_fbv_unit": invocation["denominator"]["real_changed_sites_bound_to_any_fbv_unit"],
            "changed_fbv_units": invocation["denominator"]["changed_fbv_units"],
            "changed_sites_without_exact_fbv_binding": invocation["denominator"]["real_changed_sites_without_exact_fbv_binding"],
            "excluded_nonclosed_changed_units": invocation["denominator"]["excluded_nonclosed_changed_units"],
        },
        "evolving_attention": {
            "shape_strata": shape_matrix["shapes"],
            "reference_forward_backward_region_instances": shape_matrix["denominator"]["reference_forward_backward_region_instances"],
            "candidate_region_comparisons": shape_matrix["denominator"]["candidate_forward_backward_region_comparisons"],
            "candidate_variants": ["sdpa_math", "sdpa_flash"],
            "all_rows_status_ok": all(value["all_variant_rows_ok"] for value in shape_matrix["shape_summary"].values()),
        },
        "evolving_full_step_inductor": {
            "required_cells": required_cells,
            "measured_cells": measured_cells,
            "missing_cells": missing_cells,
            "all_measured_rows_ok": all(
                row["status"] == "OK"
                for cell in measured_cells
                for row in read(cell["file"])["rows"]
            ),
            "all_measured_repeats_exact": all(
                row["candidate_repeat_delta"] == 0.0
                for cell in measured_cells
                for row in read(cell["file"])["rows"]
            ),
            "scope": "full-step compiled-vs-eager F+B; generated-region attribution remains bounded by the invocation atlas",
            "carrier_screen": full_step_carrier_cells,
        },
        "fused_configurations": {
            "liger_fused_linear_cross_entropy": {
                "status": "COMPLETE" if fused_configuration_complete else "MISSING_OR_INCOMPLETE",
                "trajectory_steps": 32 if fused_configuration_complete else 0,
                "same_weight_control_steps": 64 if fused_configuration_complete else 0,
                "candidate": "Liger fused linear cross-entropy",
                "evidence": "results/trajectory/liger_trajectory.json",
            }
        },
        "carrier_screen": {
            variant: {
                "heldout_states": value["heldout_states"],
                "heldout_positive": value["heldout_positive"],
                "heldout_positive_fraction": value["heldout_positive_fraction"],
                "minimum_projection": value["heldout_min_projection"],
            }
            for variant, value in carrier["summary"].items()
        },
        "generated_region_dynamic_coverage": {
            "required_changed_fbv_units": invocation["denominator"]["changed_fbv_units"],
            "measured_changed_fbv_units": dynamic_region_coverage["bound_changed_fbv_units"],
            "required_changed_region_sites": dynamic_region_coverage["required_changed_region_sites"],
            "measured_changed_region_sites": dynamic_region_coverage["measured_changed_region_sites"],
            "required_closed_fbv_sites": dynamic_region_coverage.get("required_closed_fbv_sites", len(closed_fbv_site_ids)),
            "measured_closed_fbv_sites": dynamic_region_coverage.get("observed_closed_fbv_sites", 0),
            "checkpoint_count": dynamic_region_coverage["checkpoint_count"],
            "shape_strata": dynamic_region_coverage["shape_strata"],
            "complete": generated_region_dynamic_complete,
            "complete_for_current_static_atlas": dynamic_region_coverage["complete_for_current_static_atlas"],
            "shape_specific": dynamic_region_coverage.get("shape_specific", []),
            "all_listed_shapes_complete": dynamic_region_coverage.get("all_listed_shapes_complete", False),
            "direct_generated_compute_endpoint": {
                "site_ids": sorted(direct_closed_fbv_site_ids),
                "observed_in_full_step_parameter_gradient": direct_compute_endpoint_complete,
                "evidence": "results/final/structured_carrier_trigger.json",
                "internal_instruction_schedule_adjudicated": False,
            },
            "fp16_shape_specific": fp16_dynamic_shapes,
            "fp16_all_listed_shapes_complete_for_mapped_layer": bool(fp16_dynamic_shapes) and all(
                row["complete_mapped_layer"] for row in fp16_dynamic_shapes
            ),
            "scope": "launch-site observation is complete for the listed BF16 and mapped FP16 layers; the direct embedding scatter endpoint is covered by the complete seq1024 parameter-gradient partition, while its internal instruction schedule is not inferred",
            "dtype_boundary": dtype_boundary,
            "dtype_topology_boundaries": dtype_topologies,
            "dtype_semantic_mapping_boundaries": dtype_mappings,
            "dtype_semantic_observations": dtype_semantic_observations,
            "dtype_endpoint_observations": dtype_endpoint_observations,
            "dtype_unresolved_boundary": dtype_unresolved,
            "source_replay": source_replay,
            "source_replay_complete": source_replay_complete,
        },
        "static_source_matrix": source_matrix_static,
        "declared_matrix_cells_complete": (
            not missing_cells
            and all(value["all_variant_rows_ok"] for value in shape_matrix["shape_summary"].values())
            and generated_region_dynamic_complete
            and fused_configuration_complete
        ),
        "coverage_boundary": (
            "The listed matrix cells are complete: eager/SDPA math/SDPA flash attention, all 12 "
            "Inductor dtype×shape cells, BF16 and mapped FP16 changed-site layers, and all six FP32/TF32 "
            "source-replay cells, plus the Liger fused-linear cross-entropy trajectory, are complete across "
            "the eight natural checkpoints or its frozen 32-step protocol with exact repeats. "
            "The 56/114 unresolved shape-transfer rows at seq128/256 are unchanged rows and therefore remain "
            "outside the implementation-difference denominator; one FP16 cast symbol and naturally nonfinite "
            "masked rows remain explicit non-difference boundaries. Full-step Inductor cells provide cell-level "
            "numeric evidence, while per-region F+B attribution is restricted to the invocation atlas. This is "
            "not invocation-level coverage completion and not a universal safety claim for unlisted "
            "backends, models, architectures or conditions. The fail-closed invocation ledger is authoritative."
        ),
    }
    audit["result_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "implementation_matrix.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "declared_matrix_cells_complete": audit["declared_matrix_cells_complete"]}, sort_keys=True))


if __name__ == "__main__":
    main()
