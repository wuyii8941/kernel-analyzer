import unittest
import json
import hashlib
import gzip
from pathlib import Path

import torch

from scripts.check import main
from scripts.evolving_triton_observation import (
    _dtype_mapping_phase,
    remap_campaign_to_warmed_symbols,
)
from scripts.run_qwen_heldout_endpoints import sample_indices
from scripts.run_qwen_current_triton_references import tensor_digest


class FinalResultsTest(unittest.TestCase):
    def test_tensor_digest_accepts_scalar_endpoint(self):
        self.assertEqual(tensor_digest(torch.tensor(1.25)), tensor_digest(torch.tensor(1.25)))
        self.assertNotEqual(tensor_digest(torch.tensor(1.25)), tensor_digest(torch.tensor(1.5)))

    def test_runtime_symbol_remap_preserves_reference_dispatch_identity(self):
        rows, unmatched = remap_campaign_to_warmed_symbols(
            [{
                "symbol": "triton_poi_fused_example_6",
                "reference_symbol": "triton_poi_fused_reference_8",
            }],
            ["triton_poi_fused_example_9"],
        )
        self.assertEqual(rows[0]["symbol"], "triton_poi_fused_example_9")
        self.assertEqual(rows[0]["reference_symbol"], "triton_poi_fused_reference_8")
        self.assertEqual(unmatched, [])

    def test_heldout_coordinate_sampler_terminates_on_composite_sizes(self):
        for size in (96, 128, 2048, 151936):
            values = sample_indices("model.example.weight", 64, size)
            self.assertEqual(len(values), 64)
            self.assertEqual(len(set(values)), 64)
            self.assertTrue(all(0 <= value < size for value in values))

    def test_current_triton_reference_plan_is_fail_closed(self):
        root = Path(__file__).resolve().parents[1]
        with gzip.open(
            root / "results/coverage/qwen_current_triton_reference_campaign.json.gz",
            "rt",
        ) as handle:
            campaign = json.load(handle)
        denominator = campaign["denominator"]
        self.assertEqual(denominator["triton_invocations"], 686)
        self.assertEqual(
            denominator["reference_adapter_exact"]
            + denominator["reference_adapter_unresolved"],
            686,
        )
        self.assertEqual(denominator["reference_adapter_exact"], 453)
        self.assertFalse(campaign["gates"]["all_reference_adapters_exact"])
        self.assertFalse(campaign["gates"]["heldout_values_observed"])
        self.assertTrue(all(
            not row["evidence"]["region_id_or_symbol_used_for_transfer"]
            for row in campaign["rows"]
            if row["adapter_status"] == "EXACT_SEMANTIC_AND_POINTER_ABI_ADAPTER_TRANSFER"
        ))
        statuses_by_symbol = {}
        for row in campaign["rows"]:
            statuses_by_symbol.setdefault(row["symbol"], set()).add(row["adapter_status"])
            if row["adapter_status"] == "UNRESOLVED_CURRENT_REFERENCE_ADAPTER":
                self.assertTrue(row["boundary_capture_mode"].startswith("RUNTIME_SKIP_"))
        self.assertTrue(any(len(statuses) > 1 for statuses in statuses_by_symbol.values()))

    def test_exhaustive_gap_matrix_has_a_closure_action_for_every_open_gate(self):
        import gzip

        root = Path(__file__).resolve().parents[1]
        with gzip.open(root / "results/coverage/gap_matrix.json.gz", "rt") as handle:
            matrix = json.load(handle)
        self.assertEqual(
            matrix["denominator"], 9269 + 56411 + 25582 + 8223 + 12044
        )
        self.assertEqual(len(matrix["rows"]), matrix["denominator"])
        self.assertEqual(
            len({row["row_id"] for row in matrix["rows"]}), matrix["denominator"]
        )
        self.assertTrue(all(
            row["complete"] == (not row["missing_gates"])
            for row in matrix["rows"]
        ))
        self.assertEqual(
            matrix["architecture_summary"]["qwen"]["fully_complete_invocations"],
            311,
        )
        self.assertTrue(all(
            gap["action"]
            for row in matrix["rows"]
            for gap in row["missing_gates"]
        ))
        self.assertEqual(
            matrix["architecture_summary"]["qwen"]["invocations"], 9269
        )
        self.assertEqual(
            matrix["architecture_summary"]["mamba"]["invocations"], 56411
        )
        self.assertEqual(
            matrix["architecture_summary"]["moe"]["invocations"], 25582
        )

    def test_package(self) -> None:
        main()

    def test_next_round_controls_are_separated(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        flash = json.loads((root / "flash_control.json").read_text())
        atlas = json.loads((root / "implementation_atlas.json").read_text())
        self.assertEqual(flash["kind"], "PAPER_REFERENCE_REPRODUCTION")
        self.assertEqual(flash["schema"], "flash-paper-control-v2")
        self.assertIn("real_sdpa", flash)
        self.assertTrue(flash["positive_control"]["closed_f_b"])
        self.assertEqual(flash["positive_control"]["v_only_live_weight"]["heldout_positive"], 31)
        self.assertEqual(flash["positive_control"]["v_only_live_weight"]["heldout_negative_repair"], 31)
        self.assertLess(flash["fb_closure"]["max_backward_abs"], 1e-5)
        self.assertEqual(atlas["denominator"], len(atlas["rows"]))
        self.assertGreater(atlas["exact_replay_count"], 0)
        self.assertGreater(atlas["changed_count"], 0)

    def test_natural_evolving_bank_and_attention_matrix(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        bank = json.loads((root / "natural_bank.json").read_text())
        self.assertTrue(bank["natural_training"])
        self.assertEqual([row["step"] for row in bank["checkpoints"]], [0, 1, 2, 4, 8, 16, 32, 64])
        matrix = json.loads((root / "checkpoint_matrix.json").read_text())
        self.assertEqual(len(matrix["rows"]), 8)
        self.assertTrue(all(row["reference_forward_regions"] == 28 for row in matrix["rows"]))
        self.assertTrue(all(row["reference_backward_regions"] == 28 for row in matrix["rows"]))
        self.assertTrue(all(value.get("status", "OK") == "OK" for row in matrix["rows"] for value in row["variants"].values()))

    def test_carrier_screen_is_not_a_persistent_case(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        carrier = json.loads((root / "checkpoint_carrier.json").read_text())
        for value in carrier["summary"].values():
            self.assertEqual(value["heldout_states"], 7)
            self.assertEqual(value["heldout_positive"], 5)
            self.assertLess(value["heldout_positive"], value["heldout_states"])
            self.assertLess(value["heldout_min_projection"], 0.0)

    def test_inductor_full_step_boundary(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        inductor = json.loads((root / "checkpoint_inductor.json").read_text())
        self.assertEqual(len(inductor["rows"]), 8)
        self.assertTrue(all(row["status"] == "OK" for row in inductor["rows"]))
        self.assertTrue(all(row["candidate_repeat_delta"] == 0.0 for row in inductor["rows"]))
        audit = json.loads((root / "implementation_matrix.json").read_text())
        self.assertTrue(audit["declared_matrix_cells_complete"])
        self.assertNotIn("matrix_exhausted", audit)
        self.assertEqual(audit["static_atlas"]["rows"], 1446)

    def test_invocation_atlas_filters_real_changes_and_preserves_unresolved(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        atlas = json.loads((root / "invocation_atlas.json").read_text())
        self.assertEqual(atlas["denominator"]["generated_sites"], 1447)
        self.assertEqual(atlas["denominator"]["semantic_forward_vjp_units"], 3491)
        self.assertEqual(atlas["denominator"]["real_changed_sites"], 507)
        self.assertEqual(atlas["denominator"]["real_changed_sites_bound_to_exact_fbv_units"], 507)
        self.assertEqual(atlas["denominator"]["real_changed_sites_without_exact_fbv_binding"], 0)
        self.assertEqual(atlas["denominator"]["real_changed_sites_bound_to_any_fbv_unit"], 507)
        self.assertEqual(atlas["denominator"]["real_changed_sites_without_any_fbv_binding"], 0)
        self.assertEqual(atlas["denominator"]["changed_fbv_units"], 949)
        self.assertEqual(atlas["denominator"]["excluded_nonclosed_changed_units"], 122)
        self.assertTrue(all(row["real_implementation_change"] for row in atlas["changed_units"]))
        self.assertTrue(all(row["mechanisms"] for row in atlas["changed_units"]))
        self.assertTrue(all(row["vjp_status"] == "EXACT_ACTUAL_BACKWARD_PROGRAM" for row in atlas["changed_units"]))
        self.assertTrue(all(not row["actual_backward_node_ids"] for row in atlas["excluded_nonclosed_changed_units"]))

    def test_all_parameter_carrier_census_is_exact_and_candidate_blind(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        census = json.loads((root / "carrier_census.json").read_text())
        denominator = census["denominator"]
        self.assertEqual(denominator["changed_closed_fbv_units"], 949)
        self.assertEqual(denominator["exact_parameter_reachability_units"], 949)
        self.assertEqual(denominator["unresolved_units"], 0)
        self.assertEqual(denominator["trainable_unique_parameters"], 310)
        self.assertEqual(denominator["parameters_reached_by_any_changed_unit"], 310)
        self.assertGreater(denominator["exact_forward_saved_tensor_backward_edges"], 0)
        self.assertFalse(census["gates"]["candidate_values_used_to_select_or_classify"])
        self.assertTrue(census["gates"]["cross_phase_runtime_identity_bridge_exact"])
        self.assertTrue(census["gates"]["all_changed_units_exactly_mapped"])
        self.assertFalse(census["natural_bias_case_added"])
        self.assertFalse(census["property_claim"])

    def test_seq1024_structural_bias_factor_is_heldout_and_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        stage = json.loads((root / "backend_stage_diagnosis.json").read_text())
        for row in stage["comparisons"]["aot_eager_minus_eager"]:
            self.assertEqual(row["loss_delta"], 0.0)
            self.assertTrue(all(value["exact"] for value in row["parameters"].values()))
        heldout = json.loads((root / "inductor_config_heldout.json").read_text())
        self.assertEqual(heldout["milestones"], [0, 64, 256, 1024, 2048, 4096])
        self.assertEqual(heldout["evaluation"]["states"], 8)
        self.assertTrue(all(
            row["loss_exact"] and all(row["parameters_exact"].values())
            for row in heldout["reference_repeat_exact"]
        ))
        passed = {
            (row["variant"], row["parameter"])
            for row in heldout["summary"]["rows"] if row["heldout_gate"]
        }
        key = "model.layers.19.self_attn.k_norm.weight"
        query = "model.layers.19.self_attn.q_norm.weight"
        self.assertEqual(passed, {
            ("minimal_fusion", key),
            ("materialize_reused_intermediates", key),
        })
        self.assertFalse(any(parameter == query for _, parameter in passed))
        campaign = json.loads((root / "seq1024_reduction_campaign.json").read_text())
        self.assertEqual(campaign["denominator"]["exact_source_node_bindings"], 112)
        self.assertEqual(campaign["denominator"]["unresolved"], 0)
        intervention = json.loads((root / "long_trigger_k_sum.json").read_text())
        trigger = json.loads((root / "long_horizon_trigger.json").read_text())
        baseline = next(
            row for row in trigger["rows"]
            if row["checkpoint_step"] == 256 and row["state_id"] == 0 and row["repeat"] == 0
        )
        self.assertEqual(intervention["model_construction"],
                         "scripts.long_horizon_trigger.build_model+load_milestone")
        self.assertEqual(intervention["eager_loss"], baseline["reference_loss"])
        self.assertEqual(intervention["baseline_candidate_loss"], baseline["candidate_loss"])

    def test_structured_carrier_scan_is_complete_and_independently_confirmed(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        discovery = json.loads((root / "structured_carrier_trigger.json").read_text())
        self.assertEqual(discovery["status"], "COMPLETE")
        self.assertEqual(discovery["coverage"]["parameters"], 310)
        self.assertFalse(discovery["coverage"]["sampled_coordinate_subset"])
        self.assertTrue(discovery["coverage"]["every_coordinate_in_exactly_one_partition_block"])
        self.assertEqual(
            discovery["coverage"]["partition_coordinate_memberships"],
            discovery["coverage"]["unique_parameter_coordinates"],
        )
        self.assertEqual(discovery["trigger_count"], 120)
        confirmation = json.loads((root / "structured_carrier_confirmation.json").read_text())
        self.assertEqual(confirmation["status"], "COMPLETE")
        self.assertEqual(confirmation["family_size"], discovery["trigger_count"])
        self.assertEqual(confirmation["confirmation_evaluation"]["states"], 32)
        self.assertTrue(all(row["all_parameter_gradients_exact"] for row in confirmation["reference_repeat_checks"]))
        self.assertEqual(confirmation["confirmed_count"], 1)
        confirmed = next(row for row in confirmation["rows"] if row["confirmed"])
        self.assertEqual(confirmed["trigger"]["parameter"], "model.layers.23.self_attn.q_proj.weight")
        self.assertEqual(confirmed["trigger"]["row_start"], 1152)
        self.assertEqual(confirmed["trigger"]["column_start"], 1664)
        self.assertTrue(confirmed["holm_rejected"])
        self.assertFalse(confirmation["natural_case_added"])

    def test_l23_composite_carrier_is_closed_without_becoming_a_local_case(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        summary = json.loads((root / "l23_go_summary.json").read_text())
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["checkpoints"], [64, 256, 1024, 2048, 4096])
        self.assertEqual(summary["states"], list(range(8, 40)))
        self.assertTrue(summary["validation"]["all_reference_bmm_replays_bitwise_exact"])
        self.assertEqual(summary["validation"]["max_candidate_restoration_sham_abs"], 0.0)
        self.assertEqual(summary["validation"]["max_shapley_closure_abs"], 0.0)
        ratios = summary["ratios"]
        self.assertGreater(ratios["s_shapley_over_total"], 0.98)
        self.assertGreater(ratios["u_shapley_over_total"], 0.80)
        self.assertGreater(ratios["d_shapley_over_total"], 0.70)
        self.assertGreater(ratios["go_residual_over_total"], 0.30)
        residual = summary["metric_summary"]["reference_s_reference_k_residual_projection"]
        self.assertLess(residual["state_cluster_bootstrap_95"][0], 0.0)
        self.assertGreater(residual["state_cluster_bootstrap_95"][1], 0.0)
        go_residual = summary["metric_summary"]["reference_go_residual_projection"]
        self.assertGreater(go_residual["state_cluster_bootstrap_95"][0], 0.0)
        for step in (64, 256, 1024, 2048, 4096):
            raw = json.loads((root / f"l23_go_step{step}.json").read_text())
            self.assertEqual(len(raw["rows"]), 32)
            self.assertTrue(all(row["reference_bmm_replay_matches_eager_query_gradient"] for row in raw["rows"]))
            self.assertTrue(all(row["reference_softmax_vjp_replay_matches_eager_s"] for row in raw["rows"]))
            self.assertTrue(all(row["reference_u_bmm_replay_matches_eager_u"] for row in raw["rows"]))
            self.assertTrue(all(row["reference_o_proj_input_vjp_replay_matches_eager_d"] for row in raw["rows"]))
        live = json.loads((root / "l23_attention_live_weight.json").read_text())
        self.assertEqual(live["status"], "COMPLETE")
        self.assertEqual(len(live["records"]), 32)
        projections = [row["fp32_master_projection"] for row in live["records"]]
        self.assertTrue(all(right > left for left, right in zip(projections, projections[1:])))
        self.assertEqual(live["records"][-1]["bf16_materialized_nonzero"], 1007)
        packaged = json.loads((root / "summary.json").read_text())
        self.assertTrue(packaged["causal"]["l23_qproj_composite"]["natural_complete_case_added"])
        self.assertEqual(packaged["causal"]["project_natural_complete_cases"], 3)
        self.assertEqual(packaged["causal"]["project_fully_single_kernel_attributed_cases"], 2)
        self.assertTrue(packaged["claims"]["l23_qproj_is_natural_complete_f_b_directional_carrier_case"])
        self.assertFalse(packaged["claims"]["l23_qproj_is_fully_single_kernel_attributed"])

    def test_l23_downstream_decomposition_localizes_forward_logits(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"

        go = json.loads((root / "l23_go_path_summary.json").read_text())
        self.assertEqual(go["status"], "COMPLETE")
        self.assertTrue(go["validation"]["all_reference_go_replays_bitwise_exact"])
        self.assertEqual(go["validation"]["max_candidate_restoration_sham_abs"], 0.0)
        self.assertGreater(go["ratios"]["r_shapley_over_original_total"], 0.60)
        self.assertGreater(go["metric_summary"]["r_shapley_removal_projection"]["state_cluster_bootstrap_95"][0], 0.0)

        l26 = json.loads((root / "l23_residual_l26_summary.json").read_text())
        attention_ci = l26["metric_summary"]["a_shapley_removal_projection"]["state_cluster_bootstrap_95"]
        self.assertEqual(l26["target_layer"], 26)
        self.assertLess(attention_ci[1], 0.0)

        l27 = json.loads((root / "l23_go_l27_summary.json").read_text())
        mlp_ci = l27["metric_summary"]["m_shapley_removal_projection"]["state_cluster_bootstrap_95"]
        self.assertEqual(l27["target_layer"], 27)
        self.assertLess(mlp_ci[1], 0.0)

        terminal = json.loads((root / "l23_terminal_summary.json").read_text())
        self.assertTrue(terminal["validation"]["all_analytic_logits_vjp_replays_bitwise_exact"])
        self.assertTrue(terminal["validation"]["all_reference_lm_head_input_vjp_replays_bitwise_exact"])
        self.assertEqual(terminal["ratios_over_original_total"]["lm_head_mm"], 0.0)
        self.assertGreater(terminal["ratios_over_original_total"]["upstream_logits"], 0.75)
        logits_ci = terminal["metric_summary"]["upstream_logits_removal_projection"]["state_cluster_bootstrap_95"]
        self.assertGreater(logits_ci[0], 0.0)

        final_norm = json.loads((root / "l23_final_norm_summary.json").read_text())
        self.assertTrue(final_norm["validation"]["all_eager_forward_replays_bitwise_exact"])
        joint_ci = final_norm["metric_summary"]["joint_final_norm_removal_projection"]["state_cluster_bootstrap_95"]
        self.assertLess(joint_ci[0], 0.0)
        self.assertGreater(joint_ci[1], 0.0)

    def test_shape_stratified_dynamic_matrix(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        matrix = json.loads((root / "checkpoint_matrix_shapes.json").read_text())
        self.assertEqual(matrix["shapes"], [64, 128, 256])
        self.assertEqual(matrix["denominator"]["candidate_forward_backward_region_comparisons"], 2688)
        self.assertTrue(all(value["all_variant_rows_ok"] for value in matrix["shape_summary"].values()))
        for seq in (64, 256):
            carrier = json.loads((root / f"checkpoint_carrier_seq{seq}.json").read_text())
            self.assertEqual(set(carrier["summary"]), {"sdpa_math", "sdpa_flash"})
            self.assertTrue(all(value["heldout_states"] == 7 for value in carrier["summary"].values()))

    def test_inductor_dtype_shape_audit_remains_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        audit = json.loads((root / "implementation_matrix.json").read_text())
        self.assertTrue(audit["declared_matrix_cells_complete"])
        self.assertEqual(len(audit["evolving_full_step_inductor"]["missing_cells"]), 0)
        self.assertEqual(len(audit["evolving_full_step_inductor"]["measured_cells"]), 12)
        self.assertTrue(audit["evolving_full_step_inductor"]["all_measured_rows_ok"])
        self.assertTrue(audit["evolving_full_step_inductor"]["all_measured_repeats_exact"])
        carrier_cells = audit["evolving_full_step_inductor"]["carrier_screen"]
        self.assertEqual(len(carrier_cells), 12)
        self.assertTrue(all(row["heldout_states"] == 7 for row in carrier_cells))
        self.assertTrue(all(not row["persistent_positive"] for row in carrier_cells))
        self.assertTrue(all(not row["persistent_negative"] for row in carrier_cells))
        self.assertTrue(audit["generated_region_dynamic_coverage"]["complete"])

    def test_evolving_triton_changed_sites_are_directly_observed(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        dynamic = json.loads((root / "evolving_triton_seq64.json").read_text())
        self.assertEqual(dynamic["checkpoint_steps"], [0, 1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(dynamic["denominator"]["changed_generated_region_sites"], 506)
        self.assertEqual(dynamic["denominator"]["changed_sites_observed_at_all_steps_and_repeats"], 506)
        self.assertTrue(dynamic["gates"]["all_changed_sites_present_at_all_steps_and_repeats"])
        self.assertTrue(dynamic["gates"]["all_worker_observations_stable"])
        audit = json.loads((root / "implementation_matrix.json").read_text())
        self.assertEqual(audit["generated_region_dynamic_coverage"]["measured_changed_region_sites"], 506)
        self.assertEqual(audit["generated_region_dynamic_coverage"]["required_closed_fbv_sites"], 507)
        self.assertEqual(audit["generated_region_dynamic_coverage"]["measured_closed_fbv_sites"], 507)
        self.assertTrue(audit["generated_region_dynamic_coverage"]["direct_generated_compute_endpoint"]["observed_in_full_step_parameter_gradient"])
        self.assertEqual(audit["generated_region_dynamic_coverage"]["checkpoint_count"], 8)
        self.assertEqual(audit["generated_region_dynamic_coverage"]["shape_strata"], [64, 128, 256])
        self.assertTrue(audit["declared_matrix_cells_complete"])

    def test_shape_specific_dynamic_generated_coverage(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        expected = {64: 506, 128: 452, 256: 450}
        audit = json.loads((root / "implementation_matrix.json").read_text())
        rows = audit["generated_region_dynamic_coverage"]["shape_specific"]
        self.assertEqual({row["seq_len"] for row in rows}, set(expected))
        for row in rows:
            self.assertEqual(row["required_changed_region_sites"], expected[row["seq_len"]])
            self.assertEqual(row["measured_changed_region_sites"], expected[row["seq_len"]])
            self.assertTrue(row["complete"])
        self.assertEqual(rows[0]["required_closed_fbv_sites"], 506)
        self.assertEqual(rows[0]["observed_closed_fbv_sites"], 506)
        self.assertTrue(audit["generated_region_dynamic_coverage"]["all_listed_shapes_complete"])
        self.assertTrue(audit["generated_region_dynamic_coverage"]["complete"])

    def test_non_bf16_dynamic_boundary_is_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        boundary = json.loads((root / "dtype_dynamic_boundary.json").read_text())
        self.assertEqual(boundary["dtype_probe"]["status"], "INVALID_REFERENCE_CAMPAIGN_DTYPE")
        self.assertFalse(boundary["dtype_probe"]["candidate_values_used_to_select_regions"])
        self.assertFalse(boundary["claim_boundary"]["property_generalization_allowed"])

    def test_completion_audit_does_not_overclaim(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        audit = json.loads((root / "completion_audit.json").read_text())
        statuses = {row["id"]: row["status"] for row in audit["requirements"]}
        self.assertEqual(statuses["flash_positive_control"], "COMPLETE")
        self.assertEqual(statuses["natural_training_checkpoint_bank"], "COMPLETE")
        self.assertEqual(statuses["qwen_fail_closed_invocation_coverage"], "PARTIAL_FAIL_CLOSED")
        self.assertEqual(statuses["property_generalization_conclusion"], "NOT_YET_ALLOWED")
        source_gate = next(row for row in audit["requirements"] if row["id"] == "source_mapping_extension_numeric_gate")
        self.assertTrue(source_gate["checks"]["source_phase_audit_all_cells_resolved"])
        bank_gate = next(row for row in audit["requirements"] if row["id"] == "natural_training_checkpoint_bank")
        self.assertEqual(bank_gate["checks"]["distinct_parameter_hashes"], 8)
        self.assertEqual(bank_gate["checks"]["distinct_file_hashes"], 8)
        self.assertEqual(bank_gate["checks"]["optimizer"], "AdamW")
        self.assertEqual(bank_gate["checks"]["optimizer_steps"], 64)

    def test_source_mapping_replay_gate_is_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        mapping = json.loads((root / "source_mapping_progress.json").read_text())
        gate = json.loads((root / "source_replay_gate.json").read_text())
        self.assertEqual(gate["status"], "COMPLETE")
        self.assertEqual(
            [(row["seq_len"], row["mapped_invocations"]) for row in mapping["cells"]],
            [(128, 670), (256, 724)],
        )
        self.assertEqual(mapping["cells"][1]["unresolved_invocations"], 0)
        self.assertEqual(mapping["cells"][1]["unresolved_symbols"], 0)
        self.assertTrue(all(row["numeric_validation"] == "COMPLETE" for row in mapping["cells"]))
        self.assertFalse(gate["natural_bias_case_added"])
        self.assertFalse(gate["property_claim"])

    def test_static_source_matrix_is_closed_but_not_numeric(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        matrix = json.loads((root / "source_matrix_static.json").read_text())
        self.assertFalse(matrix["candidate_values_used_to_select_or_classify"])
        self.assertEqual(len(matrix["cells"]), 6)
        self.assertTrue(all(row["unresolved_invocations"] == 0 for row in matrix["cells"]))
        self.assertTrue(all(row["mapped_invocations"] == row["runtime_invocations"] for row in matrix["cells"]))
        for row in matrix["cells"]:
            mapping = json.loads((root / row["mapping_file"]).read_text())
            self.assertFalse(mapping["candidate_values_used_to_select_or_classify"])
            self.assertEqual(mapping["denominator"]["unresolved_invocations"], 0)
            self.assertEqual(mapping["denominator"]["mapped_invocations"], row["runtime_invocations"])
            self.assertEqual(hashlib.sha256((root / row["mapping_file"]).read_bytes()).hexdigest(), row["mapping_sha256"])
        self.assertEqual(matrix["numeric_replay"], "COMPLETE")
        self.assertFalse(matrix["natural_bias_case_added"])
        self.assertFalse(matrix["property_claim"])

    def test_source_replay_schedule_covers_the_full_pending_cross_product(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        schedule = json.loads((root / "source_replay_schedule.json").read_text())
        self.assertEqual(schedule["cell_count"], 6)
        self.assertEqual(schedule["checkpoint_steps"], [0, 1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(schedule["planned_runs"], 48)
        self.assertEqual(schedule["repeat_count"], 2)
        self.assertTrue(all(row["status"] == "COMPLETE" for row in schedule["rows"]))
        self.assertTrue(all(row["expected_invocations"] in (670, 724) for row in schedule["rows"]))

    def test_source_replay_matrix_merge_boundary_is_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        matrix = json.loads((root / "source_replay_matrix.json").read_text())
        self.assertEqual(matrix["cell_count"], 6)
        self.assertEqual(matrix["planned_runs"], 48)
        self.assertEqual(matrix["complete_cells"], 6)
        self.assertEqual(matrix["pending_cells"], 0)
        self.assertEqual(matrix["numeric_replay"], "COMPLETE")
        self.assertFalse(matrix["candidate_values_used_to_select_or_classify"])
        self.assertFalse(matrix["natural_bias_case_added"])
        self.assertFalse(matrix["property_claim"])
        self.assertTrue(all(row["status"] == "COMPLETE" for row in matrix["cells"]))

    def test_missing_case_diagnosis_separates_cancellation_from_open_matrix(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        diagnosis = json.loads((root / "missing_case_diagnosis.json").read_text())
        local = diagnosis["measured_links"]["local_difference"]
        self.assertEqual(local["changed_closed_fbv_units"], 949)
        self.assertEqual(local["real_changed_sites"], 507)
        self.assertEqual(diagnosis["measured_links"]["local_to_carrier_intervention"]["persistent_direction_arms"], 0)
        self.assertEqual(diagnosis["open_links"]["static_source_mapping_cells"], 6)
        self.assertTrue(diagnosis["open_links"]["static_source_mapping_all_mapped"])
        self.assertEqual(diagnosis["open_links"]["real_replay_runs_planned"], 48)
        self.assertEqual(diagnosis["open_links"]["real_replay_status"], "COMPLETE")
        self.assertEqual(
            diagnosis["open_links"]["nontriton_candidates_pending_live_full_coordinate"], 0
        )
        self.assertEqual(
            diagnosis["open_links"]["nontriton_t1_positive_pending_case_gates"], 1
        )
        self.assertTrue(diagnosis["natural_bias_case_added"])
        self.assertFalse(diagnosis["property_claim"])

    def test_dtype_mapping_phase_uses_reference_phase_before_name_fallback(self) -> None:
        self.assertEqual(
            _dtype_mapping_phase("generated_symbol", "canonical_symbol", {"canonical_symbol": {"BACKWARD"}}),
            "BACKWARD",
        )
        self.assertEqual(
            _dtype_mapping_phase("foo", "forkcert:query-rotary-forward-variant", {}),
            "FORWARD",
        )
        self.assertEqual(
            _dtype_mapping_phase("foo", "forkcert:rms-weight-split-full", {}),
            "BACKWARD",
        )
        self.assertEqual(
            _dtype_mapping_phase("foo", "forkcert:loss-softmax-seq256-partial", {}),
            "FORWARD",
        )

    def test_source_phase_audit_is_closed_but_not_numeric(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        audit = json.loads((root / "source_phase_audit.json").read_text())
        self.assertEqual(len(audit["cells"]), 6)
        self.assertTrue(audit["all_cells_phase_resolved"])
        self.assertTrue(all(row["all_phase_resolved"] for row in audit["cells"]))
        self.assertEqual(audit["numeric_replay"], "COMPLETE")
        self.assertFalse(audit["candidate_values_used_to_select_or_classify"])

    def test_matrix_cross_product_has_no_static_gap_but_replay_is_pending(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        audit = json.loads((root / "matrix_cross_product_audit.json").read_text())
        self.assertTrue(audit["matrix_has_no_static_gap"])
        self.assertTrue(audit["full_step_inductor"]["complete"])
        self.assertTrue(audit["source_replay"]["complete"])
        self.assertEqual(audit["source_replay"]["expected_count"], 48)
        self.assertEqual(audit["numeric_source_replay"], "COMPLETE")
        self.assertFalse(audit["natural_bias_case_added"])

    def test_gpu_preflight_fail_closes_without_workaround(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        preflight = json.loads((root / "gpu_preflight.json").read_text())
        self.assertEqual(preflight["status"], "GPU_DEVICE_NODE_UNAVAILABLE")
        self.assertFalse(preflight["replay_authorized"])
        self.assertFalse(preflight["device_nodes_present"])
        self.assertTrue(preflight["nvidia_proc_version_present"])
        self.assertFalse(preflight["torch"]["cuda_available"])
        self.assertIn("never creates device nodes", preflight["boundary"])

    def test_objective_audit_keeps_unfinished_requirements_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        audit = json.loads((root / "objective_audit.json").read_text())
        statuses = {row["id"]: row["status"] for row in audit["requirements"]}
        self.assertEqual(statuses["flash_positive_control"], "COMPLETE")
        self.assertEqual(statuses["natural_training_checkpoint_bank"], "COMPLETE")
        self.assertEqual(statuses["invocation_level_real_difference_table"], "COMPLETE_FOR_FROZEN_SCOPE")
        self.assertEqual(statuses["evolving_checkpoint_numeric_replay"], "COMPLETE_FOR_SIX_CELL_SOURCE_REPLAY")
        self.assertEqual(statuses["qwen_fail_closed_invocation_coverage"], "PARTIAL_FAIL_CLOSED")
        self.assertEqual(audit["conclusion"]["strict_flash_style_cases"], 6)
        self.assertEqual(audit["conclusion"]["strict_root_arithmetic_op_cases"], 4)
        self.assertEqual(audit["conclusion"]["strict_semantic_region_cases"], 2)
        self.assertEqual(audit["conclusion"]["unresolved_composite_carrier_cases"], 0)
        self.assertEqual(audit["conclusion"]["historical_cases_rejected_by_direction_gate"], 0)
        self.assertEqual(audit["conclusion"]["cross_state_concrete_mechanism_passes"], 2)
        self.assertEqual(audit["conclusion"]["fully_closed_model_shape_cells"], 0)
        self.assertFalse(audit["conclusion"]["property_claim_allowed"])

    def test_single_fail_closed_invocation_coverage_ledger(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with gzip.open(root / "results/coverage/qwen_invocation_ledger.json.gz", "rt") as handle:
            ledger = json.load(handle)
        gaps = json.loads((root / "results/coverage/qwen_gap_audit.json").read_text())
        self.assertEqual(ledger["status"], "PARTIAL_FAIL_CLOSED")
        self.assertEqual(len(ledger["rows"]), 9269)
        self.assertEqual(len({row["row_id"] for row in ledger["rows"]}), 9269)
        self.assertEqual(ledger["summary"]["mathematical_local_derivation_complete"], 9269)
        self.assertEqual(len(ledger["mathematical_templates"]), 54)
        template_ids = {row["template_id"] for row in ledger["mathematical_templates"]}
        self.assertTrue(all(
            row["mathematical_fb"]["template_id"] in template_ids
            for row in ledger["rows"]
        ))
        self.assertEqual(ledger["summary"]["exact_eager_aot_fb_origin"], 9269)
        self.assertEqual(ledger["summary"]["unresolved_eager_aot_fb_origin"], 0)
        self.assertEqual(ledger["summary"]["legacy_changed_nonclosed_reclassified"], 122)
        self.assertTrue(ledger["gates"]["all_eager_aot_fb_origins_exact"])
        self.assertTrue(ledger["gates"]["all_candidate_region_bindings_exact"])
        self.assertFalse(ledger["gates"]["all_invocations_numerically_measured"])
        self.assertFalse(ledger["gates"]["all_invocations_have_bias_verdict"])
        self.assertNotIn("matrix_exhausted", ledger["gates"])
        self.assertEqual(gaps["eager_aot"]["unresolved"], 0)
        self.assertEqual(gaps["eager_aot"]["legacy_unresolved"], 1335)
        self.assertEqual(gaps["eager_aot"]["exact_or_closed_semantic_region"], 9269)
        program_bridge = gaps["candidate_binding"][
            "proof_tagged_inductor_program_bridge"
        ]
        self.assertEqual(program_bridge["canonical_aot_nodes"], 8985)
        self.assertEqual(program_bridge["candidate_post_aot_nodes"], 9531)
        self.assertEqual(program_bridge["candidate_proof_tags_observed"], 9531)
        candidate_audit = gaps["candidate_binding"][
            "original_unresolved_static_source_node_audit"
        ]
        self.assertEqual(candidate_audit["denominator"], 1476)
        self.assertEqual(candidate_audit["phase_qualified_generated_source_node_hits"], 0)
        self.assertEqual(candidate_audit["supplemental_exact_recoveries_from_original_unresolved"], 29)
        self.assertEqual(candidate_audit["remaining_unresolved"], 1447)
        self.assertEqual(
            gaps["changed_nonclosed_reclassification"]["remaining_unresolved"], 0
        )
        self.assertEqual(
            gaps["changed_nonclosed_reclassification"]["semantics_counts"],
            {
                "EXACT_ELIDED_DUPLICATE_VJP": 56,
                "EXACT_ELIDED_IDENTITY_VJP": 56,
                "EXACT_EMPTY_VJP_NO_FLOATING_POINT_DIFFERENTIABLE_INPUT": 10,
            },
        )

    def test_mamba_and_moe_full_fb_invocation_atlases(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "mamba": (56411, 23013, 33398, 43, 24),
            "moe": (25582, 7615, 17967, 63, 51),
        }
        for architecture, counts in expected.items():
            path = root / f"results/coverage/{architecture}_invocation_ledger.json.gz"
            with gzip.open(path, "rt") as handle:
                ledger = json.load(handle)
            total, forward, backward, overloads, observer_detaches = counts
            self.assertEqual(ledger["status"], "PARTIAL_FAIL_CLOSED")
            self.assertEqual(len(ledger["rows"]), total)
            self.assertEqual(len({row["row_id"] for row in ledger["rows"]}), total)
            self.assertEqual(ledger["summary"]["actual_invocations"], total)
            self.assertEqual(ledger["summary"]["forward_invocations"], forward)
            self.assertEqual(ledger["summary"]["backward_invocations"], backward)
            self.assertEqual(ledger["summary"]["unique_overloads"], overloads)
            self.assertEqual(
                ledger["summary"]["local_map_adjoint_and_argument_binding_complete"], total
            )
            self.assertEqual(
                ledger["summary"]["fb_origin_or_explicit_auxiliary_complete"], total
            )
            self.assertEqual(ledger["summary"]["unresolved_fb_origin"], 0)
            self.assertTrue(ledger["gates"]["every_actual_invocation_in_exactly_one_row"])
            self.assertTrue(ledger["gates"]["all_local_maps_and_adjoints_declared"])
            self.assertTrue(ledger["gates"]["all_fb_origins_or_explicit_auxiliaries_complete"])
            self.assertFalse(ledger["gates"]["all_candidate_region_bindings_exact"])
            self.assertFalse(ledger["gates"]["all_invocations_numerically_measured"])
            self.assertFalse(ledger["gates"]["all_invocations_have_bias_verdict"])
            instrumentation = ledger["instrumentation_audit"]
            self.assertEqual(
                instrumentation["observer_induced_extra_invocations_excluded"],
                observer_detaches,
            )
            self.assertEqual(
                instrumentation["observer_induced_extra_overloads"],
                {"aten.detach.default": observer_detaches},
            )
            self.assertTrue(instrumentation["all_nonextra_invocations_exactly_aligned"])
            self.assertTrue(instrumentation["baseline_vs_weak_loss_and_gradient_exact"])
            self.assertTrue(instrumentation["baseline_vs_strong_loss_and_gradient_exact"])

        overview = json.loads((root / "results/coverage/summary.json").read_text())
        self.assertTrue(overview["global_gates"]["execution_and_origin_accounting_complete"])
        self.assertFalse(overview["global_gates"]["analytic_fb_proof_complete"])
        self.assertFalse(overview["global_gates"]["candidate_fb_binding_complete"])
        self.assertFalse(overview["global_gates"]["valid_triton_numerical_oracle_complete"])
        self.assertFalse(overview["global_gates"]["same_dtype_optimization_oracle_complete"])
        self.assertFalse(overview["global_gates"]["property_induction_allowed"])

        completion = json.loads(
            (root / "results/coverage/completion_audit.json").read_text()
        )
        self.assertEqual(completion["status"], "COMPLETE_REQUESTED_INFRASTRUCTURE")
        self.assertTrue(completion["all_requirements_complete"])
        self.assertEqual(completion["scientific_coverage_status"], "PARTIAL_FAIL_CLOSED")

    def test_priority_carrier_chains_are_exact_but_not_bias_verdicts(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        chains = json.loads((root / "priority_carrier_chains.json").read_text())
        self.assertEqual(chains["chain_count"], 28)
        self.assertEqual(chains["one_state_weight_final_exact_count"], 28)
        self.assertFalse(chains["candidate_values_used_to_select_or_classify"])
        self.assertTrue(all(row["one_state_dot_metric"] for row in chains["rows"]))
        self.assertTrue(all(row["evolving_carrier_replay"] == "COMPLETE" for row in chains["rows"]))
        self.assertFalse(chains["natural_bias_case_added"])
        self.assertFalse(chains["property_claim"])

    def test_region_intervention_pilot_is_causal_but_not_a_bias_case(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        pilot = json.loads((root / "region_intervention_pilot.json").read_text())
        self.assertEqual(pilot["checkpoint_steps"], [0, 1])
        self.assertTrue(pilot["candidate_blind"])
        self.assertTrue(pilot["gates"]["all_requested_regions_observed"])
        self.assertFalse(pilot["gates"]["natural_bias_case_added"])
        projections = pilot["fixed_projection_values"]
        self.assertTrue(projections)
        self.assertTrue(all(abs(value) < pilot["fixed_projection_threshold"] for value in projections))
        self.assertEqual(pilot["projection_signs"], [])

    def test_region_intervention_batch_is_cross_checkpoint_screen_only(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        batch = json.loads((root / "region_intervention_batch.json").read_text())
        self.assertEqual(batch["checkpoint_steps"], [0, 1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(batch["arm_count"], 8)
        self.assertTrue(batch["candidate_blind"])
        self.assertTrue(batch["gates"]["all_repeats_match"])
        self.assertFalse(batch["gates"]["natural_bias_case_added"])
        self.assertFalse(batch["gates"]["property_claim"])
        self.assertTrue(all(not row["persistent_direction"] for row in batch["arms"]))

    def test_fp16_dynamic_layer_and_fp32_topology_boundary_are_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        boundary = json.loads((root / "dtype_dynamic_boundary.json").read_text())
        probe = boundary["fp16_dynamic_probe"]
        self.assertEqual(probe["dtype"], "fp16")
        self.assertEqual(probe["changed_sites_observed"], {"seq64": "506/506", "seq128": "452/452", "seq256": "450/450"})
        self.assertTrue(probe["all_observation_repeats_stable"])
        self.assertFalse(boundary["claim_boundary"]["natural_bias_case_added"])
        topology = json.loads((root / "dtype_topology_fp32_seq64.json").read_text())
        self.assertEqual(topology["dtype"], "fp32")
        self.assertEqual(topology["runtime_symbol_count"], 45)
        self.assertEqual(topology["runtime_invocation_count"], 670)
        self.assertFalse(topology["candidate_values_used_to_select_or_classify"])
        audit = json.loads((root / "implementation_matrix.json").read_text())
        topologies = audit["generated_region_dynamic_coverage"]["dtype_topology_boundaries"]
        self.assertEqual(len(topologies), 6)
        self.assertTrue(all(row["semantic_mapping_status"] == "UNRESOLVED_TOPOLOGY_ONLY" for row in topologies))
        self.assertEqual({row["seq_len"] for row in topologies}, {64, 128, 256})
        mappings = audit["generated_region_dynamic_coverage"]["dtype_semantic_mapping_boundaries"]
        self.assertEqual(len(mappings), 6)
        self.assertTrue(all(not row["candidate_values_used_to_select_or_classify"] for row in mappings))
        self.assertEqual({row["mapped_invocations"] for row in mappings if row["seq_len"] == 64}, {318})
        self.assertEqual({row["mapped_invocations"] for row in mappings if row["seq_len"] in (128, 256)}, {264})
        self.assertTrue(all(row["unresolved_invocations"] > 0 for row in mappings))
        observations = audit["generated_region_dynamic_coverage"]["dtype_semantic_observations"]
        self.assertEqual(len(observations), 6)
        self.assertEqual({row["seq_len"] for row in observations}, {64, 128, 256})
        self.assertEqual({row["tf32"] for row in observations}, {False, True})
        self.assertEqual({row["mapped_invocations"] for row in observations if row["seq_len"] == 64}, {318})
        self.assertEqual({row["mapped_invocations"] for row in observations if row["seq_len"] != 64}, {264})
        self.assertTrue(all(row["mapped_regions_observed_at_all_steps_and_repeats"] == row["mapped_invocations"] for row in observations))
        self.assertTrue(all(row["all_mapped_regions_present_at_all_steps_and_repeats"] for row in observations))
        self.assertTrue(all(row["rows_with_any_nonexact_endpoint"] > 0 for row in observations))
        self.assertTrue(all(not row["natural_bias_case_added"] for row in observations))
        unresolved = json.loads((root / "dtype_unresolved_boundary.json").read_text())
        self.assertEqual(len(unresolved["entries"]), 6)
        self.assertTrue(all(not row["candidate_values_used_to_select_or_classify"] for row in unresolved["entries"]))
        self.assertTrue(all(not row["correctness_verdict_assigned"] for row in unresolved["entries"]))
        self.assertTrue(all(sum(item["invocations"] for item in row["by_boundary_class"]) == row["unresolved_invocations"] for row in unresolved["entries"]))

    def test_expanded_fp32_endpoint_campaign_is_complete_only_for_mapped_subset(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results" / "final"
        mapping = json.loads((root / "dtype_endpoint_mapping_fp32_seq128.json").read_text())
        self.assertEqual(mapping["denominator"]["mapped_invocations"], 526)
        self.assertEqual(mapping["denominator"]["unresolved_invocations"], 144)
        endpoint = json.loads((root / "dtype_evolving_fp32_seq128_expanded.json").read_text())
        self.assertEqual(endpoint["checkpoint_steps"], [0, 1, 2, 4, 8, 16, 32, 64])
        self.assertTrue(endpoint["gates"]["all_mapped_invocations_observed_at_all_checkpoints_and_repeats"])
        self.assertTrue(endpoint["gates"]["all_repeats_match"])
        self.assertFalse(endpoint["gates"]["f_b_closure_complete"])
        self.assertFalse(endpoint["gates"]["natural_bias_case_added"])


if __name__ == "__main__":
    unittest.main()
