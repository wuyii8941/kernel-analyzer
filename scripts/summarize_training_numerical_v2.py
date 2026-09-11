#!/usr/bin/env python3
"""Report every retry; never select a result by its numerical verdict."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_training_numerical_v2 import BASE, CASES, save_new
from scripts.recover_training_numerical_runtime import CASES as RECOVERY_CASES


def build_progress():
    attempts=[]
    for path in sorted(BASE.rglob("*.json")):
        if path.name not in {"failure.json","status.json","recomputed.json"} and path.parent.name not in {"execution","recomputed"}:
            continue
        d=json.loads(path.read_text())
        attempts.append({"path":str(path.relative_to(BASE)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "case_id":d.get("case_id"),"measurement_status":d.get("measurement_status",d.get("status")),
                         "equivalence_decision":d.get("equivalence_decision"),
                         "fixed_suite_total_rms":d.get("bias_analysis",{}).get("fixed_suite_total_rms"),
                         "aligned_ratio_of_sums":d.get("bias_analysis",{}).get("fixed_suite_aligned_ratio_of_sums"),
                         "reason":d.get("error",d.get("reason")),
                         "write_protocol":d.get("provenance",{}).get("write_protocol")})
    valid={x['case_id'] for x in attempts if x['measurement_status']=='VALID'}
    expected = set(CASES.values()) | {row[0] for row in RECOVERY_CASES.values()}
    validation=json.loads((BASE/"synthetic_validation.json").read_text())
    verification_path = BASE / "report_recomputation_verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    reports_verified = (verification.get("status") == "VERIFIED_RECOMPUTATION"
                        and bool(verification.get("reports")))
    for record in verification.get("reports", []):
        for name in ("raw", "report"):
            path = BASE / record[name]
            reports_verified = reports_verified and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record[name + "_sha256"]
    training = BASE / "language_training_confirmation_iid"
    training_summary = json.loads((training / "summary.json").read_text()) if (training / "summary.json").exists() else {}
    training_verification = json.loads((training / "execution_verification.json").read_text()) if (training / "execution_verification.json").exists() else {}
    training_complete = (training_summary.get("status") == "COMPLETE_FROZEN_INITIALIZATION_CONFIRMATION"
                         and training_verification.get("status") == "VERIFIED_RECORDED_EXECUTION")
    probe_path = BASE / "language_checkpoint_direct_probe/summary.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    identity_path = BASE / "language_accumulation_identity_retry/result.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    mechanism_linked = (probe.get("status") == "COMPLETE_SAME_TRAINING_CHECKPOINT_MECHANISM_FOLLOWUP"
                        and identity.get("status") == "IDENTITY_VERIFIED")
    new_path = BASE / "granite_expert_order_confirmation_v2/recomputed.json"
    new_report = json.loads(new_path.read_text()) if new_path.exists() else {}
    new_complete = (new_report.get("measurement_status") == "VALID"
                    and new_report.get("provenance", {}).get("data_use") == "NEW_MODEL_AND_MOE_COMBINATION_IMPLEMENTATION_FIXED_SUITE_CONFIRMATION")
    temporal_path = BASE / "language_temporal_extension/summary.json"
    temporal = json.loads(temporal_path.read_text()) if temporal_path.exists() else {}
    temporal_complete = temporal.get("status") == "COMPLETE_VERIFIED_TEMPORAL_EXTENSION"
    optimizer_root = BASE.parent / "numerical_coverage_v1"
    optimizer_mechanism_path = optimizer_root / "torchao_adamw8bit_block_mechanism_v1/summary.json"
    optimizer_training_path = optimizer_root / "mamba_adamw8bit_training_confirmation_v1/verification.json"
    optimizer_component_path = optimizer_root / "torchao_adamw8bit_component_mechanism_v1/summary.json"
    optimizer_hybrid_path = optimizer_root / "mamba_adamw8bit_hybrid_training_confirmation_v1/verification_v2.json"
    optimizer_population_path = optimizer_root / "adamw8bit_population_update_v1/verification.json"
    optimizer_full_model_path = optimizer_root / "adamw8bit_full_model_block_probe_v1/verification_v2.json"
    optimizer_compensation_path = optimizer_root / "adamw8bit_error_compensation_training_v1/verification.json"
    optimizer_compensation_summary_path = optimizer_root / "adamw8bit_error_compensation_training_v1/summary.json"
    optimizer_mechanism = (json.loads(optimizer_mechanism_path.read_text())
                           if optimizer_mechanism_path.exists() else {})
    optimizer_training = (json.loads(optimizer_training_path.read_text())
                          if optimizer_training_path.exists() else {})
    optimizer_component = (json.loads(optimizer_component_path.read_text())
                           if optimizer_component_path.exists() else {})
    optimizer_hybrid = (json.loads(optimizer_hybrid_path.read_text())
                        if optimizer_hybrid_path.exists() else {})
    optimizer_population = (json.loads(optimizer_population_path.read_text())
                            if optimizer_population_path.exists() else {})
    optimizer_full_model = (json.loads(optimizer_full_model_path.read_text())
                            if optimizer_full_model_path.exists() else {})
    optimizer_compensation = (json.loads(optimizer_compensation_path.read_text())
                              if optimizer_compensation_path.exists() else {})
    optimizer_compensation_summary = (json.loads(optimizer_compensation_summary_path.read_text())
                                      if optimizer_compensation_summary_path.exists() else {})
    optimizer_mechanism_confirmed = optimizer_mechanism.get("prediction_result") == "CONFIRMED"
    optimizer_training_material = (optimizer_training.get("status") == "VERIFIED"
                                   and optimizer_training.get("primary", {}).get("decision") == "MATERIAL_EFFECT")
    optimizer_modification_confirmed = (
        optimizer_training.get("modified_variant", {}).get("decision")
        == "CONFIRMED_CLOSER_TO_REFERENCE"
        or optimizer_hybrid.get("primary", {}).get("decision") == "MATERIAL_IMPROVEMENT"
        or (optimizer_compensation.get("status") == "VERIFIED"
            and optimizer_compensation.get("recomputed", {}).get("decision") == "MATERIAL_IMPROVEMENT")
    )
    supporting_pairs = []
    for row in temporal.get("rows", []):
        trajectories = row["trajectory_diagnostics"].values()
        if (all(d["all_later_windows_same_halfspace"] and d["each_window_split_half_same_halfspace"] for d in trajectories)
                and row["loss_gaps"][-1]["candidate_minus_reference_loss"] != 0):
            supporting_pairs.append(row["pair"])
    remaining = []
    if not reports_verified:
        remaining.append("verify report decisions against unchanged original-coordinate statistics")
    if not new_complete:
        remaining.append("new implementation confirmation")
    if not temporal_complete:
        remaining.append("finish and verify all frozen evolving-state windows and paired extensions")
    if not mechanism_linked:
        remaining.append("link same-training source identity and final-state direct update measurements")
    if expected - valid:
        remaining.append("complete original-case runtime binding and capture")
    if not training_complete:
        remaining.append("finish and verify frozen normal-language initialization confirmation")
    selected_execution_complete = (
        not remaining and validation["status"] == "PASS_FIXED_SUITE_PATH"
    )
    if not optimizer_modification_confirmed:
        remaining.append(
            "explain the AdamW8bit recursive trajectory response and validate an effective modification"
        )
    if optimizer_population.get("status") != "VERIFIED":
        remaining.append("collect and verify one real declared state-population endpoint")
    summary={"schema":"training-numerical-v2-progress-audit","status":"INCOMPLETE_FULL_RESEARCH_PLAN",
             "selected_experiment_execution_complete":selected_execution_complete,
             "audit_scope":"SELECTED_RECAPTURE_AND_LANGUAGE_EXPERIMENTS_NOT_ENTIRE_20260905_PLAN",
             "full_plan_unverified_requirements":["cross-model and warm-state generalization of the compensation chain", "a production-scope optimizer modification beyond the dense prototype", "bounded validation of an additional operator family with a valid three-stage measurement", "an unconditional random-state mean-energy Q certificate remains unavailable without pre-observation energy bounds"],
             "recapture_valid_cases":sorted(valid),"recapture_missing_cases":sorted(expected-valid),
             "all_attempts":attempts,"fixed_suite_validation":validation['status'],
             "saved_reports_recomputation_verified":reports_verified,
             "unconditional_mean_energy_population_guarantee_enabled":False,
             "unconditional_mean_energy_population_guarantee_is_mathematically_unavailable_without_tail_information":True,
             "optimizer_history_exceedance_population_guarantee_enabled":optimizer_population.get("status") == "VERIFIED",
             "optimizer_history_exceedance_population_decision":optimizer_population.get("decision"),
             "optimizer_history_exceedance_count":optimizer_population.get("exceedance_count"),
             "optimizer_history_exceedance_probability_bounds":optimizer_population.get("one_sided_probability_bounds"),
             "optimizer_history_population_scope":"Exact prevalence endpoint for iid length-8 histories drawn from one fixed empirical token-block population; not mean Q or recursive training.",
             "training_confirmation_execution_complete":training_complete,
             "training_confirmation_primary_prediction_confirmed":training_summary.get("primary_prediction_confirmed"),
             "training_confirmation_scope":training_summary.get("scope"),
             "same_training_mechanism_followup_complete":mechanism_linked,
             "same_training_mechanism_scope":"One exact accumulation identity plus all final-checkpoint fixed-state probes; not a theorem of temporal persistence or sufficient causation of loss.",
             "new_model_implementation_measurement_complete":new_complete,
             "temporal_extension_complete":temporal_complete,
             "observed_window_direction_and_loss_chain_supported":bool(supporting_pairs),
             "optimizer_quantization_mechanism_confirmed":optimizer_mechanism_confirmed,
             "optimizer_training_material_effect_confirmed":optimizer_training_material,
             "optimizer_training_modification_confirmed":optimizer_modification_confirmed,
             "optimizer_compensation_training_execution_verified":optimizer_compensation.get("status") == "VERIFIED",
             "optimizer_compensation_training_decision":optimizer_compensation.get("recomputed", {}).get("decision"),
             "optimizer_compensation_training_mean_default_minus_compensated":optimizer_compensation.get("recomputed", {}).get("mean"),
             "optimizer_compensation_training_interval_95":optimizer_compensation.get("recomputed", {}).get("interval_95"),
             "optimizer_compensation_training_scope":optimizer_compensation_summary.get("claim_scope"),
             "optimizer_compensation_training_note":"Dense FP32 compensation prototype; verified on eight new streams for one Mamba checkpoint and optimizer state, not a production optimizer or cross-model guarantee.",
             "optimizer_component_prediction_confirmed":optimizer_component.get("prediction_result") == "CONFIRMED",
             "optimizer_component_primary_write_reduced":optimizer_component.get("primary_parameter_write_reduction"),
             "optimizer_hybrid_training_execution_verified":optimizer_hybrid.get("status") == "VERIFIED",
             "optimizer_hybrid_training_decision":optimizer_hybrid.get("primary", {}).get("decision"),
             "optimizer_hybrid_training_mean_default_minus_modified":optimizer_hybrid.get("primary", {}).get("mean"),
             "optimizer_hybrid_training_interval_95":optimizer_hybrid.get("primary", {}).get("interval_95"),
             "optimizer_full_model_block_prediction":optimizer_full_model.get("prediction_result"),
             "optimizer_full_model_block64_lower_count":optimizer_full_model.get("block64_lower_rms_count"),
             "optimizer_full_model_block_probability_bounds":optimizer_full_model.get("one_sided_probability_bounds"),
             "optimizer_full_model_block_mean_rms":optimizer_full_model.get("mean_full_model_write_rms"),
             "optimizer_full_model_interpretation":"Block64 reduces full-model direct write RMS on new iid histories, while its prior 1024-step loss improvement remains unconfirmed; direct RMS is not a sufficient training-loss predictor.",
             "optimizer_training_scope":optimizer_training.get("inference_boundary"),
             "pairs_with_observed_cross_window_direction_and_final_loss_difference":supporting_pairs,
             "bounded_chain_rule":"Descriptive conjunction of the already declared window signs and loss non-identity, not a new calibrated test or proof over unmeasured steps. Repeated pairs remain one implementation case.",
             "remaining_acceptance":remaining,
             "unclaimed_guarantees":["arbitrary-state or full-training safety", "every unmeasured step is directional", "numerical degradation is universal", "the directional component alone accounts for all loss change"],
             "optional_population_claim":"A real exact exceedance-prevalence claim is enabled for the declared AdamW8bit history population. Mean-energy Q remains disabled without useful pre-observation bounds.",
             "claim_boundary":"Completion concerns execution and verification of the bounded plan, independent of positive outcomes. VALID concerns a declared fixed-suite measurement, not proved bias or loss consequence. The observed chain has a separate field and cannot be inferred from completion."}
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    summary = build_progress()
    save_new(args.output,summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
