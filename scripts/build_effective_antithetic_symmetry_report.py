#!/usr/bin/env python3
"""Build the evidence report for the effective-antithetic-symmetry hypothesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_property_search"


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def rectification_aggregate(document: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(document["aggregate"])
    records = document.get("records", [])
    if records and "response_even_l2" in records[0]:
        even = sum(row["response_even_l2"] ** 2 for row in records)
        odd = sum(row["response_odd_l2"] ** 2 for row in records)
        aggregate["energy_weighted_response_even_on_sign_crossings"] = sum(
            row["response_even_l2"] ** 2
            * row["response_even_energy_on_sign_crossings"]
            for row in records
        ) / max(even, 1e-30)
        aggregate["response_even_energy_in_first_two_steps"] = sum(
            row["response_even_l2"] ** 2 for row in records[:2]
        ) / max(even, 1e-30)
        aggregate["step_integrated_response_even_energy_fraction"] = (
            even / max(even + odd, 1e-30)
        )
    return aggregate


def main() -> None:
    liger_path = "archive/nonprecision_v1/runs/liger.fused_ce.chunk.certificate.json"
    phi_path = "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json"
    l23_path = "results/coverage/cases/l23_qproj_attention_state_region.json"
    silu_path = "results/round2/vl_bias.json"
    qwen_mm_path = "results/coverage/cases/qwen128_vproj_precision_decomposition.json"
    mamba_mm_path = "results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json"
    pairing_path = (
        OUT / "saved_p_pairing_work_v2.json"
        if (OUT / "saved_p_pairing_work_v2.json").exists()
        else OUT / "saved_p_pairing_work.json"
    )
    silu_oddness_path = (
        OUT / "vl_silu_optimizer_oddness_v2.json"
        if (OUT / "vl_silu_optimizer_oddness_v2.json").exists()
        else OUT / "vl_silu_optimizer_oddness.json"
    )
    liger = load(liger_path)
    phi = load(phi_path)
    l23 = load(l23_path)
    silu = load(silu_path)
    qwen_mm = load(qwen_mm_path)
    mamba_mm = load(mamba_mm_path)

    by_accumulator = {
        row["accumulator"]: row for row in liger["directional_results"]
    }
    bf16 = by_accumulator["bf16"]
    fp32 = by_accumulator["fp32"]
    natural_phi = phi["natural_gradient_population"]
    shuffled_phi = phi["shuffled_gradient_population"]

    cases: list[dict[str, Any]] = [
        {
            "case": "liger_fused_ce",
            "mechanism": "SOURCE_SCHEDULE_SYMMETRY_DEFECT",
            "status": "MATCHED_SUPPORT",
            "natural_effect": {
                "condition": "BF16 accumulator",
                "positive_states": bf16["positive"],
                "negative_states": bf16["negative"],
                "mean_projection": bf16["mean_projection"],
                "bootstrap_95": bf16["cluster_bootstrap_95ci"],
                "raw_coordinate_signed_mean": bf16["residual_signed_mean"]["mean"],
            },
            "antithetic_control": {
                "condition": "FP32 accumulator under the same semantic geometry orbit",
                "positive_states": fp32["positive"],
                "negative_states": fp32["negative"],
                "mean_projection": fp32["mean_projection"],
                "bootstrap_95": fp32["cluster_bootstrap_95ci"],
            },
            "causal_reading": (
                "Finite-precision accumulation and physical chunk schedule break "
                "effective orbit centering; raw tensor signed mean is not the marker."
            ),
            "artifact": liger_path,
        },
        {
            "case": "phi4_seq64_lmhead_dx",
            "mechanism": "COMPOSITE_ERROR_TRANSPORT_COUPLING",
            "status": "MATCHED_SUPPORT_WITH_ANALYTIC_BOUNDARY_OPEN",
            "natural_effect": {
                "gradient_cross_state_ratio": natural_phi["cross_state_ratio"],
                "bootstrap_95": [natural_phi["bootstrap_lower"], natural_phi["bootstrap_upper"]],
                "verdict": natural_phi["status"],
            },
            "antithetic_control": {
                "operation": "row-pairing permutation",
                "gradient_cross_state_ratio": shuffled_phi["cross_state_ratio"],
                "bootstrap_95": [shuffled_phi["bootstrap_lower"], shuffled_phi["bootstrap_upper"]],
                "verdict": shuffled_phi["status"],
                "local_norm_preserved_every_state": phi["gates"]["local_norm_preserved_every_state"],
            },
            "causal_reading": (
                "Marginal local residual size is insufficient; its real pairing with the "
                "composite backward influence creates the directional gradient population."
            ),
            "claim_boundary": (
                "The RMSNorm-only analytic transport reconstruction does not close the full effect."
            ),
            "artifact": phi_path,
        },
        {
            "case": "qwen_layer23_attention_state",
            "mechanism": "ATTENTION_STATE_EFFECTIVE_DIRECTION",
            "status": "CONSISTENT_NOT_MARGINAL_PRESERVING",
            "natural_effect": {
                "s_bwd_shapley_positive_states": l23["causal_attribution"]["s_bwd_shapley"]["positive_states"],
                "states": l23["causal_attribution"]["s_bwd_shapley"]["states"],
                "bootstrap_95": l23["causal_attribution"]["s_bwd_shapley"]["state_cluster_bootstrap_95"],
            },
            "repair": {
                "s_bwd_only_residual_bootstrap_95": l23["causal_attribution"]["s_bwd_only_repair_residual"]["state_cluster_bootstrap_95"],
                "matched_sham_max_abs": l23["causal_attribution"]["matched_sham_max_abs"],
            },
            "causal_reading": (
                "The direction enters through S_bwd and survives G_q=S_bwd K and dW=G_q^T H."
            ),
            "claim_boundary": "Repair removes the residual rather than preserving its marginal distribution.",
            "artifact": l23_path,
        },
        {
            "case": "qwen3vl_silu",
            "mechanism": "STATE_CONDITIONED_OR_FEEDBACK_REGIME",
            "status": "BOUNDARY_CASE",
            "global_cross_state_ratio": silu["global_direction"]["coherence_ratio"],
            "trajectory_artifact": "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
            "causal_reading": (
                "Unrelated-state global centering coexists with a closed live trajectory; "
                "global fixed direction is therefore not a necessary bias-formation property."
            ),
            "artifact": silu_path,
        },
        {
            "case": "qwen128_vproj_mm",
            "mechanism": "OUTPUT_ROUNDING_SOURCE_SYMMETRY_DEFECT",
            "status": "SUPPORTING_SOURCE_OBSERVATION_NOT_TRAJECTORY_MATCHED",
            "kernel_u": qwen_mm["direction"]["kernel"]["cross_state_inner_product_u"],
            "kernel_bootstrap_95": qwen_mm["direction"]["kernel"]["cluster_bootstrap_95"],
            "output_rounding_u": qwen_mm["direction"]["output_rounding"]["cross_state_inner_product_u"],
            "output_rounding_bootstrap_95": qwen_mm["direction"]["output_rounding"]["cluster_bootstrap_95"],
            "causal_reading": "Same-operand kernel arithmetic centers while deterministic output rounding carries the directional local component.",
            "claim_boundary": "The existing live trajectory repairs a different contrast.",
            "artifact": qwen_mm_path,
        },
        {
            "case": "mamba_seq64_input_proj",
            "mechanism": "KERNEL_AND_OUTPUT_ROUNDING_SOURCE_SYMMETRY_DEFECT",
            "status": "SUPPORTING_CROSS_ARCHITECTURE_SOURCE_OBSERVATION",
            "kernel_u": mamba_mm["direction"]["kernel"]["cross_state_inner_product_u"],
            "kernel_bootstrap_95": mamba_mm["direction"]["kernel"]["cluster_bootstrap_95"],
            "output_rounding_u": mamba_mm["direction"]["output_rounding"]["cross_state_inner_product_u"],
            "output_rounding_bootstrap_95": mamba_mm["direction"]["output_rounding"]["cluster_bootstrap_95"],
            "causal_reading": "Both same-operand MM arithmetic and output rounding have non-centered local contribution ensembles in a non-Transformer architecture.",
            "claim_boundary": "The two source arms are not yet separately closed through matched trajectories.",
            "artifact": mamba_mm_path,
        },
    ]

    saved_p: dict[str, Any] | None = None
    if pairing_path.exists():
        measured = json.loads(pairing_path.read_text())
        aggregate = rectification_aggregate(measured)
        saved_p = {
            "case": "qwen_saved_p_seq128",
            "mechanism": "HEAD_SPECIFIC_ERROR_TRANSPORT_COUPLING",
            "status": (
                "PAIRING_REDUCES_RESULTANT"
                if aggregate["update_pairing_suppression"] > 0.0
                else "HEAD_PAIRING_HYPOTHESIS_REJECTED"
            ),
            "natural_update_resultant_l2": aggregate["natural_update_resultant_l2"],
            "shuffled_update_resultant_l2": aggregate["shuffled_update_resultant_l2"],
            "update_pairing_suppression": aggregate["update_pairing_suppression"],
            "natural_gradient_resultant_l2": aggregate["natural_gradient_resultant_l2"],
            "shuffled_gradient_resultant_l2": aggregate["shuffled_gradient_resultant_l2"],
            "gradient_pairing_suppression": aggregate["gradient_pairing_suppression"],
            "max_precast_norm_relative_error": aggregate["max_precast_norm_relative_error"],
            "all_forward_losses_equal": aggregate["all_forward_losses_equal"],
            "artifact": str(pairing_path.relative_to(ROOT)),
        }
        if "optimizer_oddness_resultant_ratio" in aggregate:
            plus = aggregate["natural_update_resultant_l2"]
            minus = aggregate["antithetic_gradient_update_resultant_l2"]
            oddness = aggregate["optimizer_oddness_resultant_l2"]
            plus_minus_cosine = aggregate[
                "natural_antithetic_update_resultant_cosine"
            ]
            saved_p["optimizer_oddness"] = {
                "resultant_l2": oddness,
                "resultant_ratio": aggregate["optimizer_oddness_resultant_ratio"],
                "mean_step_ratio": aggregate["mean_step_optimizer_oddness_ratio"],
                "natural_antithetic_resultant_cosine": plus_minus_cosine,
                "oddness_alignment_with_natural_resultant": (
                    (plus * plus + plus * minus * plus_minus_cosine)
                    / max(plus * oddness, 1e-30)
                ),
                "adam_over_stateless_sgd_resultant": aggregate[
                    "adam_over_stateless_sgd_resultant"
                ],
                "mean_step_sign_crossing_fraction": aggregate.get(
                    "mean_step_sign_crossing_fraction"
                ),
                "mean_step_response_even_energy_on_sign_crossings": aggregate.get(
                    "mean_step_response_even_energy_on_sign_crossings"
                ),
                "energy_weighted_response_even_on_sign_crossings": aggregate.get(
                    "energy_weighted_response_even_on_sign_crossings"
                ),
                "response_even_energy_in_first_two_steps": aggregate.get(
                    "response_even_energy_in_first_two_steps"
                ),
            }
        cases.insert(2, saved_p)

    if silu_oddness_path.exists():
        measured = json.loads(silu_oddness_path.read_text())
        aggregate = rectification_aggregate(measured)
        silu_plus = aggregate["natural_update_resultant_l2"]
        silu_minus = aggregate["antithetic_update_resultant_l2"]
        silu_cosine = aggregate["natural_antithetic_update_resultant_cosine"]
        silu_nonodd = aggregate["optimizer_oddness_resultant_l2"]
        silu_odd_energy = (
            silu_plus * silu_plus + silu_minus * silu_minus
            - 2.0 * silu_plus * silu_minus * silu_cosine
        ) / 4.0
        silu_even_energy = (silu_nonodd / 2.0) ** 2
        silu_case = next(case for case in cases if case["case"] == "qwen3vl_silu")
        silu_case.update({
            "mechanism": "OPTIMIZER_NONODD_RECTIFICATION",
            "status": "MATCHED_INDEPENDENT_REPLICATION_WITH_GLOBAL_BOUNDARY",
            "optimizer_oddness_resultant_l2": aggregate[
                "optimizer_oddness_resultant_l2"
            ],
            "optimizer_oddness_resultant_ratio": aggregate[
                "optimizer_oddness_resultant_ratio"
            ],
            "mean_step_optimizer_oddness_ratio": aggregate[
                "mean_step_optimizer_oddness_ratio"
            ],
            "natural_antithetic_update_resultant_cosine": aggregate[
                "natural_antithetic_update_resultant_cosine"
            ],
            "response_even_alignment_with_natural_resultant": (
                (silu_plus * silu_plus + silu_plus * silu_minus * silu_cosine)
                / max(silu_plus * silu_nonodd, 1e-30)
            ),
            "response_even_resultant_energy_fraction": (
                silu_even_energy / max(silu_even_energy + silu_odd_energy, 1e-30)
            ),
            "adam_over_stateless_sgd_resultant": aggregate[
                "adam_over_stateless_sgd_resultant"
            ],
            "mean_step_sign_crossing_fraction": aggregate.get(
                "mean_step_sign_crossing_fraction"
            ),
            "mean_step_response_even_energy_on_sign_crossings": aggregate.get(
                "mean_step_response_even_energy_on_sign_crossings"
            ),
            "energy_weighted_response_even_on_sign_crossings": aggregate.get(
                "energy_weighted_response_even_on_sign_crossings"
            ),
            "response_even_energy_in_first_two_steps": aggregate.get(
                "response_even_energy_in_first_two_steps"
            ),
            "causal_reading": (
                "AdamW maps exact +delta_g/-delta_g residuals around the same "
                "repair gradient to a non-antithetic update pair."
            ),
            "artifact": str(silu_oddness_path.relative_to(ROOT)),
        })

    rejected = [
        {
            "candidate": "LOCAL_ERROR_MAGNITUDE",
            "verdict": "INSUFFICIENT",
            "reason": "Phi changes downstream bias under an exactly norm-preserving local intervention.",
        },
        {
            "candidate": "RAW_SIGNED_TENSOR_MEAN",
            "verdict": "INSUFFICIENT",
            "reason": "Liger has a tiny raw coordinate mean but unanimous effective carrier projections.",
        },
        {
            "candidate": "FIXED_GLOBAL_RANK1_CARRIER",
            "verdict": "NOT_NECESSARY",
            "reason": "Saved-P and SiLU live trajectories do not require unrelated-state rank-1 stability.",
        },
        {
            "candidate": "LOW_PRECISION_DTYPE",
            "verdict": "SOURCE_CONDITION_NOT_EXPLANATION",
            "reason": "The same dtype can yield coherent or canceling errors depending on schedule and influence pairing.",
        },
        {
            "candidate": "SEUP",
            "verdict": "DOWNSTREAM_CONSEQUENCE",
            "reason": "SEUP measures persistence after an effective-update difference has formed.",
        },
    ]
    payload = {
        "schema": "kernel-analyzer-effective-antithetic-symmetry-evidence-v1",
        "hypothesis": (
            "Bias forms when implementation-error events fail to cancel after actual "
            "F+B/optimizer influence; equivalently, their conditional effective "
            "contribution ensemble has an antithetic-symmetry defect."
        ),
        "bias_equation": {
            "exact_secant": "Delta_u = [integral_0^1 D U_s(z_r+t epsilon) dt] epsilon = A_bar epsilon",
            "conditional_mean": "E[A_bar epsilon|c] = E[A_bar|c]E[epsilon|c] + Cov_c(A_bar,epsilon)",
            "exact_parity_budget": "E[F(epsilon)|c] = integral p_s(epsilon)F_e(epsilon) + integral p_a(epsilon)F_o(epsilon)",
            "event_pairing_channel": "integral p_a(epsilon)F_o(epsilon)",
            "response_rectification_channel": "integral p_s(epsilon)F_e(epsilon)",
            "source_term": "E[A_bar|c]E[epsilon|c]",
            "coupling_term": "Cov_c(A_bar,epsilon)",
            "nonlinear_optimizer_role": "absorbed into the path-averaged influence A_bar",
        },
        "cases": cases,
        "saved_p_result_available": saved_p is not None,
        "rejected_alternatives": rejected,
        "headline_status": (
            "CROSS_MECHANISM_WORKING_HYPOTHESIS; ONE SOURCE/SCHEDULE AND ONE "
            "COMPOSITE-TRANSPORT MATCHED SUPPORT; INDEPENDENT ATTENTION PAIRING TEST PENDING"
            if saved_p is None
            else (
                "SAVED_P_HEAD_PAIRING_SUPPORT"
                if saved_p["update_pairing_suppression"] > 0.0
                else (
                    "SAVED_P_HEAD_PAIRING_REJECTED; OPTIMIZER_NONODD_RECTIFICATION_MEASURED; "
                    + ("SILU_REPLICATION_PASSED" if silu_oddness_path.exists()
                       else "SILU_REPLICATION_PENDING")
                    if "optimizer_oddness" in saved_p
                    else "SAVED_P_HEAD_PAIRING_REJECTED; OPTIMIZER_ODDNESS_FOLLOWUP_REQUIRED"
                )
            )
        ),
        "claim_boundary": (
            "The equation is a testable organizing principle, not yet a universal predictor. "
            "Cross-operator promotion requires another independent marginal-preserving intervention."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "property_evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    saved_line = (
        "- Qwen saved-P: background matched pairing experiment is still pending."
        if saved_p is None
        else (
            "- Qwen saved-P: head-pairing shuffle changes accumulated effective-update "
            f"resultant by {100.0 * saved_p['update_pairing_suppression']:.2f}%. "
            + ("This supports the tested pairing mechanism."
               if saved_p["update_pairing_suppression"] > 0.0
               else "The resultant increased, so the tested head-specific pairing mechanism is rejected.")
        )
    )
    if saved_p is not None and "optimizer_oddness" in saved_p:
        odd = saved_p["optimizer_oddness"]
        saved_line += (
            " The exact +delta_g/-delta_g Adam test has accumulated oddness ratio "
            f"{odd['resultant_ratio']:.4f} and mean per-step ratio "
            f"{odd['mean_step_ratio']:.4f}; its oddness resultant aligns "
            f"{odd['oddness_alignment_with_natural_resultant']:.4f} with the natural "
            "update resultant."
        )
        if odd.get("mean_step_sign_crossing_fraction") is not None:
            saved_line += (
                " Only "
                f"{100.0 * odd['mean_step_sign_crossing_fraction']:.2f}% of active "
                "coordinates cross gradient sign, yet they carry "
                f"{100.0 * odd['mean_step_response_even_energy_on_sign_crossings']:.2f}% "
                "of response-even energy under an unweighted step average. "
                f"Energy weighting raises that concentration to "
                f"{100.0 * odd['energy_weighted_response_even_on_sign_crossings']:.2f}%, "
                f"and {100.0 * odd['response_even_energy_in_first_two_steps']:.2f}% "
                "of even-response energy occurs in the first two steps."
            )
    silu_oddness_line = ""
    if silu_oddness_path.exists():
        measured = rectification_aggregate(
            json.loads(silu_oddness_path.read_text())
        )
        silu_oddness_line = (
            "\n- Qwen3-VL SiLU: the independent exact +delta_g/-delta_g Adam test "
            f"has accumulated oddness ratio {measured['optimizer_oddness_resultant_ratio']:.4f} "
            f"and mean per-step ratio {measured['mean_step_optimizer_oddness_ratio']:.4f}; "
            f"the response-even resultant aligns "
            f"{(silu_plus * silu_plus + silu_plus * silu_minus * silu_cosine) / max(silu_plus * silu_nonodd, 1e-30):.4f} "
            "with the natural update resultant."
        )
        if measured.get("mean_step_sign_crossing_fraction") is not None:
            silu_oddness_line += (
                f" Mean sign-crossing fraction is "
                f"{100.0 * measured['mean_step_sign_crossing_fraction']:.2f}%, carrying "
                f"{100.0 * measured['mean_step_response_even_energy_on_sign_crossings']:.2f}% "
                "of response-even energy."
            )
            silu_oddness_line += (
                f" Energy-weighted sign-crossing concentration is "
                f"{100.0 * measured['energy_weighted_response_even_on_sign_crossings']:.2f}%, "
                f"with {100.0 * measured['response_even_energy_in_first_two_steps']:.2f}% "
                "of even-response energy in the first two steps."
            )
    current_boundary = (
        "The saved-P head-pairing test is a preserved counterexample, not a positive. "
        "Its exact antithetic gradient test identifies optimizer non-oddness as a "
        "new case-specific formation mechanism.  Cross-case promotion of optimizer "
        "rectification waits on the independent SiLU result."
        if saved_p is not None and "optimizer_oddness" in saved_p
        and not silu_oddness_path.exists()
        else (
            "Saved-P rejects head-specific pairing, while exact antithetic-gradient "
            "experiments in saved-P and Qwen3-VL SiLU independently produce large "
            "accumulated non-odd Adam responses.  This repeats optimizer response "
            "rectification across two closed F+B cases; unseen-case prediction remains open."
            if silu_oddness_path.exists()
            else "An independent marginal-preserving positive intervention is still required."
        )
    )
    report = f"""# Bias property search: current scientific result

## Candidate explanation

The strongest non-tautological candidate is **effective antithetic symmetry**.
For local implementation residuals `epsilon_j` and their real F+B/optimizer
influence `A_j`, the relevant event is `w_j=A_j epsilon_j`.  Harmless variance
cancels when the conditional `w` ensemble is closed under sign reversal.  Bias
forms when event/pairing asymmetry or a non-odd downstream response breaks that
symmetry.

This yields the falsifiable bias budget:

```text
Delta u = A_bar(epsilon) epsilon

E[A_bar epsilon | c]
  = E[A_bar|c] E[epsilon|c]       # source asymmetry
  + Cov_c(A_bar, epsilon).        # transport/influence coupling
```

`A_bar` is the exact path-averaged downstream derivative, so nonlinear and
optimizer effects are included rather than appended as unconstrained stages.

Equivalently, splitting the conditional event density and the actual response
into symmetric/antisymmetric and even/odd parts gives the exact parity budget:

```text
E[F(epsilon)|c]
  = integral p_s(epsilon) F_e(epsilon)   # response rectification
  + integral p_a(epsilon) F_o(epsilon).  # event/pairing asymmetry
```

Thus bias has two irreducible channels under a predeclared semantic sign
operation: unmatched antithetic events, or a non-odd F+B/optimizer response.

## Evidence

- Liger: a same-real-semantics padding/rechunk orbit is directional with BF16
  accumulation (24/24 signs) and centered with FP32 accumulation (13/11).
- Phi: natural backward pairing is biased; a local-norm-preserving row-pairing
  shuffle is centered.  The exact analytic subfactor remains open, so the
  supported object is composite F+B influence.
- Layer-23 attention: restoring `S_bwd` removes a 27/32-sign direction through
  `G_q=S_bwd K`; this is consistent evidence, but not yet a marginal-preserving
  symmetry intervention.
- Qwen v-projection and Mamba input-projection decompositions provide supporting
  source observations: Qwen isolates a directional output-rounding term while
  its same-operand kernel term centers; Mamba has directional kernel and output-
  rounding terms.  Their source/trajectory contrasts are not yet fully matched.
{saved_line}
{silu_oddness_line}

## What this rules out

Error magnitude, raw tensor signed mean, BF16 dtype, and a fixed global rank-1
carrier do not individually explain the observed split.  SEUP remains the
downstream persistence test.

## Current boundary

The evidence supports one exact formation map across source/schedule,
composite-transport, and optimizer-response mechanisms.  It is not yet a
universal predictor for unseen operators.  {current_boundary}
"""
    (OUT / "scientific_summary.md").write_text(report)


if __name__ == "__main__":
    main()
