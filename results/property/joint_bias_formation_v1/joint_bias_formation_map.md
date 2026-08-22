# Joint Bias Formation Map v1

This report separates source distribution, F+B/optimizer response, and closed-loop propagation.
Missing replay vectors are unresolved; no formation label is inferred from SEUP or trajectory drift.

| Case | Source | Response | Propagation |
|---|---|---|---|
| `qwen_saved_p_seq128` | `MEASURED_CASE_INTERVENTION` | `MEASURED_ANTITHETIC_RESPONSE` | `MEASURED_TRAJECTORY_FEEDBACK_SIGNATURE` |
| `qwen_layer23_attention_state` | `MEASURED_CASE_INTERVENTION` | `UNRESOLVED_MISSING_ANTITHETIC_REPLAY` | `MEASURED_TRAJECTORY_SUMMARY` |
| `mamba_seq64_input_proj` | `SUPPORTING_ONLY` | `UNRESOLVED_MISSING_ANTITHETIC_REPLAY` | `UNRESOLVED_NO_SEPARATE_VECTOR_PROPAGATOR` |
| `liger_fused_ce_t128` | `MEASURED_CASE_INTERVENTION` | `UNRESOLVED_MISSING_ANTITHETIC_REPLAY` | `UNRESOLVED_NO_SEPARATE_VECTOR_PROPAGATOR` |
| `phi4_lm_head_dx_seq64` | `MEASURED_CASE_INTERVENTION` | `MEASURED_TRANSPORT_PAIRING_INTERVENTION` | `MEASURED_TRAJECTORY_SUMMARY` |
| `qwen_l23_key_materialization_seq1024` | `UNRESOLVED_NO_FORMATION_ARTIFACT` | `UNRESOLVED_NO_REPLAY_ARTIFACT` | `UNRESOLVED_NO_PROPAGATION_ARTIFACT` |
| `qwen_rsqrt_seq128` | `UNRESOLVED_NO_FORMATION_ARTIFACT` | `UNRESOLVED_NO_REPLAY_ARTIFACT` | `UNRESOLVED_NO_PROPAGATION_ARTIFACT` |
| `qwen_bmm_seq64` | `UNRESOLVED_NO_FORMATION_ARTIFACT` | `UNRESOLVED_NO_REPLAY_ARTIFACT` | `UNRESOLVED_NO_PROPAGATION_ARTIFACT` |
| `qwen_vproj_seq128` | `SUPPORTING_ONLY` | `UNRESOLVED_MISSING_ANTITHETIC_REPLAY` | `UNRESOLVED_NO_SEPARATE_VECTOR_PROPAGATOR` |
| `qwen3vl_silu_seq160` | `MEASURED_CASE_INTERVENTION` | `MEASURED_ANTITHETIC_RESPONSE` | `MEASURED_TRAJECTORY_FEEDBACK_SIGNATURE` |

## Current interpretation

The existing repository has exact antithetic response replays for saved-P and SiLU, a composite transport intervention for Phi, and source-event/chunk-geometry evidence for Liger. A host-GPU Liger 32-state rerun completed, but its confirmation population remained unresolved under the frozen gate. The 24-state chunk intervention confirmed BF16-specific directional geometry (24/24 versus FP32 13/11), while the separate RN→SR default residual screen was diffusive (A=0.942 versus SR=0.975/1.001) and is retained as a negative/diagnostic result. A half-learning-rate Phi trajectory still passed SEUP with local accumulation 2.37477e-5, feedback 2.04859e-6, and recurrence residual 1.01e-8. The deterministic 12-row screen-negative audit is screen-level only; no full trajectory label is imputed.

The next causal measurements are therefore: real Liger SR/order-breaking, a generic response replay for Liger/Phi, and fixed-update propagation probes. No missing vector is imputed.
