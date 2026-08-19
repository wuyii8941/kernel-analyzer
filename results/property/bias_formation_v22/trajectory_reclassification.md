# v2.2 trajectory-level reclassification

This is an artifact audit, not a new formation experiment.  v2.1 fixed
carrier failures are not used as trajectory negatives.  Mechanism and
property claims remain unresolved until the original P1-P6 theory is
tested with endpoint-level interventions.

| case | old status | old fixed-direction gate | v2.2 trajectory status | initial norm | final norm |
|---|---|---:|---|---:|---:|
| liger_fused_ce | COMPLETE_LIGER_ACCUMULATOR_REPAIR_LIVE_WEIGHT_CAUSAL_CHAIN | True | TRAJECTORY_BIAS | 8.586806677537658e-06 | 0.0022393549008995974 |
| phi4_seq64_lmhead_dx | COMPLETE_PAIRED_EVOLVING_FINAL_NORM_TRAJECTORY | True | TRAJECTORY_BIAS | 2.127923607986304e-06 | 9.185909584630281e-05 |
| qwen64_vproj_mm | PASS_STRICT_FLASH_STYLE_CASE | True | TRAJECTORY_BIAS | 0.003998710308223963 | 0.010890079662203789 |
| qwen128_vproj_mm | FAIL_DIRECTIONAL_ACCUMULATION | False | TRAJECTORY_BIAS | 0.004516110755503178 | 0.010251143015921116 |
| qwen_saved_p_seq128 | PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE | True | TRAJECTORY_BIAS | 0.0044684866443276405 | 0.008657907135784626 |
| qwen3vl_silu_layer0 | FAIL_DIRECTIONAL_ACCUMULATION | False | TRAJECTORY_BIAS | 0.0020657628774642944 | 0.08343788981437683 |
| mamba_seq64_input_proj | PASS_STRICT_FLASH_STYLE_CASE | True | TRAJECTORY_BIAS | 0.004031004849821329 | 0.008289474993944168 |
| qwen_layer23_attention_state | COMPLETE | True | TRAJECTORY_BIAS | 0.00032629986526444554 | 0.0006082479958422482 |

A v2.2 trajectory case means a complete causal candidate/repair run has
basis-free live parameter separation above its initial separation.  It does
not mean that the local residual is globally biased, nor that a common
property has been discovered.
