# Unified trajectory evidence audit

This is an artifact audit, not a new experiment.  It separates causal
paired parameter separation from signed directional persistence and from
same-contrast mechanism-to-persistence closure.

| case | separation | directional persistence | contrast alignment | same-contrast full chain | initial norm | final norm |
|---|---|---|---|---:|---:|---:|
| liger_fused_ce | TRAJECTORY_SEPARATION | CONFIRMED | ALIGNED | True | 8.586806677537658e-06 | 0.0022393549008995974 |
| phi4_seq64_lmhead_dx | TRAJECTORY_SEPARATION | CONFIRMED | ALIGNED | True | 2.127923607986304e-06 | 9.185909584630281e-05 |
| qwen64_vproj_mm | TRAJECTORY_SEPARATION | CONFIRMED | MISMATCH | False | 0.003998710308223963 | 0.010890079662203789 |
| qwen128_vproj_mm | TRAJECTORY_SEPARATION | NOT_CONFIRMED | ALIGNED | False | 0.0037479896564036608 | 0.008157934993505478 |
| qwen_saved_p_seq128 | TRAJECTORY_SEPARATION | CONFIRMED | ALIGNED | True | 0.0044684866443276405 | 0.008657907135784626 |
| qwen3vl_silu_layer0 | TRAJECTORY_SEPARATION | CONFIRMED | ALIGNED_BASE_CONTRAST | False | 0.0020657628774642944 | 0.0813409760594368 |
| mamba_seq64_input_proj | TRAJECTORY_SEPARATION | CONFIRMED | MISMATCH | False | 0.004031004849821329 | 0.008289474993944168 |
| qwen_layer23_attention_state | TRAJECTORY_SEPARATION | CONFIRMED | ALIGNED_SEMANTIC_SUPERSET | True | 0.00032629986526444554 | 0.0006082479958422482 |

The table contains only complete artifact rows.  The strict count is
8 complete paired trajectory artifacts and 8 semantic cases.
7 have confirmed ordered-trajectory directional persistence; 1 have separation without that proof.
4 connect the current formation mechanism to persistence using an aligned repair contrast.
Excluded candidates (including the incomplete layer-23 key repair) are
listed in the JSON audit and are not silently counted as duplicates.

All eight are paired-separation artifacts.  The directional subset may
be called persistent trajectory-bias cases.  Flash-style persistent-local
and feedback-sustained regimes remain distinct, and only an aligned-contrast
subset closes the identified formation mechanism to source persistence.
