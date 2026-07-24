# Phase 6 Twin Training

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- same_initial_weights: PASS
- same_seed_data_optimizer: PASS
- backend_only_difference: PASS
- fixed_token_alignment: PASS
- exact_trainable_weight_divergence: PASS

## Delta Self Control
Phase 1 self-consistency gate remains authoritative for backend attribution.

## Summary
Backend-only twins ran in lockstep; fork timestamps and exact trainable-weight divergence were recorded.

The full-model twins start from identical weights and reach L2 divergence `1.58346e-5` after 100 lockstep SGD updates. Exact divergence was measured after every optimizer step.

Across 85 fork steps, mean divergence increment is `1.81361e-7`; across 15 no-fork steps it is `2.79281e-8`, a ratio of `6.494`. A 10,000-draw step bootstrap gives a 95% CI of `[6.54e-8, 2.46e-7]` for the mean increment difference, excluding zero. Median increments are `4.98e-8` and `2.01e-8`, respectively.

This supports temporal coupling between clipping branch forks and faster parameter divergence. It does not by itself prove that forks are the only cause of divergence: backend rounding drift remains present, and the experiment repeatedly alternates two replay rollout batches.

## External Validity
This mechanism experiment uses the exact step-5 state, T4 FP16, full-model SGD, and two replay rollout batches. Long-horizon production BF16 training remains a separate replication target.

## Summary
| status | backend_only_difference | exact_weight_divergence | weight_scope | optimizer | learning_rate | measure_every | model_parameter_bytes_each | trainable_parameter_bytes_each | pre_alt_free_gpu_bytes | estimated_increment_gpu_bytes | path_ref | path_alt | optimizer_steps | trajectory_rows | weight_measurements | total_fork_events | fork_steps | final_weight_divergence | final_relative_weight_divergence | mean_divergence_jump_fork_intervals | mean_divergence_jump_no_fork_intervals | fork_intervals | no_fork_intervals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| completed | True | True | full_model | SGD | 1e-06 | 1 | 2384199680 | 2384199680 | 13142392832 | 9300082688 | hf-eager-fp16-sdpa-math-step5 | hf-compile-fp16-sdpa-math-step5 | 100 | 101 | 101 | 201 | 85 | 1.583463686203052e-05 | 1.0989925563271244e-08 | 1.8136136251944036e-07 | 2.792806985853939e-08 | 85 | 15 |
