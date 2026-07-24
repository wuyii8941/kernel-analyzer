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

The full-model twins start at exact zero divergence and reach L2 weight divergence `1.58346e-5` after 100 SGD steps (`relative=1.099e-8`). There are 201 fork events across 85 optimizer steps.

This first trajectory measures every five steps. Every measured interval contains at least one fork (`fork_intervals=20`, `no_fork_intervals=0`), so it establishes backend-only weight divergence but cannot compare fork-associated jumps against no-fork drift. A stepwise measurement rerun is required for that causal timing comparison.

## External Validity
This full-model coupling experiment uses T4 FP16 and repeatedly trains on the two exact replay rollout batches from the earliest fork state. It is mechanism evidence, not a replacement for a fresh long-horizon production BF16 run.

## Summary
| status | backend_only_difference | exact_weight_divergence | weight_scope | optimizer | learning_rate | measure_every | model_parameter_bytes_each | trainable_parameter_bytes_each | pre_alt_free_gpu_bytes | estimated_increment_gpu_bytes | path_ref | path_alt | optimizer_steps | trajectory_rows | weight_measurements | total_fork_events | fork_steps | final_weight_divergence | final_relative_weight_divergence | mean_divergence_jump_fork_intervals | mean_divergence_jump_no_fork_intervals | fork_intervals | no_fork_intervals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| completed | True | True | full_model | SGD | 1e-06 | 5 | 2384199680 | 2384199680 | 13142392832 | 9300082688 | hf-eager-fp16-sdpa-math-step5 | hf-compile-fp16-sdpa-math-step5 | 100 | 101 | 21 | 201 | 85 | 1.583463686203052e-05 | 1.0989925563271244e-08 | 7.91731843101526e-07 | None | 20 | 0 |
