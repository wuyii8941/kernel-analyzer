# Phase 10 Zero-Initial-Clipping-Fork Mutation Trajectories

## Objective

Determine whether mutations with zero clipping forks at the frozen initial state can still change PPO gradients and parameter updates through the continuous within-branch channel.

## Results

| Mutation | First clipping-fork step | Total branch forks | Step-1 parameter L2 | Step-1 ratio to legal pair | Continuous divergence before fork |
|---|---:|---:|---:|---:|---|
| rotary_phase_fp16 | 3 | 78 | 5.40373e-06 | 0.489058 | True |
| decoder0_output_bf16_round | 2 | 79 | 4.89427e-06 | 0.44295 | True |
| logsoftmax_fp16 | 2 | 74 | 4.48932e-06 | 0.406301 | True |
| logsumexp_chunked_reverse | 2 | 76 | 2.84488e-06 | 0.257473 | True |

## Parameter Distance Growth

| Mutation | Step 1 vs legal | Step 5 vs legal | Step 20 vs legal |
|---|---:|---:|---:|
| rotary_phase_fp16 | 0.489058 | 1.12815 | 1.74262 |
| decoder0_output_bf16_round | 0.44295 | 0.911941 | 0.991914 |
| logsoftmax_fp16 | 0.406301 | 0.676463 | 0.868934 |
| logsumexp_chunked_reverse | 0.257473 | 0.516893 | 0.967998 |

## Independent Clean Control

The clean trajectory was independently rerun. Its saved full-model checkpoints are bitwise identical to the earlier clean A trajectory at steps 1, 5, and 20 (`L2=0` for all three).

## Interpretation Boundary

A zero clipping-fork count is not training equivalence. Distances show state divergence, not task harm; the legal eager/compile trajectory is an empirical comparison envelope, not a certified bound.

A mutation may be called equivalent only if no input can distinguish it from the original program. These finite traces can establish neither program equivalence nor downstream task harm.

## Artifacts

- `results/phase10_zero_fork_mutation_trajectories.json`
- `scripts/phase10_mutation_trajectory_arm.py`
- `scripts/phase10_merge_mutation_trajectories.py`
