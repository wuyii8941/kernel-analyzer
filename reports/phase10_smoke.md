# Phase 10 Zero-Initial-Clipping-Fork Mutation Trajectories

## Objective

Determine whether mutations with zero clipping forks at the frozen initial state can still change PPO gradients and parameter updates through the continuous within-branch channel.

## Results

| Mutation | First clipping-fork step | Step-1 parameter L2 | Ratio to legal eager/compile | Continuous divergence before fork |
|---|---:|---:|---:|---|
| logsumexp_chunked_reverse | none | 2.84488e-06 | 0.257473 | True |

## Interpretation Boundary

A zero clipping-fork count is not training equivalence. Distances show state divergence, not task harm; the legal eager/compile trajectory is an empirical comparison envelope, not a certified bound.

A mutation may be called equivalent only if no input can distinguish it from the original program. These finite traces can establish neither program equivalence nor downstream task harm.

## Artifacts

- `results/phase10_smoke/merged.json`
- `scripts/phase10_mutation_trajectory_arm.py`
- `scripts/phase10_merge_mutation_trajectories.py`
