# Phase 14 Held-out Task Reward

## Objective

Determine whether the clean/mutation parameter divergence observed after matched updates changes held-out generated answers or the actual Phase-0 numeric reward.

## Controls

- Arithmetic indices 64-127 are disjoint from the Phase-0 fallback training prompts 0-63.
- Greedy generation, fixed tokenizer, eager SDPA-MATH FP16 evaluation and two independent processes per arm.
- Mutation is absent during evaluation; only its saved training-state consequence remains.

## Arm Summary

| Arm | Prompts | Exact answers | Mean reward | Independent outputs exact |
|---|---:|---:|---:|---|
| initial_step14 | 64 | 53 | 1.7455 | True |
| clean_step20 | 64 | 50 | 1.68367 | True |
| mutation_step20 | 64 | 51 | 1.69935 | True |

## Paired Comparisons

| Left -> right | Completion forks | Reward differences | Exact-outcome forks | Mean reward difference | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---|
| initial_step14 -> clean_step20 | 12 | 5 | 5 | -0.061826 | [-0.16339201835097103, 0.024132984394688382] |
| initial_step14 -> mutation_step20 | 8 | 2 | 2 | -0.0461471 | [-0.11541496266002846, 0.0] |
| clean_step20 -> mutation_step20 | 5 | 3 | 3 | 0.0156789 | [-0.054175724637681166, 0.08717946793177997] |

## BF16 Status

Native BF16 was not run: all visible GPUs are Tesla T4 (SM 7.5), which do not provide native BF16 execution. This is an external-validity gap requiring Ampere or newer hardware.

## Interpretation Boundary

Held-out greedy generation on the synthetic arithmetic reward used in Phase 0. Reward divergence is task-level evidence for this finite task; reward equality does not prove trajectory equivalence or harmlessness.

## Artifacts

- `results/phase14_task_reward.json`
- `results/phase14_task_reward/`
- `scripts/phase14_task_reward_eval_once.py`
- `scripts/phase14_merge_task_reward.py`
