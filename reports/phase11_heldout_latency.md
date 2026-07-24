# Phase 11 Held-out Fork Latency

## Objective

Test whether the step-2/3 delayed clipping forks from Phase 10 reproduce after changing both checkpoint and replay batch.

## Controls

- Held-out optimizer step: `14`
- Held-out rollout batch: `4`
- Replay batch SHA256: `30495faf75b074eb03fc0c067ae5c6148c21f84ad441a41b959815d559110c84`
- Clean and mutation arms use identical token IDs, old logprobs, advantages, optimizer, seed and initial weights.
- Mutation eligibility is determined on the held-out frozen state before trajectory execution.
- No step-5 legal-path envelope is reused at this checkpoint.

## Held-out Initial Gate

| Mutation | Discovery-state forks | Held-out-state forks | Immediate latency |
|---|---:|---:|---:|
| rotary_phase_fp16 | 0 | 2 | 0 |
| decoder0_output_bf16_round | 0 | 2 | 0 |
| logsoftmax_fp16 | 0 | 1 | 0 |
| logsumexp_chunked_reverse | 0 | 0 | not immediate |

## Results

| Mutation | Initial held-out forks | Fork latency | Total trajectory forks | Step-1 parameter L2 | Continuous divergence before fork |
|---|---:|---:|---:|---:|---|
| logsumexp_chunked_reverse | 0 | 2 | 34 | 2.71235e-06 | True |

## Interpretation Boundary

A held-out checkpoint/batch replication of mutation-to-clipping-fork latency under a frozen repeated-batch matched-update protocol. It tests checkpoint/batch dependence, but not fresh-batch training or task harm.

Parameter divergence is a state-level effect, not evidence of reward or task-quality harm.

## Artifacts

- `results/phase11_heldout_latency.json`
- `results/phase11_heldout_mutations/summary.json`
- `results/phase11_heldout_twins/`
- `scripts/phase11_heldout_latency.py`
