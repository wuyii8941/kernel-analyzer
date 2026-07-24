# ForkCert Phase 11-14 Extension Summary

## Objective

Close four evidence gaps: held-out fork latency, cross-decision propagation, real upstream bug replay, and task-level reward consequence with native-BF16 validation where hardware permits.

## Intervention / Comparison

1. Re-evaluated the four step-5 zero-clipping-fork mutations at a held-out optimizer-step-14 checkpoint and rollout batch 4.
2. Applied the same four mutations to the exact step-5 clipping-survival batch and coupled top-k/top-p sampling with identical uniform random numbers.
3. Replayed PyTorch upstream issues `#186577` and `#183986` in two independent processes each.
4. Compared initial-step14, clean-step20 and mutation-step20 checkpoints on held-out arithmetic prompts 64-127 using the original Phase-0 numeric reward.

## Controls

- Held-out checkpoint and batch are both disjoint from the latency discovery state; replay batch SHA256 is `30495faf75b074eb03fc0c067ae5c6148c21f84ad441a41b959815d559110c84`.
- Sampling is restricted to rollout batch 1, exactly the 512-token batch on which all four mutations had zero clipping fork.
- Sampling uses 64 common random draws per state and two exact self-runs per clean/mutation path.
- Historical bugs use upstream minimal reproducers, eager and `aot_eager` controls, and two independent Inductor processes.
- Task evaluation removes the mutation implementation: only saved parameter-state consequences remain. All three arms have two bitwise-identical independent generation runs.

## Result

### Held-out latency

At the held-out frozen state, `rotary_phase_fp16`, `decoder0_output_bf16_round`, and `logsoftmax_fp16` already cause 2, 2 and 1 clipping forks, respectively. Their state-conditional latency is therefore zero. `logsumexp_chunked_reverse` still has zero initial fork, accumulates a nonzero one-step parameter distance (`2.71235e-6`), and first causes a clipping fork at matched update step 2. It produces 34 branch-fork events over 20 steps.

The fixed step-2/3 latency is not universal. The reproducible result is that fork latency is state-conditioned and ranges from immediate to delayed under the tested states.

### Cross-decision propagation

On the exact 512-token clipping-survival batch:

| Mutation | top-k candidate-set forks | top-p candidate-set forks | top-k fork draws | top-p fork draws | First-draw token-fork states |
|---|---:|---:|---:|---:|---:|
| rotary phase FP16 | 59 | 33 | 482 | 504 | 10 |
| decoder-0 BF16 round trip | 43 | 23 | 355 | 412 | 14 |
| log-softmax FP16 | 0 | 18 | 39 | 95 | 1 |
| reverse-chunk logsumexp | 0 | 0 | 0 | 0 | 0 |

All self-run failures are zero. Three of four mutations that survived the clipping boundary cross an actual common-random-number sampling boundary on the same frozen batch.

### Upstream bugs

- PyTorch `#186577` reproduces in both independent runs on Torch `2.13.0.dev20260609+cu126` and T4. Eager equals `aot_eager` exactly; Inductor max error is `20.703125`. The pre-registered argmax, top-16 and fixed-threshold decisions do not fork.
- PyTorch `#183986` no longer produces a silent wrong result in this runtime. Both runs fail closed with an explicit alias-write RuntimeError.

This is evidence that delta and semantic events are complementary: a real large-amplitude wrong result need not cross every observed decision boundary. These are upstream bug replays, not new bug discoveries.

### Task reward

The clean-step20 versus mutation-step20 comparison has 5/64 generated-sequence forks, 3/64 numeric-reward differences and 3/64 exact-answer outcome forks. Mean reward difference is `+0.01568` for mutation minus clean, with paired bootstrap 95% CI `[-0.05418, 0.08718]`.

The finite task therefore contains reproducible task-level outcome forks, but no statistically supported direction in average reward.

## Certificate / Artifacts

- `results/phase11_heldout_latency.json`
- `reports/phase11_heldout_latency.md`
- `results/phase12_mutation_sampling_gated/summary.json`
- `reports/phase12_mutation_sampling_gated.md`
- `results/phase13_historical_bug_replays.json`
- `reports/phase13_historical_bug_replays.md`
- `results/phase14_task_reward.json`
- `reports/phase14_task_reward.md`

## Interpretation

The new evidence rejects two overly broad simplifications: fork latency is not a mutation-level constant, and a large real-bug delta does not guarantee a fork at a particular observed decision. It supports a staged SE4DL analysis: numerical infection, decision fork, state divergence and task outcome are distinct checkpoints.

## Next Decision

`GO` for the revised complementary SE4DL claim. `REVISE` for any claim that fork replaces delta or that task reward changes have a stable direction. Native BF16 remains `NOT RUN`: all visible GPUs are T4/SM75, so Ampere-or-newer hardware is required.
