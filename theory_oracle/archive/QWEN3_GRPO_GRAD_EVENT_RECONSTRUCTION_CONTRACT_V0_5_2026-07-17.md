# Qwen3 GRPO Grad-Event Reconstruction Contract v0.5 — 2026-07-17

## Status

Frozen before reconstruction execution.  This is a state/endpoint realization
gate, not an update-effect experiment.

## Selected witness

The v0.4 bank's predeclared ordering selects trajectory A, optimizer step 14,
rollout batch 4, policy iteration 2, case `grpo_000004_956b7dace630`, batch index
2, response token index 21, flat index 277 and token id 220.

The token has negative advantage.  In the bank, eager is unclipped and compiled is
clipped (`0->1`).  The frozen scalar anchors are:

```text
old logp       -1.9253578186035156
eager logp     -2.1442928314208984
compiled logp  -2.170785903930664
```

The frozen complete 512-token scorer hashes are:

```text
eager     ab27446d8506e839f751221a49e74ecb78290827838f931060a475a2b7606562
compiled  b3cdc03d205a1df320a423ebf5fb326945498aa5732fe3e5b22cb8566a276ce9
```

Each hash was identical across the two original measured repeats.

## Reconstruction

Replay the exact trajectory-A configuration and inputs from
`data/r1_from240_step242_pre` to the step-14 pre-minibatch point.  Before the
ordinary step-14 transition:

- save model parameters, buffers, tokenizer and reconstruction metadata;
- retain the exact prompt/completion batch, masks, old log-probabilities and
  advantages;
- rerun the actual Accelerate-wrapped eager and tracked compiled scorers with
  autograd enabled, SDPA MATH and the frozen warm-up/repeat protocol;
- restore RNG and prove the probe does not mutate gradients, tensor versions or
  Trainer steps.

## Acceptance rule

Reconstruction is valid only if all of the following hold:

1. the replayed margin/sample trajectory prefix is byte-identical to the matching
   prefix of the valid v0.4 trajectory;
2. the selected state, case, token alignment, old logp and advantage are unique and
   exact;
3. eager and compiled complete scorer hashes exactly equal the frozen hashes above
   in both repeats;
4. selected scalar log-probabilities and clipping decisions exactly equal the bank;
5. all autograd, candidate invocation, RNG and non-mutation gates pass.

No numerical tolerance may be introduced after execution.  Failure terminates this
witness and emits `INVALID`, not zero update effect.

## Claim boundary

Passing this contract only licenses freezing a subsequent controlled one-step
total/repair query.  It does not prove update impact, compiler incorrectness,
operator cause, population prevalence or long-run harm.
