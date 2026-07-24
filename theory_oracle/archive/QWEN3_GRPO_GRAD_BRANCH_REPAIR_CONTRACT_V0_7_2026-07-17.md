# Qwen3 GRPO Grad-Branch Repair Contract v0.7 — 2026-07-17

## Status

Frozen before v0.7 execution. The v0.5 and v0.6 `INVALID` results remain part
of the audit trail.

## Added matched-state component

The v0.6 A arm exactly reproduced all eager values, but its cold-start compiled
B arm did not reproduce the event bank. A frozen diagnostic established that
the missing component was shape-specialization history:

- cold compilation at the target `[4,167]` input produced a 455-node graph and
  did not reproduce the bank;
- first compiling a `[4,168]` input and then the target input produced the
  original 455-node static graph followed by the original 457-node dynamic graph;
- under that history, all 512 candidate values and the native `[4,128]` scorer
  hash exactly reproduced the bank.

v0.7 therefore treats compiler specialization state as part of the matched
execution state. Before the first measured B/C target score, it makes one
discarded call with the frozen rollout-0 shape `[4,168]`. Values from this call
are not used in the loss or update.

## Three arms

1. `A_reference`: eager scorer and ordinary branch;
2. `B_candidate`: history-realized compiled scorer and ordinary branch;
3. `C_branch_repair`: the same history-realized compiled scorer and graph as B,
   changing only the selected token's clipping branch to the reference action.

All arms start from the same saved step-14 parameters and use the same controlled
SGD probe (`lr=1e-5`, no momentum). This is not the natural GRPO optimizer.

## Hard gates

- native `[4,128]` eager and candidate hashes exactly equal the v0.4 bank;
- selected scalar and event direction exactly equal the bank;
- B and C each compile exactly the frozen graph sequence
  `455 -> 457` with the frozen graph-code hashes;
- B and C have identical scorer tensors and graph identities;
- the ordinary candidate branch has zero selected-logp loss derivative and the
  repaired branch has nonzero derivative;
- an independent verifier recomputes saved-weight distances.

No tolerance, alternative witness, or cold-start candidate may be substituted
after execution.

## Claim boundary

A valid result estimates an intervention-dependent clipping-branch contribution
to one controlled update at one matched execution state. It does not prove
compiler incorrectness, an operator root cause, natural-update impact,
population prevalence, or long-run training harm.
