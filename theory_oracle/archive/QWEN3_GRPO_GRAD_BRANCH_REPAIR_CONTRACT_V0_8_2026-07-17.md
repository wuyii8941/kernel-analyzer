# Qwen3 GRPO Grad-Branch Repair Contract v0.8 — 2026-07-17

## Status

Frozen before v0.8 execution. Results from v0.5, v0.6 and v0.7 remain
`INVALID` and are retained as provenance.

## Endpoint realization correction

v0.7 reproduced every eager/candidate scorer value and B/C scorer identity, but
its hand-written autocast wrapper compiled to a structurally different graph
from the original Trainer (`453 -> 455` nodes rather than `455 -> 457`). Its
predeclared graph gate therefore rejected the run.

v0.8 uses the exact execution structure already validated by the independent
history diagnostic:

1. load FP32 master parameters and enable gradient checkpointing;
2. apply `Accelerator(mixed_precision="fp16").prepare_model`, which installs the
   native Trainer-style autocast/output wrapper;
3. do not add a second outer autocast wrapper;
4. before the first measured compiled target call, make one discarded
   `[4,168]` specialization call;
5. score the target `[4,167]` input and hash its native `[4,128]` output.

The diagnostic showed this produces the original graph-code hashes, node counts,
all 512 candidate values and selected scalar exactly.

## Intervention and hard gates

The arms and controlled SGD probe are unchanged from v0.7. A is eager ordinary,
B is compiled ordinary, and C is the identical compiled scorer with only the
selected clipping branch forced to the reference action.

Execution is accepted only if:

- complete eager/candidate hashes and selected values equal the v0.4 bank;
- B and C both compile exactly the frozen graph sequence and have identical
  full scorer tensors;
- candidate runtime invocation gates pass;
- the selected-logp loss derivative is nonzero for A, zero for B, and nonzero
  for C;
- the independent verifier recomputes all saved-weight distances.

No tolerance or post-execution graph substitution is permitted.

## Claim boundary

A valid result is a single-state, intervention-dependent estimate of the
selected clipping branch's contribution to a controlled one-step update. It is
not a correctness verdict, operator cause, estimate of natural AdamW/GRPO update,
population result, or long-run training claim.
