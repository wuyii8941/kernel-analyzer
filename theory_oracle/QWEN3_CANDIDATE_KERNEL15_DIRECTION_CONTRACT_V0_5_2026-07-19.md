# Qwen3 original-candidate kernel-15 direction contract v0.5

## Why this is a separate revision

Revision v0.4 established, before this contract was written, that repairs of
calls 0, 13 and 26 have nonzero effects on the selected-token log-prob vector.
That result is preserved.  It does not establish that the repair explains the
eager--compiled discrepancy: a nonzero repair could point toward eager, away
from eager, or in an unrelated direction.

This revision keeps the frozen state, kernel family, repair implementation and
call indices unchanged.  It adds only predeclared directional endpoints.

## Arms and endpoints

At the same frozen state, measure exact-repeat output vectors for:

- whole eager/reference execution;
- the unmodified original compiled candidate;
- original-candidate repairs at calls 0, 13 and 26;
- the restored original candidate after all repairs.

For each repair, report:

- candidate-to-repair effect;
- eager-to-candidate distance;
- eager-to-repair distance;
- L2 distance change: `distance(eager, repair) - distance(eager, candidate)`;
- fractional L2 reduction relative to the eager--candidate distance;
- cosine alignment between `repair - candidate` and `eager - candidate`.

Negative L2 distance change and positive cosine alignment are evidence that the
intervention explains part of the implementation-relative discrepancy at this
state.  They are not correctness evidence because eager has no independent
truth authority here.

## Fail-closed gates

- Eager and candidate reproduce their frozen anchors twice.
- The observed Dynamo graph family is exact.
- The live generated module and exact named kernel are resolved.
- Each repaired run observes 27 family calls and replaces exactly one selected
  call.
- All arms repeat exactly.
- No backend compilation occurs during repair arms.
- Restoring the kernel object reproduces the candidate anchor.

## Claim limits

A passing result is an original-candidate-preserving fused-kernel invocation
repair contrast at one matched state.  It does not identify any constituent
`add`, reduction, `rsqrt`, cast, or RMSNorm operator; does not establish
injection sufficiency; does not estimate a state-population effect; and does not
declare either implementation correct.
