# Qwen3 backward repeated-family repair findings v0.1

## Result

The frozen early/middle/late campaign is valid.  Every arm observed one
backward, one generated module, exactly 27 calls of
`triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32`, and exactly one
repair at its predeclared call index.  Candidate identity, scorer anchors and
the complete natural transition passed in all arms.

The three calls had the same recorded tensor signature but did not have the
same endpoint effect:

| call position | clipped-gradient distance change | AdamW-update distance change |
|---|---:|---:|
| early (0) | -0.0960% | -0.2268% |
| middle (13) | +0.1941% | +0.0187% |
| late (26) | +0.0043% | +0.0014% |

Positive means the repaired endpoint moved closer to the eager endpoint;
negative means it moved farther away.  These are eager-relative impact
directions, not correctness judgments.

Position also changed the repair-effect vector itself.  Pairwise effect-vector
cosines were 0.374 (early/middle), 0.035 (early/late) and 0.045
(middle/late) for clipped gradients, and 0.384, 0.059 and 0.075 for parameter
updates.  Effect magnitudes also differed by more than an order of magnitude
between early and late.

## Interpretation

This is a direct counterexample to assigning one unconditional contribution to
a generated family merely because its name, body, shape and dtype signature are
shared.  The appropriate estimand must condition at least on invocation
position or a mapped source role.  A family-level aggregate would mix one
slightly harmful, one beneficial and one nearly null selected-call effect.

The result does not prove that execution position itself is causal: the calls
also consume different layer activations and gradients.  Nor does it establish
that all 27 positions are distinct.  It establishes only that the proposed
early/middle/late representatives are not interchangeable for this state.

## Instrumentation sham

A separately frozen proxy-forward sham installed the same module proxy and
forwarded all 27 invocations to the original compiled kernel.  It was
exact-null for scaled, unscaled and clipped gradients, parameter updates, all
semantic fields and the complete post-state.  This rules out Python proxy
dispatch and module monkey-patching themselves as the source of the observed
repair effects.

The sham does not match the additional eager arithmetic launches, temporary
allocations or changed fusion boundary of a real replacement.  Those mechanics
remain part of the intervention, so the effects are still
intervention-dependent attribution rather than isolated arithmetic causes.

## Claim boundary

These are selected-state repair effects relative to eager.  There is no
independent correctness authority, injection arm, population transport,
necessity, sufficiency, root-cause or long-run-training claim.

Machine-readable evaluation:
`results/operator_oracle/qwen3_backward_repeated_family_repair_v0_1/evaluation.json`.
Sham verification:
`results/operator_oracle/qwen3_backward_repeated_family_proxy_sham_v0_1/verification.json`.
