# Qwen3 backward repeated-family repair contract v0.1

## Question

Does one generated backward family have the same selected-state semantic impact
at different invocation positions, or does a family-level aggregate conceal
position-conditioned effects?

## Frozen subject and interventions

The subject is
`triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32`, observed exactly
27 times in the audited compiled backward.  All calls have the same recorded
tensor signature.  Before execution, this campaign selects the zero-based call
indices 0, 13 and 26, labelled `early`, `middle` and `late`.

Each arm replaces exactly one selected invocation with the corresponding eager
source-graph operations for the SiLU value, up-branch gradient and gate-branch
gradient.  The other 26 invocations remain compiled.  The complete natural
training transition and the clipped-gradient and AdamW-update vectors are the
endpoints.

## Validity gates

A result is valid only if the base transition, compiled candidate identity and
scorer anchor are valid; exactly one backward hook and one generated module are
observed; the family executes exactly 27 times; and the preselected call is hit
and repaired exactly once.  Any failed gate invalidates that arm.

## Interpretation

The primary comparison is the change in distance from the compiled endpoint to
the eager endpoint for clipped gradients and parameter updates.  Effect-vector
alignment across the three repairs is secondary evidence about whether the
positions are interchangeable for this state.

This is selected-state, selected-call, intervention-dependent attribution.  A
call position is not automatically a source-layer identity.  The treatment is
not an independent specification and cannot establish correctness, root cause,
necessity, sufficiency, population transport or long-run training impact.
