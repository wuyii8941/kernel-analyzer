# Qwen3 source-operator causal identifiability boundary v0.1

## Question

Can the current generated-region repair experiments be promoted to complete
causal contributions for individual source operators such as `aten.sum`,
`aten.mul`, `aten.rsqrt` or `aten._to_copy`?

Answer: no.  They identify a declared generated-call intervention, not a
unique decomposition over its constituent source operators.

## Why the decomposition is not identified

A fused generated callable can contain several source operators and can remove
their materialized intermediate tensors.  The experiment observes only the
compiled region output and the full-step endpoint before/after replacing the
whole callable.

Many different assignments of effect to the constituent operators are
compatible with the same observed region effect.  Interactions make the
problem stronger: reassociation, cast placement and reduction order may be
properties of the fused realization rather than of any one source node.
Without additional interventions, there is no unique operator-level solution.

This is not fixed by dividing the region effect equally, assigning it to every
constituent, or choosing the operator whose name appears most salient.  Those
are attribution conventions, not identified causal quantities.

## Why a graph break is not automatically a valid operator repair

Forcing one source operator out of a fused graph can change fusion, layout,
temporary materialization, launch count, scheduling and downstream compiler
choices.  The resulting effect compares two different compiled contexts.  It
cannot be interpreted as “only this operator changed” unless equivalence of all
other relevant mechanisms is independently established.

Likewise, re-running the complete source subgraph in eager form is a valid
region repair but does not isolate any constituent operator.

## What the existing experiments do identify

They identify intervention-dependent effects of:

- one selected invocation of a generated family in a fixed compiled context;
- one singleton generated region in a fixed compiled context;
- their propagation to clipped-gradient, update, semantic-event and post-state
  endpoints;
- variation of those intervention effects across selected matched states.

They do not identify discrepancy-generation, propagation and boundary-
conversion roles separately for every constituent operator.

## Minimum additional capability for source-operator identification

At least one of the following is required:

1. a backend-supported tap/override of a virtual source-node intermediate that
   leaves the surrounding compiled schedule and layout invariant;
2. paired compiled variants with a proof or strong audit that every mechanism
   outside the selected source operator is unchanged;
3. a formal or high-precision node-local specification plus a semantics-
   preserving way to inject its output into the unchanged candidate context;
4. enough independent, valid interventions to identify declared interaction
   terms rather than forcing an additive decomposition.

Until such a capability exists, the scientifically complete result is a
complete source/runtime inventory plus partial generated-region causal
coverage.  Reporting 41/41 runtime observation is valid; reporting 41/41
source-operator causal attribution is not.
