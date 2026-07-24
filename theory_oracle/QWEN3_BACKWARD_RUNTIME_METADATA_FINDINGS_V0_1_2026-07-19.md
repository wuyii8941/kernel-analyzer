# Qwen3 backward runtime metadata findings v0.1

## Result

The metadata census is valid for the heldout-transport-B step-29 natural
transition.  All 1,126 generated Triton calls and all 731 external calls
(`mm=563`, `bmm=168`) match the independently frozen static denominator.  The
transition retained the anchored scorer and completed loss, backward, gradient
clipping, AdamW, GradScaler and scheduler behavior.

The log contains tensor shape, stride, dtype, device, storage offset and
requires-grad only.  It contains no tensor values or storage contents.

## What the signatures resolve

Ignoring storage offset and tensor contents yields 68 exact argument-signature
strata across 41 runtime-observed family names.  Seven families have multiple
signatures.  In particular, backward `bmm` separates into six 28-call shape
strata and backward `mm` into seventeen strata.

This is useful for intervention design, but it is not a semantic equivalence
result.  The same signature can still contain different layers, parameter
roles, attention branches, or gate/up paths.

## What remains unresolved

Ten families remain explicitly multi-role or ordering-unresolved:

- external `bmm` and `mm`;
- one 57-call fused slice/reduction family;
- seven repeated cast/layout families with 56 or 84 calls.

Several of those cast families have only one signature despite having multiple
semantic roles.  Therefore shape-based clustering alone cannot justify a
representative-call repair or the claim that all backward operators are
causally covered.

## Consequence for the next experiment

The next causal evidence should begin with singleton families whose eager
semantic boundary can be reconstructed, then move to role-aware repeated-call
interventions.  Complex fused reductions remain uninstantiated until their
source-graph boundary is reproduced or their non-identifiability is stated.

This census grants runtime-denominator and argument-stratification credit only:
no repair, injection, population, long-run, root-cause or correctness credit.
