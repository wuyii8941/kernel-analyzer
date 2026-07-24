# Generic Locator / Oracle Role — Case004

Status: exploratory evidence, not a root-cause claim.

## What was supplied to the blind locator

The locator received only the opaque case package (`model.onnx`, `input.npz`,
and the declared reference output), a TVM checkout, and an output path. It did
not receive a Gather name, negative-index rule, candidate region, or repair
recipe. Candidate regions were the complete reachable TIR symbol inventory,
not a delta-ranked shortlist.

## What the Oracle did

The endpoint Oracle compared the compiled output with the declared reference
and recorded shape, exact equality, element disagreement, signed/absolute
delta, finite-value validity, and output fingerprints. It also compared the
two repeated executions separately. This distinction matters in this case:
the first endpoint mismatch is repeatable at the coarse semantic level, but
some runs show very large within-run variation while other runs show only
roundoff-scale variation. Therefore the output cannot honestly be described
as a single deterministic compiler bias.

The pre-reveal certificate consequently remains `OBSERVATION`. The Oracle is
being used as a gate and a diagnostic, not as a root-cause label:

- complete endpoint mismatch: observed;
- repeatability: measured, with runtime variability exposed;
- local same-input replay: not instantiated by the generic locator;
- intervention/repair: not executed by the blind locator;
- provenance: only generic IR-symbol reachability, not source/kernel proof.

## What was found without bug-specific guidance

The buggy run inventory contains the reachable TIR region `tir::take`. The
fixed-version comparison, performed only after the pre-reveal certificate was
frozen, finds the earliest changed generic snapshot at `frontend_relax`, with
the changed region represented as the Relax `main` function. The fixed run
matches the declared reference exactly. The post-reveal result is therefore
`STAGE_AND_IR_REGION_CANDIDATE_ONLY`, not a unique kernel or root-cause claim.

## Why this is evidence that the Oracle is active

If the Oracle were merely recording a raw delta, the two runs would be
indistinguishable except for their first output. Instead it separates:

1. endpoint discrepancy against the declared reference;
2. within-state runtime variability;
3. a fail-closed claim level when local replay, intervention, and provenance
   invariants are absent.

The large repeatability values are themselves a result: they prevent us from
calling this a stable signed shift. They do not invalidate the endpoint
witness, but they lower the strength of any attribution claim.

## Current limit

The generic locator still cannot automatically execute arbitrary TIR symbols
with a fixed suffix, preserve autograd-like state, or prove non-target context
invariance. It therefore cannot yet claim generic operator-level causality.
The next implementation step must add those capabilities only where their
invariants can be checked; otherwise it must stop at `OBSERVATION`.

## Cross-case sanity check

The same bug-agnostic runner was also applied to the patch-excluded Scatter
Elements case003 (not counted as a blind benchmark). It discovered the
reachable `tir::scatter_elements` symbol without a Scatter-specific argument,
observed the endpoint mismatch on the buggy checkout, and observed exact
reference agreement on the fixed checkout. Its post-reveal comparison likewise
identified the earliest changed snapshot as `frontend_relax`. This supports
reuse of the measurement/inventory layer across two ONNX/TVM cases, but does
not establish general localization accuracy.

## Generic region replay after reveal

Two isolated processes were then run without naming a target operator. Each
process enumerated TIR functions and probed only symbols whose buffer
signatures could be mapped unambiguously to the case inputs plus one output.
The post-reveal comparator paired regions by the complete ABI contract rather
than by symbol name, so regenerated names such as `tir::take` and `tir::take1`
could be compared. The Case004 pair produced a same-input cross-version local
output discrepancy, but the isolated probes themselves were not stable across
repeats; it is therefore recorded as `CROSS_VERSION_LOCAL_REPLAY_UNSTABLE`,
not as a stable local-injection claim.

This is stronger than a textual delta, but it is not fixed-suffix mediation:
the isolated probe itself can expose undefined/out-of-bounds behavior and the
compiler context differs between versions. Case003 intentionally fails the
same generic probe because its input and output buffers have an ambiguous
shape/dtype signature; that failure is retained rather than guessed through.

## TVM adapter mediation result

The first TVM adapter now performs a stricter post-reveal test. It swaps only
the ABI-aligned fixed `PrimFunc` into the original buggy Relax module, leaving
the original `main` function unchanged, then recomputes the same endpoint
Oracle. In Case004 the swap changes the endpoint only at a tiny numerical
level and does **not** remove the reference disagreement. The report records
the non-target Relax functions as invariant, but compiler-kernel context as
unproven. This is useful negative evidence: the TIR region swap alone is not
the complete fix; the frontend normalization is still required.

The resulting claim is `INTERVENTION_DEPENDENT_ATTRIBUTION`, explicitly scoped
to this post-reveal PrimFunc swap. It is not an operator-level or compiler
root-cause claim.
