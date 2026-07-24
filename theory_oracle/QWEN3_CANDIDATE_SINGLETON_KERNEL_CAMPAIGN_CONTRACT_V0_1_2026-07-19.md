# Qwen3 original-candidate singleton-kernel campaign contract v0.1

## Scope

This campaign shares one frozen model compilation but defines seven separate
generated-kernel treatments: embedding+input norm, causal mask construction,
zero initialization for safe-softmax, final residual RMSNorm `rsqrt`, lm-head
weight cast, final normalization+slice, and logits FP16-to-FP32 cast.

Each family has exactly one invocation.  Each replacement is an eager PyTorch
realization of the named generated fragment and writes the same live output
buffer.  No coverage is transferred between families.

## Fail-closed behavior

Global eager/candidate anchors and graph family must be exact.  For each family,
the exact live module must resolve, two repaired runs must each execute exactly
one call and one repair, repeats must be exact, no backend compilation may
occur, and a restored candidate run must reproduce its anchor.  Runtime
exceptions produce `INVALID_TREATMENT` for that family and zero coverage while
other family records remain independently auditable.  A final two-run candidate
restoration is required for the campaign.

## Endpoints and limits

For each valid family report intervention impact and direction/distance relative
to whole eager.  Null and either directional sign are admissible.  Evidence is
single-state, repair-only and implementation-relative; constituent primitives,
injection sufficiency, population transport and correctness are out of scope.
