# Held-out property validation v2

This is the current audit after the v3 short-screen engineering revalidation.
It supersedes neither the frozen development protocol nor the v1 provenance
record; it records what is and is not evidence for the shared low-cost Oracle.

## What was measured

The v3 screen (256 streamed CountSketch coordinates, after-warmup prefix rule)
was exercised on three engineering records: Llama-3.2 3B lm-head dX,
Ministral-3 3B lm-head dX, and Qwen saved-P. These are useful implementation
and memory-path checks, but they are not held-out validation because the first
two reuse the seen lm-head implementation family and the Qwen record is a
development case.

The results are retained in
`short_screen_v3_engineering_validation.json`; none adds a case.

## New implementation attempt

Gemma-4 E2B is the frozen new-implementation target. Its current-v3
confirmation was rerun in one process that froze the wrapper release before
formation and then executed the consequence. The source prediction was
`NO_SOURCE_PERSISTENCE_UNDER_PROTOCOL` (formation amplification 0.9983).
The local path was null-like (1.0032 versus sign-flip null upper 1.0065), while
feedback and actual paths were risk candidates (2.642 and 2.515 versus null
upper bounds 1.621 and 1.558). This reproduces the existing interpretation:
Gemma is a feedback-sustained out-of-domain case, not a source-persistence
positive and not a new Flash-style case. The compact result is
`heldout_validation_v2_gemma_v3.json`.

## Current claim boundary

The short Oracle now has **one current-v3 held-out execution record** on a new
implementation class, with the expected scope separation, and zero new
implementation source positives. This is not enough for universal accuracy:
the next valid step is a pre-enumerated pool with more release-matching
captures; screen negatives and centered controls remain in the denominator.
No universal all-operator safety claim is made.

The accompanying escalation selector sends only the `feedback` and `actual`
risk paths to exact follow-up and leaves the null-like `local` path as an
abstention (`2` escalations, `1` non-escalation, `0` invalid). It never emits a
SAFE verdict.
