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

Gemma-4 E2B is the frozen new-implementation target. Its existing source
prediction and 32-step consequence remain valid as a source-scope negative /
feedback-sustained out-of-domain record. A current-v3 confirmation run was
attempted with the release's engineering warm-up state and the frozen
transformers-5 runtime. It failed closed before measurement because the
generated forward/backward wrapper bytes differed from the frozen release.
The exact expected and observed digests are in
`gemma_v3_runtime_attempt.json`. This is provenance failure, not a scientific
negative and not a new case.

## Current claim boundary

The short Oracle has passed code-level and engineering-path checks, but it has
**zero current-v3 held-out validation records** and zero new implementation
positives. The next valid step is a pre-enumerated pool with release-matching
captures; screen negatives and centered controls remain in the denominator.
No universal all-operator safety claim is made.
