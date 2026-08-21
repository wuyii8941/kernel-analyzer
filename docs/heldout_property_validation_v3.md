# Held-out property validation v3

This is the authoritative current-v3 result. Formation and consequence use
disjoint state banks and the same compiled wrapper release in one process.

## Gemma-4 E2B new implementation

The source predictor was frozen on 16 confirmation states:

```text
formation amplification = 0.99834
odd/even cosine          = 0.00306
prediction               = NO_SOURCE_PERSISTENCE_UNDER_PROTOCOL
```

The subsequent 16-step consequence used an independent trajectory bank. The
shared 256-coordinate screen reported:

| path | observed | sign-flip null 95% | result |
|---|---:|---:|---|
| local | 1.0260 | 1.0337 | null-like/abstain |
| feedback | 2.6583 | 1.6386 | risk candidate |
| actual | 2.6457 | 1.6320 | risk candidate |

The full consequence agrees: local amplification 1.0008, feedback 2.6580,
actual 2.6397, final drift L2 0.05548. This is a feedback-sustained,
out-of-domain result, not a source-persistence positive and not a new
Flash-style case.

## Scope conclusion

The current short Oracle has one genuine new-implementation validation record
and zero new source-persistence positives. It correctly preserves the
distinction between a null-like local source and a later feedback risk. This
does not establish universal accuracy; more pre-enumerated implementation
classes are required before making that claim.
