# Qwen3 SDPA decomposition findings v0.1

The original eager arm repeated exactly.  The explicit qk-bmm / safe-softmax /
pv-bmm reconstruction also repeated exactly, but it did not reproduce the
original eager tensor.  The selected-token log-probability reconstruction delta
had L2 approximately `0.07020`.

Consequently all nine attempted primitive treatments receive zero coverage.
The result is a valid invalidation, not a failed infrastructure run.

The main theoretical lesson is that an AOT graph's visible primitive sequence
does not by itself define a semantics-preserving source-level replacement.
Hidden dispatch rules, causal-mask representation, dtype promotion, scaling,
layout, or backend implementation details can remain part of the treatment.
Primitive attribution requires an exact replacement gate or a lower-level
candidate-preserving intervention.
