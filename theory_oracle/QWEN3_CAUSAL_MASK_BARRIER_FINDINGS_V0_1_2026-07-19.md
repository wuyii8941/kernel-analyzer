# Qwen3 causal-mask barrier findings v0.1

The instrumented eager model reproduced the frozen eager anchor exactly and all
arms repeated exactly.  The eager mask target returned `None`; the compiled mask
target returned a Tensor.  Switching target mode caused a new outer compilation,
so the continuation graph was not held fixed.

The eager-context injection contrast was zero, while the compiled-context repair
contrast was nonzero (L2 approximately `0.0720`).  This asymmetry is not evidence
that mask construction is necessary but not sufficient.  It is confounded by
the `None`/Tensor representation change and the associated continuation graph.

The treatment is therefore `INVALID_TREATMENT`, receives zero coverage, and
supports no original-candidate root-cause claim.
