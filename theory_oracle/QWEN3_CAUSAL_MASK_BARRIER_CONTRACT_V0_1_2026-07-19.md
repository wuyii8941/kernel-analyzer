# Qwen3 causal-mask barrier contract v0.1

At the frozen step-29 state, isolate the singleton high-level
`create_causal_mask` invocation using a fixed boundary.  The instrumented eager
model forward must exactly reproduce the original eager selected-log-probability
tensor.

The experiment records whether eager and compiled target arms return `None` or
a Tensor.  Injection and repair receive `BARRIER_CONDITIONED` credit only if
repeats are exact, target modes execute as declared, and switching mask mode does
not cause a new outer compilation.  A representation-type change that alters
the continuation graph is an invalid treatment for fixed-context attribution,
even if the end observable changes.

No result identifies the individual indexing, comparison, `where`, or cast
primitives used to construct a materialized mask.
