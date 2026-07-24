# Qwen3 backward singleton-repair contract v0.1

This campaign intervenes on five generated backward families that each execute
exactly once in the audited step-29 compiled backward.  Each treatment replaces
that generated invocation with a composition of PyTorch operations matching the
source graph's eager dtype boundary.

The predeclared treatments are: tangent cast/view, embedding-gradient zero
initialization, SiLU-multiply, its local backward, and FP16-to-FP32 add.

A run is valid only if the frozen scorer anchor and compiled candidate identity
remain valid, one backward hook occurs, exactly one generated module is resolved,
and exactly one family call is replaced once.  Gradient and parameter-update
vectors must be retained for comparison with the existing eager and compiled
natural-transition arms.

The estimand is the selected-state change in the complete clipped-gradient and
parameter-update vectors caused by this exact local replacement.  Direction
toward eager is implementation-relative impact, not correctness.  A null result
does not prove the family harmless outside this state; a non-null result does not
prove that the generated family is a root cause.
