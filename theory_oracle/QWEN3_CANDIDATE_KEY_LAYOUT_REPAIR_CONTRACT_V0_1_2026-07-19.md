# Qwen3 original-candidate key layout/materialization contract v0.1

The frozen forward has 28 calls to the generated key layout family, one per
decoder layer. It consumes the strided FP32 eight-head rotary key, duplicates
each key/value head twice for the 16 query heads, transposes sequence and head
width for the following BMM, applies the attention scale and writes a
contiguous `(batch, 16, 128, sequence)` FP32 destination.

Predeclare calls 0, 14 and 27. Replace only the selected live `.run` with eager
PyTorch operations. Require a four-dimensional eight-head width-128 source,
the exact destination shape and the exact contiguous destination strides from
the generated code. Preserve all other candidate calls, graph, fusion and
specialization.

Report intervention impact and direction/distance relative to eager. Accept
zero or either direction. Fail closed unless both anchors, graph family, exact
28-call/one-repair accounting, repeats, absence of backend recompilation and
candidate restoration all pass.

Credit is for the whole generated head-repeat/layout/scaling invocation. It
does not separate repeat materialization, permutation or scaling, and it is
one-state repair evidence without injection, population or correctness claims.
