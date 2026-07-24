# Qwen3 original-candidate query RMSNorm-rotary contract v0.1

The frozen forward has 28 calls to the generated query RMSNorm-rotary family,
one per decoder layer. Each call consumes the FP16 query projection, performs
the width-128 RMSNorm and weight multiplication in FP32, applies rotary
position encoding, applies the query scale, and writes the existing strided
FP32 query destination.

Predeclare calls 0, 14 and 27. Replace only the selected live `.run` with eager
PyTorch operations that preserve the generated kernel's operation order,
rotary frequency duplication, rotate-half convention, query scale, destination
shape and destination strides. Require the runtime destination to have 16
query heads of width 128 and all source widths and element counts to agree.

Report intervention impact and direction/distance relative to eager. Accept
zero or either direction. Fail closed unless both anchors, graph family, exact
28-call/one-repair accounting, repeats, absence of backend recompilation and
candidate restoration all pass.

Credit is for the entire fused invocation. It does not separately identify the
RMSNorm reduction, weight multiplication, trigonometric evaluation, rotary
application, scaling or layout. This is one-state repair evidence without
injection, population or correctness claims.
