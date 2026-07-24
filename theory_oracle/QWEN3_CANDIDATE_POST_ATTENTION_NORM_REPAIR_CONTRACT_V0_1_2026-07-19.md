# Qwen3 original-candidate post-attention residual-RMSNorm contract v0.1

The frozen forward has 28 calls to
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`, one after
each attention output projection.  It fuses FP32 residual plus FP16 attention
output, RMSNorm reduction/`rsqrt`/weight and FP16 conversion.

Predeclare calls 0, 14 and 27.  Replace only the selected live `.run` with eager
PyTorch operations writing the existing FP16 output buffer.  Require equal
element counts for the residual, attention output and destination and preserve
all other candidate calls, graph, fusion, layout and specialization.

Report intervention impact and direction/distance relative to eager.  Accept
zero or either direction.  Fail closed unless both anchors, graph family, exact
28-call/one-repair accounting, repeats, absence of backend recompilation and
candidate restoration all pass.

Credit is for the whole fused invocation only.  It does not separate residual
addition, reduction tree, `rsqrt`, weight multiply or cast placement; it is
one-state repair evidence without injection, population or correctness claims.
