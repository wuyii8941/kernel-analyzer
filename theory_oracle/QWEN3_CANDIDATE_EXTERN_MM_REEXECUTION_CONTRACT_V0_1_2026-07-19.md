# Qwen3 candidate external MM reexecution contract v0.1

The frozen forward contains 197 external MM calls: seven projection roles per
decoder layer and one vocabulary projection. Predeclare every projection role
at layers 0, 14 and 27 (21 calls) plus the singleton LM head (call 196).

For each selected call, preserve its candidate inputs and output buffer but
replace the generated module's `extern_kernels.mm` wrapper call with eager
`torch.mm(..., out=...)`. Record all operand shapes, strides and dtypes. Require
exact 197-call/one-reexecution accounting on both repeats, shared live external
object identity, both anchors, graph family, deterministic repeats, no backend
recompilation and exact restoration.

This is reexecution rather than an independent numeric repair: candidate and
eager ATen may resolve to the same CUDA library and algorithm. A null effect
supports a shared-path claim for identical inputs; it cannot establish that
the upstream projection inputs agree or that either implementation is correct.
