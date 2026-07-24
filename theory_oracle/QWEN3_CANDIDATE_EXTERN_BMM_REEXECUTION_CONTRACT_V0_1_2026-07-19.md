# Qwen3 candidate external BMM reexecution contract v0.1

The frozen forward contains 56 external BMM calls: query-key score and
probability-value products in each of 28 decoder layers. Calls 0/1, 28/29 and
54/55 are predeclared as the two roles at layers 0, 14 and 27.

For each selected call, preserve its candidate inputs and output buffer but
replace the generated module's `extern_kernels.bmm` wrapper call with eager
`torch.bmm(..., out=...)`. Record all operand shapes, strides and dtypes. Require
exact 56-call/one-reexecution accounting on both repeats, shared live external
object identity, both anchors, graph family, deterministic repeats, no backend
recompilation and exact restoration.

This is deliberately described as reexecution rather than an independent
numeric repair: candidate and eager ATen may resolve to the same CUDA library
and algorithm. A null effect supports a shared-path claim for identical inputs;
it does not establish that eager and candidate BMM inputs agree, nor does it
establish correctness.
