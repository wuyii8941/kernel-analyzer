# Direct Triton matmul call binding review — 2026-10-18

This review covers `corpus/calls/triton_matmul.py` and
`corpus/call_bindings/triton_matmul.json`. It applies only to the declared
16×32 by 32×16 contiguous float32 call.

## Task correspondence

The wrapper accepts exactly two CUDA tensors. It requires `a` to have shape
16×32, `b` to have shape 32×16, and both tensors to be contiguous float32
values. It returns the complete 16×16 output from one Triton launch.

The kernel reads `a[row, k]` and `b[k, col]`, calls `tl.dot(a, b,
input_precision="ieee")`, and writes every `out[row, col]`. The installed
Triton 3.6.0 definition states that `tl.dot` returns the matrix product of two
compatible two-dimensional blocks. Its `ieee` option selects the input
precision for float32 tensor-core execution. These facts establish the
`aten.mm` task and operand order independently of the observed output.

The admitted MATMUL `real` profile treats each finite received float32 value as
an exact real number and computes the exact sums of products. Device
multiplication, accumulation order, and final float32 rounding remain in the
program-numerics comparison. The precision option therefore does not change
the task formula.

## Complete boundary

The plan maps both tensor pre-states, the full output, and requires both inputs
to remain unchanged. The wrapper has no tensor parameters, buffers, scalar task
arguments, or closed-over tensor state. Its single program id covers the whole
16×16 output, including all 256 coordinates.

An earlier isolated controlled experiment used the same wrapper and mapping on
eight development and sixteen confirmation inputs. Its baseline established a
complete reference for every output coordinate. A second wrapper added `0.01`
after the matrix product: all 6,144 sampled output coordinates then excluded
zero in the task residual, while the program-numerics comparison did not
confirm a stable mean. This supports separation of an injected task term from
ordinary floating execution, but it is not the basis for retaining the current
files.

After retention, the repository measurement declaration passed the formal
entry without a temporary registry override. All 24 calls used the reviewed
binding and an admitted MATMUL reference. Each call captured one TTIR kernel
with complete IR coverage; both loaded tensors were identified as the declared
inputs, every output coordinate was written, both inputs remained unchanged,
and no hidden source or failure class was reported. Repeated executions were
bitwise identical.

The reference covered all 6,144 sampled output coordinates and met the declared
resolution. The task residual excluded zero at no coordinate and none of R1,
R2, R3, or R5 confirmed a nonzero mean. The program-numerics comparison and
complete `K-f` likewise confirmed no mean on these projections. The complete
report is
`.cache/spec_trial_20261018_diverse/triton_matmul_controlled_20261018/production_reviewed_report_20261018.json`.

## Scope and limits

This binding does not cover other shapes, strides, dtypes, batched products,
transposed operands, multiple program ids, split reductions, or different
`tl.dot` precision choices. It also does not certify the exact real formula as
the device arithmetic. Those variants require their own call correspondence
and measurements.

One sampled input population can detect bias under its declared distribution;
the non-detection above cannot prove absence of bias for every finite input.
