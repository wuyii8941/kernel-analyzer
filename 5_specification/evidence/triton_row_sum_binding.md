# Direct Triton row-sum call binding review — 2026-10-18

This review covers `corpus/calls/triton_row_sum.py` and
`corpus/call_bindings/triton_row_sum.json`. It applies only to the declared
contiguous 32×256 float32 call.

## Task correspondence

The wrapper accepts exactly one CUDA tensor, requires shape 32×256, float32
dtype and contiguous storage, and returns one complete length-32 tensor. The
kernel launches one program for every row. Each program loads columns 0 through
255 from that row, applies `tl.sum` along the column axis, and stores the result
at the same row index.

The installed Triton 3.6.0 `standard.py` is retained as
`sources/TRITON_STANDARD.py.gz`. Its `sum` definition selects the reduction
dtype, converts when requested, and calls `core.reduce` with the addition
combiner. Together with the explicit pointer indexing, this supports binding
the call to `aten.sum(input, [1])`. The REDUCE `real` profile treats each
received float32 value as an exact real and sums all 256 row elements. Triton's
tree order and float32 rounding remain in the program-numerics comparison.

## Complete boundary

The plan maps the complete input pre-state, the literal reduction dimension,
the full output and the requirement that the input remain unchanged. There are
no other tensor arguments, scalar inputs, hidden buffers or public outputs. The
32-program grid covers all 32 output coordinates, and every program has the
same fixed 256-element read extent.

A direct device comparison on GPU 3 found the expected output shape and dtype,
unchanged input storage, finite outputs, and a maximum absolute difference of
`3.814697265625e-06` from native PyTorch sum. This numeric comparison is only a
sanity check; the source and explicit index mapping establish the task.

After retention, the formal contract entry evaluated 24 independently sampled
calls. Every call used the admitted REDUCE reference and reviewed binding,
captured one complete TTIR launch family, traced the input to the declared
source, verified the unchanged input and complete output writes, and reported
no failure class. All 768 task and program-reference coordinates met the
declared resolution at the first precision level.

The baseline program-to-task difference excluded zero at no coordinate. The
program-numerics and complete candidate-to-task comparisons did not confirm a
nonzero mean under R1, R2, R3 or R5. The complete baseline report is
`.cache/spec_trial_20261018_diverse/triton_row_sum_controlled_20261018/production_reviewed_report_20261018.json`.

A controlled copy added `0.01` after `tl.sum` while retaining the same public
row-sum task. All 768 program-to-task residual coordinates then excluded zero,
with mean absolute residual `0.009999999776482582`; R1 and R5 confirmed a
nonzero effect. The program-numerics comparison still confirmed no rule. Its
report is `.cache/spec_trial_20261018_diverse/triton_row_sum_controlled_20261018/injected_report.json`.
This confirms that the formal entry assigns the injected formula term to the
task difference rather than ordinary reduction rounding.

## Scope and limits

This binding does not cover other shapes, strides, dtypes, masking, multiple
rows per program, split reductions, atomics, persistent kernels, or fused
epilogues. The sampled normal input population supports only the reported bias
queries; lack of detection in the baseline is not proof that every finite input
is unbiased.
