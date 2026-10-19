# Direct Triton greater-than-zero binding review — 2026-10-18

This review covers `corpus/calls/triton_greater_zero.py` and
`corpus/call_bindings/triton_greater_zero.json`. It applies only to one contiguous physical finite float32 input of
shape `[1024]` whose magnitudes do not exceed 16.

## Task correspondence

The wrapper accepts exactly one input, checks the declared dtype, shape, finiteness and contiguous layout, allocates
one complete Boolean output, and returns no other value. One launch uses four Triton programs. Each program loads and
compares its own 256 input coordinates and writes the corresponding 256 output coordinates once. The plan binds the
complete input pre-state to the reviewed LOGIC `aten.gt` Tensor/Scalar task with the exactly representable scalar zero,
binds the complete returned tensor, and requires the input to remain unchanged.

The LOGIC profile derives the selected comparison schema and result type from retained PyTorch 2.10 operator schemas,
and the strict greater-than meaning from the retained operator documentation. The call boundary chooses that profile
explicitly; its name is not used to infer the task. A direct GPU 3 sanity run matched `x > 0` at all 1,024 coordinates
and left the input unchanged. This PyTorch comparison was an independent implementation check. The admitted task
value in the formal run came from the reviewed LOGIC evaluator.

## Boolean measurement

The specification keeps Boolean values as Boolean values and requires exact agreement. Only after strict validation,
the measurement layer records `False` as 0 and `True` as 1. It reports exact mismatch, false-positive and false-negative
counts as the primary result. The signed 0/1 difference is retained for the existing statistical projections, but is
not allowed to hide errors that cancel in sign.

## Formal measurement

The reviewed run used 8 development inputs and 16 confirmation inputs, for 24,576 complete output coordinates. Every
contract binding was admitted, every TTIR program reference was complete, and repeated launches were bitwise
identical. The baseline had 24,576 exact matches, no false positive, no false negative, and zero `K-G_pi`, `G_pi-f`
and complete `K-f` residual at every coordinate. The all-zero statistical samples were degenerate, so they do not
establish a population-level absence of bias. The exact statement is agreement on the measured calls. The report is
`.cache/spec_trial_20261018_diverse/triton_greater_zero_controlled_20261018/baseline_report.json`.

A controlled candidate changed only the implemented threshold from zero to `0.25`, while the bound public task
remained strict greater-than zero. Its program reference still matched the candidate, but the task comparison found
2,474 false negatives among 24,576 coordinates, a mismatch rate of about 10.07%, with no false positives. Every
statistical rule on complete `K-f` confirmed a nonzero average effect. The temporary candidate and binding were
removed after the run. Its report is
`.cache/spec_trial_20261018_diverse/triton_greater_zero_controlled_20261018/controlled_threshold_quarter_report.json`.

## Scope

This binding does not cover other shapes, dtypes, strides, nonfinite inputs, thresholds other than the declared scalar
zero, integer or complex comparison, broadcasting between two tensors, in-place or aliased outputs, gradients, fused
surrounding operations, or another Triton kernel with a similar name.
