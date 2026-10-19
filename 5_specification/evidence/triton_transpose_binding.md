# Direct Triton float32 transpose binding review — 2026-10-18

This review covers `corpus/calls/triton_transpose.py` and
`corpus/call_bindings/triton_transpose.json`. It applies only to one contiguous finite float32 input of shape
`[32, 64]` whose magnitudes do not exceed 16.

## Task correspondence

The wrapper accepts exactly one input, checks the declared dtype, shape, finiteness and contiguous layout, allocates
one complete contiguous float32 output of shape `[64, 32]`, and returns no other value. One launch uses eight Triton
programs. Together they write all 2,048 public output coordinates once. The binding maps the complete input pre-state
to the LAYOUT `aten.transpose` task with dimensions 0 and 1, maps the complete returned tensor, and requires the input
to remain unchanged. The wrapper allocation removes any output-alias question from this call boundary.

A GPU 3 sanity run returned the declared dtype and shape, was bit identical to PyTorch's logical transpose values,
and left the input unchanged. PyTorch was used here only as an independent implementation check. The admitted task
value is evaluated by the reviewed LAYOUT contract.

## Formal measurement

The reviewed run used 8 development inputs and 16 confirmation inputs, giving 49,152 output coordinates. Every
coordinate had a complete call-level program reference and exact task value. Repeated launches were bitwise
identical. The program-reference mode was `numerical_difference`.

For the normal kernel, `K-G_pi`, `G_pi-f` and complete `K-f` were exactly zero at every coordinate. The statistical
rules reported a degenerate all-zero sample and therefore made no population-level equivalence claim. This result
shows exact agreement for the measured calls; it does not prove that every transpose implementation or every input
is unbiased. The report is
`.cache/spec_trial_20261018_diverse/triton_transpose_controlled_20261018/baseline_report.json`.

A controlled candidate added `0.25` to every stored output while the required task remained transpose. Its
`K-G_pi` stayed at floating-error scale, with mean absolute difference about `5.36e-9` and maximum about `2.38e-7`.
`G_pi-f` excluded zero at all 49,152 coordinates and had mean and maximum absolute difference `0.25`; complete
`K-f` had mean absolute difference about `0.24999999993`. The aligned semantic rules and fixed complete-output rules
confirmed a nonzero average effect. The temporary candidate and binding were removed after the run. Its report is
`.cache/spec_trial_20261018_diverse/triton_transpose_controlled_20261018/controlled_plus_quarter_report.json`.

## Scope

This binding does not cover other shapes or dtypes, noncontiguous or nonfinite inputs, strides and storage identity,
in-place or aliased outputs, gradients, fused surrounding operations, or another Triton kernel with the same name.
