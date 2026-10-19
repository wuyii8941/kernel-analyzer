# Direct Triton float32 multiply-add binding review — 2026-10-18

This review covers `corpus/calls/triton_multiply_add.py` and
`corpus/call_bindings/triton_multiply_add.json`. It applies only to three contiguous finite float32 inputs of shape
`[1024]` whose magnitudes do not exceed 16.

## Task correspondence

The wrapper accepts exactly `x`, `y` and `z`, allocates one complete float32 output, and returns no other value. One
launch uses four programs. Program `p` loads and stores coordinates `256p` through `256p+255`, so every public output
coordinate is written once. The kernel source computes `x*y+z`; the binding maps the three complete input pre-states
to the ARITH `prims.fma` task, maps the complete returned tensor, and requires all three inputs to remain unchanged.
There are no scalar options, hidden buffers or omitted public outputs.

The task value is the exact real formula. The binding does not assume that the compiler emits a particular fused
instruction. A GPU 3 sanity run returned the declared dtype and shape and left every input unchanged. Its output was
not bit identical to the separate PyTorch `x*y+z` expression; the maximum difference was about `4.77e-7`. This is
why the PyTorch expression is not used as ground truth.

## Formal measurement

The reviewed run used 8 development inputs and 16 confirmation inputs, giving 24,576 output coordinates. Every
coordinate had a complete call-level program reference and exact task value, and all repeated launches were bitwise
identical. The program-reference mode was `numerical_difference`.

For the normal kernel, `G_pi-f` was exactly zero at all coordinates. `K-G_pi` had mean absolute difference
`2.3181847733477897e-8` and maximum absolute difference `4.134674327360699e-7`; the complete `K-f` comparison had the
same scale. The tested projections did not confirm a stable nonzero mean. This does not prove the call is generally
unbiased, but it shows that the observed floating execution difference was not misclassified as a task-formula
difference. The report is
`.cache/spec_trial_20261018_diverse/triton_multiply_add_controlled_20261018/baseline_report.json`.

A controlled candidate added `0.25` after the same multiply-add while the required task remained `x*y+z`.
`K-G_pi` remained at floating-error scale, with mean absolute difference about `2.73e-8`. `G_pi-f` excluded zero at
all 24,576 coordinates and had mean absolute difference exactly `0.25`; complete `K-f` had mean absolute difference
about `0.25000000035`. The aligned semantic rules and fixed complete-output rules confirmed a nonzero average effect.
The temporary candidate and its binding were removed after the run. Its report is
`.cache/spec_trial_20261018_diverse/triton_multiply_add_controlled_20261018/controlled_plus_quarter_report.json`.

## Scope

This binding does not cover broadcasting, other shapes or dtypes, noncontiguous inputs, nonfinite values, gradients,
in-place writes, other arithmetic operations, or any other Triton kernel with the same informal name.
