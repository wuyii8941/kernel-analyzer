# Direct Triton exponential call binding review — 2026-10-18

This review covers `corpus/calls/triton_exp.py` and `corpus/call_bindings/triton_exp.json`. It applies only to the
declared contiguous 1024-element float32 call and the sampled input population.

## Task correspondence

The wrapper accepts exactly one CUDA float32 tensor with shape `[1024]`, contiguous storage, finite values and the
MATHFN magnitude bound. It returns one complete tensor of the same shape and dtype. One launch uses four Triton
programs; program `p` loads and stores coordinates `256p` through `256p+255`. Every loaded value is passed once to
`tl.exp`, with no reduction, mask-dependent public value or fused epilogue.

The installed Triton 3.6.0 `language/math.py` is retained as `sources/TRITON_MATH.py.gz`. It defines `tl.exp` for
fp32/fp64 blocks as an element-wise exponential builder operation and preserves the block type. Together with the
explicit indexing, this supports binding the call to `aten.exp(input)` in the reviewed MATHFN `real` profile. The
device implementation of that exponential and float32 output rounding remain in the program-numerics comparison.

## Complete boundary and production result

The plan maps the full input pre-state, full public output and unchanged input state. There are no other tensor or
scalar arguments, hidden buffers or public outputs. A direct GPU 3 sanity check confirmed the shape, dtype, finite
output and unchanged input; the largest difference from native `torch.exp` was `1.9073486328125e-06`. This comparison
is only a device sanity check.

The formal contract entry evaluated 24 independently sampled calls. All 24 bindings used the admitted MATHFN
reference, captured the complete TTIR launch, traced the only input to the declared source, checked the unchanged
input and full output writes, and reported no failure class. All 24,576 reference coordinates were finite, complete
and resolved at precision level 1; repeated launches were bitwise identical.

For the normal candidate, no `Gπ−f` coordinate excluded zero and its mean and maximum absolute task residual were
zero. The device-to-program and complete device-to-task comparisons found the same small numerical effect: R1 and R2
were negative with mean projection about `-1.2066e-7` and adjusted p about `0.00230`; R5 and R3 were not confirmed.
The complete device-to-task mean absolute residual was about `6.57e-8`, maximum absolute residual about `1.27e-5`,
and relative RMS about `6.90e-8`. This is a numerical `tl.exp` result under the declared population, not a task
formula error. The report is
`.cache/spec_trial_20261018_diverse/triton_exp_controlled_20261018/production_reviewed_report_20261018.json`.

A controlled copy added `0.01` after `tl.exp` while retaining the same public exponential task. All 24,576 `Gπ−f`
coordinates then excluded zero; the mean absolute task residual was `0.009999999776482582`, and R1, R5, R2 and R3
all confirmed nonzero effects. Its report is
`.cache/spec_trial_20261018_diverse/triton_exp_controlled_20261018/injected_report.json`. This confirms that the formal
entry separates the injected task-formula term from the ordinary device approximation.

## Scope and limits

This binding does not cover other shapes, strides, dtypes, input populations, special values, larger magnitudes,
other math functions, libdevice calls or fused epilogues. The baseline finding is conditional on 24 calls drawn from
the declared normal sampler; it is evidence for that population, not a universal statement about every Triton
exponential implementation or every finite input.
