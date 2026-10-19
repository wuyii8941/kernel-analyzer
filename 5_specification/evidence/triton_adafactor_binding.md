# Direct Triton float32 Adafactor binding review — 2026-10-19

This review covers `corpus/calls/triton_adafactor.py` and `corpus/call_bindings/triton_adafactor.json`. The call
receives one contiguous 32×32 float32 parameter and gradient, a 32×1 row state, a 1×32 column state and an int64
scalar step equal to 3. It fixes lr 0.125, beta2 decay -1, eps1 and eps2 0.25, clipping threshold 1, decoupled weight
decay 0.25 and maximize. One Triton program computes both reductions, updates the complete factored state and
parameter, and advances the step to 4.

The binding maps every complete input, all four complete post-states and the only launch. The wrapper checks shape,
dtype, device, contiguity, finiteness, nonnegative variance factors and the selected pre-step. The first GPU compile
revealed that Triton drops a reduction axis: adding a length-32 row reduction directly to a 32×1 state broadcast it to
32×32. The call was corrected by explicitly restoring the row and column singleton dimensions before any evidence
run. A GPU 3 comparison with public PyTorch 2.10 then had maximum absolute differences of `2.38e-7` for theta and
`1.19e-7` for each variance factor; step was exact and the gradient remained bitwise unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were usable. Theta supplied
24,576 complete coordinates, row and col supplied 768 each, and step supplied 24. No program-to-task coordinate
excluded zero. Maximum displayed program-to-task residuals were `2.22e-16` for theta and `1.11e-16` for row and col;
step was exact. Mean absolute device-to-task differences were about `1.87e-8` for theta, `2.71e-8` for row and
`2.74e-8` for col. The report is
`.cache/spec_trial_20261019_adafactor/triton_adafactor/baseline_dtype_checked_report.json`. The earlier
`baseline_report.json` was retained rather than overwritten when the complete-output dtype checks were tightened;
the reference formula and measured values are unchanged.

A temporary controlled candidate first computed the correct parameter and states, then added 0.25 only to every row
state coordinate. All 768 row program-to-task coordinates excluded zero with mean and maximum absolute difference
0.25, and both fixed and aligned rule families confirmed a nonzero effect. Theta, col and step had no
program-to-task coordinate excluding zero. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_adafactor/triton_adafactor/controlled_row_plus_quarter_report.json`.

This binding does not cover vector state, cold state, absent gradients, another matrix shape or dtype, another option
choice, multiple parameters or another kernel with a similar name.
