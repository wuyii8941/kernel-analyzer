# Direct Triton float32 RMSprop binding review — 2026-10-19

This review covers `corpus/calls/triton_rmsprop.py` and `corpus/call_bindings/triton_rmsprop.json`. The call receives
one contiguous float32 parameter, gradient, square average, centered gradient average and momentum buffer, each with
4,096 coordinates, plus an int64 scalar step equal to 4. It fixes lr 0.125, alpha 0.75, eps 0.25, coupled weight
decay 0.25, momentum 0.5, centered mode and maximize. The first launch updates every vector coordinate; the second
launch increments step to 5.

The binding maps all six complete input tensors, all five complete post-states and both launches. The wrapper checks
shape, dtype, device, layout, finiteness, nonnegative square averages and that the selected centered estimate is in
the finite-real domain. A GPU 3 sanity run confirmed output aliases, an unchanged gradient and all five state updates
against an independent Torch expression within expected float32 rounding.

The formal baseline used 8 development and 16 confirmation inputs. Every binding was reviewed and usable. The four
vector outputs each supplied 98,304 complete coordinates and step supplied 24. No program-to-task difference excluded
zero; exact v and gave coordinates were zero throughout, theta was bounded within `2.22e-16`, and b within
`4.44e-16`. Mean absolute device-to-task differences were about `1.80e-8` for theta, `3.40e-8` for v, `6.60e-9`
for gave and `3.14e-8` for b; step was exact. The aligned numerical rules detected a small direction in b, which is
a device effect because its program-to-task formula remained compatible. The report is
`.cache/spec_trial_20261019_rmsprop/triton_rmsprop/baseline_report.json`.

A temporary controlled candidate computed the correct parameter and averages but stored `b+0.25`. All 98,304 buffer
program-to-task coordinates excluded zero with mean absolute difference 0.25. Theta, v, gave and step retained zero
program-to-task difference. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_rmsprop/triton_rmsprop/controlled_buffer_plus_quarter_report.json`.

This binding does not cover cold state, absent gradients, another option combination, another shape or dtype,
capturable/differentiable execution, multiple parameters or another kernel with a similar name.
