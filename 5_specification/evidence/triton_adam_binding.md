# Direct Triton float32 Adam binding review — 2026-10-19

This review covers `corpus/calls/triton_adam.py` and `corpus/call_bindings/triton_adam.json`. The call receives one
contiguous float32 parameter, gradient, first moment, second moment and AMSGrad maximum, each with 4,096 coordinates,
plus an int64 scalar step equal to 4. It fixes lr 0.125, beta1 0.5, beta2 0.75, eps 0.25, coupled weight decay 0.25,
maximize and AMSGrad. The first launch updates every vector coordinate; the second launch increments step to 5.

The binding maps all six complete input tensors, all five complete post-states and both launches. The wrapper checks
shape, dtype, device, layout, finiteness, nonnegative second moments and the selected step. A GPU 3 sanity run
confirmed the returned tensors alias their mutated inputs, the gradient remains bitwise unchanged and all five state
updates agree with an independent Torch expression within expected float32 rounding.

The formal baseline used 8 development and 16 confirmation inputs. Every binding was reviewed and usable. The four
vector outputs each supplied 98,304 complete coordinates and step supplied 24. The program-to-task difference was
zero for all exact moment, maximum and step coordinates; the parameter enclosure residual was at most
`2.22e-16`, far below float32 rounding. Mean absolute device-to-task differences were about `1.75e-8` for theta,
`1.63e-8` for m, `2.87e-8` for v and `1.07e-8` for vmax; step was exact. The report is
`.cache/spec_trial_20261019_adam/triton_adam/baseline_report.json`.

A temporary controlled candidate computed the correct parameter and moments but stored `vmax+0.25`. All 98,304
vmax program-to-task coordinates excluded zero with mean absolute difference 0.25. Theta, m, v and step retained zero
program-to-task difference. This shows why an optimizer specification must include internal public state even when
the parameter output is correct. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_adam/triton_adam/controlled_vmax_plus_quarter_report.json`.

This binding does not cover cold state, absent gradients, another option combination, another shape or dtype,
capturable/differentiable execution, AMP overflow skipping, multiple parameters or another kernel with a similar name.
