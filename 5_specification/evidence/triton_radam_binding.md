# Direct Triton float32 RAdam binding review — 2026-10-19

This review covers `corpus/calls/triton_radam.py` and `corpus/call_bindings/triton_radam.json`. The call receives one
contiguous float32 parameter, gradient, first moment and second moment, each with 4,096 coordinates, plus an int64
scalar step equal to 5. It fixes lr 0.125, beta1 0.5, beta2 0.875, eps 0.25, coupled weight decay 0.25 and maximize.
The resulting `rho_6` is about 5.23, safely inside the rectified branch. The first launch updates every vector
coordinate and the second launch advances the step to 6.

The binding maps all five complete inputs, all four complete post-states and both launches. The wrapper checks shape,
dtype, device, contiguity, finiteness, nonnegative second moments and the selected pre-step. A GPU 3 sanity comparison
with public PyTorch 2.10 RAdam had maximum absolute differences of about `1.19e-7` for theta, `2.38e-7` for m and
`2.38e-7` for v; step was 6, the gradient remained bitwise unchanged and every output retained the declared alias.

The first formal attempt exposed a binding error rather than accepting a small discrepancy. It precomputed the
rectification factor as a host float, so the program's real formula differed from the task formula on theta by mean
absolute `3.10e-11` and maximum `1.60e-10`. That report is retained at
`.cache/spec_trial_20261019_radam/triton_radam/baseline_report.json`. The admitted call now constructs beta2 powers,
rho and the rectification factor inside the Triton program from exactly representable beta2. This is required for the
program formula and task formula to describe the same computation; the earlier float constant is not treated as an
acceptable approximation.

The corrected formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and
usable. Theta, m and v each supplied 98,304 complete coordinates and step supplied 24. No program-to-task coordinate
excluded zero. Program-to-task m, v and step were exactly zero; theta's interval display residual had mean absolute
`1.09e-17` and maximum `2.22e-16`. Mean absolute device-to-task differences were about `1.72e-8` for theta,
`1.63e-8` for m and `2.58e-8` for v; step was exact. The corrected report is
`.cache/spec_trial_20261019_radam/triton_radam/runtime_formula_baseline_report.json`.

A temporary controlled candidate computed theta and v from the correct moment but stored `m+0.25`. All 98,304 m
program-to-task coordinates excluded zero with mean and maximum absolute difference 0.25; the aligned rule family
confirmed a nonzero average effect. Theta, v and step retained no program-to-task coordinate excluding zero. Temporary
source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_radam/triton_radam/controlled_m_plus_quarter_report.json`.

This binding does not cover cold state, absent gradients, the unrectified branch, decoupled decay, another shape or
dtype, capturable or differentiable execution, multiple parameters or another kernel with a similar name.
