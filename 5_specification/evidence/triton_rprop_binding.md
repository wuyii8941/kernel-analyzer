# Direct Triton float32 Rprop binding review — 2026-10-19

This review covers `corpus/calls/triton_rprop.py` and `corpus/call_bindings/triton_rprop.json`. The call receives one
contiguous length-4096 float32 parameter, gradient, previous-gradient state and step-size state, plus an int64 scalar
step equal to 4. It fixes lr 0.125, factors 0.5 and 1.5, bounds 0.03125 and 0.75, and maximize. Sixteen Triton programs
update all vector coordinates and one program advances the step to 5.

The binding maps every complete input, all four complete post-states and every launch. The wrapper checks shape,
dtype, device, contiguity, finiteness, nonnegative loaded step size and the selected pre-step. A direct GPU 3
comparison was bitwise identical to public PyTorch 2.10 for theta, prev, eta and step; the sampled input exercised
2,000 positive-product and 2,096 negative-product coordinates, and the received gradient remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were usable. Theta, prev and eta
each supplied 98,304 complete coordinates, and step supplied 24. No program-to-task coordinate excluded zero, and
the displayed program-to-task residual was zero for every output. Mean absolute device-to-task differences were about
`1.02e-8` for theta and `3.15e-9` for eta, with maxima `2.38e-7` and `2.98e-8`; prev and step were exact. The report is
`.cache/spec_trial_20261019_rprop/triton_rprop/baseline_report.json`.

A temporary controlled candidate first performed the correct transition, then added 0.25 only to every eta
coordinate. All 98,304 eta program-to-task coordinates excluded zero with mean and maximum absolute difference 0.25.
Theta, prev and step had no program-to-task coordinate excluding zero. The report's top-level statistical labels
describe device-to-program differences, so they do not replace this separate semantic comparison. Temporary source
and binding files were removed; the report remains at
`.cache/spec_trial_20261019_rprop/triton_rprop/controlled_eta_plus_quarter_report.json`.

This binding does not cover cold state, absent gradients, another vector length or dtype, another option choice,
multiple parameters or another kernel with a similar name.
