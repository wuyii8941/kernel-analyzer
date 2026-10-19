# Direct Triton float32 Mamba selective-state-update binding review — 2026-10-19

This review covers `corpus/calls/triton_mamba_ssu.py` and
`corpus/call_bindings/triton_mamba_ssu.json`. The call fixes B=8, H=4, D=32, N=8 and G=2; all nine received tensors
are contiguous finite float32 values, D, z and dt_bias are supplied, threshold-20 softplus is enabled, and state row
b updates state slot b. One Triton program handles each batch/head pair. The binding maps every input, the complete y
return and every coordinate of the state post-state, and checks all non-state tensors unchanged.

Before admission, seventeen random GPU calls were compared coordinate by coordinate with a public PyTorch formula.
The largest absolute difference was `2.38e-7` for y and `4.77e-7` for state; all non-state inputs stayed unchanged.
The reviewed call, binding, contract and adapter are retained as readable files in `validated/`.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable. The
strict task reference was complete and resolved for all 24,576 y coordinates and 196,608 state coordinates, and
repeated launches were bitwise identical. No program-formula coordinate excluded the admitted task. Device execution
had mean absolute task residuals of about `5.17e-9` for y and `1.01e-8` for state. No rule confirmed a directional y
effect. For state, the two aligned rules found small positive average effects, with projected means about `2.27e-7`
and `2.58e-7`; the report preserves this as measured float32 execution bias rather than treating it as a contract
error. The report is `.cache/spec_trial_20261019_ssu/triton_mamba_ssu/baseline_report.json`.

A temporary controlled candidate added 0.25 only when storing final state, after the correct state had already been
used to compute y. All 196,608 state coordinates excluded the admitted task reference; their mean absolute residual
was `0.24999999999`. All 24,576 y coordinates remained at normal baseline scale. This verifies that the complete
in-place state is part of the decision and cannot be hidden by a correct return value. Temporary source and binding
files were removed; the report remains at
`.cache/spec_trial_20261019_ssu/triton_mamba_ssu/controlled_state_plus_quarter_report.json`.

This binding does not extend to another shape, dtype, grouping, optional-argument combination, state-slot mapping,
sequence scan, gradient, layout, alias pattern or kernel selected only by its name.
