# Direct Triton float32 BSR softmax values binding review — 2026-10-19

This review covers `corpus/calls/triton_bsr_softmax_values.py` and
`corpus/call_bindings/triton_bsr_softmax_values.json`. The call receives one contiguous float32 value buffer
`[16,16,16]`. The binding supplies the fixed matrix shape `[128,64]`, block shape `[16,16]`, and canonical pattern of
two blocks per block row. One Triton launch writes all 4,096 output values and the input values remain unchanged. The
binding is to the reviewed PyTorch 2.10 BSR softmax values task; a BSR or sparse-softmax name alone is insufficient.

A direct GPU 3 comparison was bitwise identical to the installed PyTorch 2.10 official Triton `bsr_softmax` result.
Against the public float32 softmax formula, the maximum absolute difference was `4.47e-8` and the mean absolute
difference was `1.88e-9`. The input remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable, and all
98,304 task coordinates were complete and reached the requested resolution. No program-to-task interval excluded
zero; its mean absolute residual was about `1.80e-18`. The device-to-program mean absolute residual was about
`2.00e-9`, and the fixed and aligned complete comparisons detected a small negative float32 execution effect relative
to the exact real task. Repeated launches were bitwise identical. The report is
`.cache/spec_trial_20261019_bsr_softmax/triton_bsr_softmax_values/baseline_report.json`.

A temporary controlled candidate added 0.25 only at the final store. All 98,304 program-to-task intervals excluded
zero and their mean absolute difference was exactly 0.25. The aligned semantic rule R3 and both complete comparison
classes detected a nonzero effect. The fixed semantic projections cancelled and correctly remained unestablished;
their result was not overwritten by the other rules. Device-to-program mean absolute residual remained about
`7.79e-9`, separating the injected task-formula error from ordinary device execution. Temporary source and binding
files were removed; the report remains at
`.cache/spec_trial_20261019_bsr_softmax/triton_bsr_softmax_values/controlled_plus_quarter_report.json`.

This binding does not cover another BSR pattern, matrix shape, block shape, dtype, sparse operation, batch or hybrid
layout, gradient, special value, storage effect, or a kernel selected only by its name.
