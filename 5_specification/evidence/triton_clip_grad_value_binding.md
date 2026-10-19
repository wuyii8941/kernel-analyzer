# Direct Triton float32 gradient value clipping binding review — 2026-10-19

This review covers `corpus/calls/triton_clip_grad_value.py` and
`corpus/call_bindings/triton_clip_grad_value.json`. It applies only to one contiguous physical finite float32 gradient
buffer of shape `[4096]`, clipping bound 0.75 and the selected non-foreach implementation.

The wrapper accepts exactly one gradient tensor, checks its complete shape, dtype, device, layout and finite range,
and returns the same tensor view after mutation. One launch uses sixteen programs. Each program covers 256 consecutive
coordinates, so every gradient coordinate is read and written once. The binding maps the complete input pre-state and
both scalar choices to the reviewed CLIP profile, maps the complete post-state from the same storage as the measured
output, and declares that state change explicitly. A GPU 3 sanity comparison with `torch.clamp` was bit identical and
confirmed that the returned tensor aliases the mutated input.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all
98,304 post-state coordinates were complete. Repeated launches were bitwise identical. Program-to-task,
device-to-program and complete device-to-task differences were exactly zero at every coordinate. The report is
`.cache/spec_trial_20261019_clip/triton_clip_grad_value/baseline_report.json`.

A temporary controlled candidate added 0.25 after the clamp. Device-to-program difference remained at float32 scale,
with mean absolute difference about `1.17e-9` and maximum `2.98e-8`. All 98,304 program-to-task coordinates excluded
zero and had absolute difference exactly 0.25. The semantic direction rules and the complete-output rules detected a
nonzero effect. The temporary candidate and binding were removed after the run. Its report is
`.cache/spec_trial_20261019_clip/triton_clip_grad_value/controlled_plus_quarter_report.json`.

This binding does not cover several gradients in one public call, missing gradients, another clipping bound, norm
clipping, other shapes or dtypes, special values, signed zero, non-contiguous views or any other Triton kernel with a
similar name.
