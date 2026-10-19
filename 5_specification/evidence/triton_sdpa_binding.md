# Direct Triton float32 causal GQA SDPA binding review — 2026-10-19

This review covers `corpus/calls/triton_sdpa.py` and `corpus/call_bindings/triton_sdpa.json`. The call receives
contiguous float32 q `[2,4,16,16]`, k `[2,2,32,16]` and v `[2,2,32,16]`. It selects scale 0.25, upper-left causal
masking, contiguous-group GQA and no dropout or explicit mask. One launch uses 128 Triton programs, one per batch,
query-head and query-position row, to write the complete public output `[2,4,16,16]`.

The binding maps all three complete inputs, every public output coordinate, the only launch and all three unchanged
input states. The wrapper checks shape, dtype, device, contiguity, finiteness and a bounded input range. A direct GPU 3
comparison with PyTorch 2.10's math backend had maximum absolute difference `4.77e-7` and mean absolute difference
`3.95e-8`; all inputs remained bitwise unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were usable, and all 49,152 output
coordinates had complete call-level references at the requested resolution. No program-to-task coordinate excluded
zero; displayed program-to-task mean and maximum absolute residuals were about `3.51e-17` and `6.11e-16`, with the
small widths enclosing zero. Device-to-program mean and maximum absolute differences were about `3.01e-8` and
`4.73e-7`. The aligned device-to-program rules detected a small negative numerical effect, while the separate
program-to-task rules did not detect a task-formula effect. The report is
`.cache/spec_trial_20261019_sdpa/triton_sdpa/baseline_report.json`.

A temporary controlled candidate added 0.25 only to the final output expression. All 49,152 program-to-task
coordinates excluded zero; their mean absolute difference was 0.25, and both fixed and aligned semantic rule families
detected a nonzero effect. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_sdpa/triton_sdpa/controlled_plus_quarter_report.json`.

This binding does not cover an explicit mask, noncausal attention, another shape or dtype, dropout, another GQA
layout, LSE or backward outputs, or another kernel with a similar name.
