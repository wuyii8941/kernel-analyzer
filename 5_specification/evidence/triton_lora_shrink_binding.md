# Direct Triton float16 LoRA shrink binding review — 2026-10-19

This review covers `corpus/calls/triton_lora_shrink.py` and
`corpus/call_bindings/triton_lora_shrink.json`. The call fixes 64 tokens, four LoRAs, rank 8, hidden size 32, two
weight slices, float16 x and weights, float32 output, and scaling 0.5. A reviewed input generator supplies `-1` and
all four valid LoRA ids. The binding maps every received tensor before execution, the complete output post-state,
and checks x, weights and mapping unchanged.

The direct GPU comparison agreed with a public float32-accumulation PyTorch formula to `2.38e-7`, wrote exact zeros
for all `-1` rows and left every read-only input unchanged. The reviewed call, binding, contract and adapter are
retained as readable files under `validated/`.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable, all
24,576 output coordinates had complete references at the requested resolution, and repeated launches were bitwise
identical. Program formulas matched the admitted exact-real task on every coordinate. Device execution had mean
absolute task residual about `2.31e-8` and maximum about `2.51e-7`; none of the four selected direction rules
confirmed a stable effect. The report is
`.cache/spec_trial_20261019_lora/triton_lora_shrink/baseline_report.json`.

A temporary controlled candidate added 0.25 to every otherwise-correct output. All 24,576 program-to-task
coordinates excluded zero, with mean and maximum absolute residual exactly `0.25`; both fixed-direction and aligned
semantic comparisons confirmed nonzero effects. The command summary separately reported device-to-program floating
error, which remained at its normal scale. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_lora/triton_lora_shrink/controlled_plus_quarter_report.json`.

This binding does not extend to other shapes, dtypes, scaling values, all-inactive calls, arbitrary wrapper metadata,
split-K atomics, expand, the complete B*A update, gradients, layout, aliasing or a kernel selected only by its name.
