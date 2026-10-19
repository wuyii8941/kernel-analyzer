# Direct Triton float32 Llama RoPE binding review — 2026-10-19

This review covers `corpus/calls/triton_llama_rope.py` and
`corpus/call_bindings/triton_llama_rope.json`. The call receives q `[2,4,16,16]`, k `[2,2,16,16]`, cos/sin
`[2,16,16]` and same-shaped q/k upstream tensors. Two Triton launches write all four public results: q output and q
VJP in the first launch, then k output and k VJP in the second. The plan maps every task input and complete output and
checks all six input tensors unchanged. It binds specifically to the Transformers 4.57.3 supplied-table,
rotate-half task; a name containing `rope` would not be enough.

A direct GPU 3 comparison against the public PyTorch formula and autograd gave maximum absolute differences of
`4.77e-7` for q output, `4.77e-7` for k output, `4.77e-7` for q VJP and `2.38e-7` for k VJP. Mean absolute differences
were between `1.18e-8` and `1.43e-8`. All inputs remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. Every binding was reviewed and usable, and all
four outputs had complete and resolved rates of 1.0. The q output and q VJP each supplied 49,152 coordinates; the k
output and k VJP each supplied 24,576. Program-to-task differences were exactly zero at every coordinate. Mean
absolute device-to-program differences were `1.61e-8`, `1.63e-8`, `1.62e-8` and `1.64e-8` respectively; the largest
device-to-task difference over the four outputs was about `2.31e-7`. The retained report is
`.cache/spec_trial_20261019_llama_rope/triton_llama_rope/baseline_dtype_checked_report.json`.

A temporary controlled candidate added 0.25 only when writing both gradients after the correct computation. All
49,152 q-gradient and 24,576 k-gradient program-to-task coordinates excluded zero, with mean absolute difference
exactly 0.25. Both aligned semantic rules detected a nonzero effect, and the complete candidate-to-task fixed rules
also detected it. The two forward results retained exact zero program-to-task difference. Device-to-program mean
absolute differences remained around `2.0e-8` for the gradients, showing that the injected term was assigned to the
task-formula difference rather than ordinary device execution. Temporary source and binding files were removed; the
report remains at
`.cache/spec_trial_20261019_llama_rope/triton_llama_rope/controlled_grad_plus_quarter_report.json`.

This binding does not cover another shape, dtype, table producer, partial rotary dimension, interleaved layout,
position lookup, another model family, or a kernel selected only by its name.
