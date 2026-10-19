# Direct Triton float32 embedding binding review — 2026-10-19

This review covers `corpus/calls/triton_embedding.py` and `corpus/call_bindings/triton_embedding.json`. The call
receives a contiguous float32 weight `[64,16]`, int64 indices `[128]` and float32 upstream `[128,16]`. One Triton launch
writes the complete forward result and a second deterministic launch writes the complete dense weight VJP. The task
fixes `padding_idx=0`, `scale_grad_by_freq=true`, `max_norm=null` and `sparse=false`. All three input tensors are
checked unchanged.

A direct GPU 3 comparison gave a bitwise identical forward result against public PyTorch 2.10. The dense gradient had
maximum absolute device difference `1.19e-7` and mean absolute difference `4.56e-9`; its padding row was exactly zero.

The first formal run used independently random indices. It exposed an analysis gap for vocabulary rows absent from an
input: the kernel computed a division before selecting the zero result, so only about 86.5% of dense-gradient program
coordinates were established. The kernel was rewritten to use a safe denominator, and the final declared input source
uses a seed-dependent permutation in which every vocabulary row occurs exactly twice. This is the measured input
distribution, not a claim about every possible index distribution.

The final baseline used 8 development and 16 confirmation inputs. All 24 bindings were usable. The forward supplied
49,152 complete call-level coordinates and the dense VJP supplied 24,576; both had complete rate 1.0 and resolved rate
1.0. No program-to-task coordinate excluded zero. Forward was exact. Dense-gradient device-to-program mean absolute
difference was about `1.11e-8`, while program-to-task mean absolute difference was about `5.94e-17` with maximum
`2.22e-16`. The retained report is
`.cache/spec_trial_20261019_embedding/triton_embedding/final_dtype_checked_baseline_report.json`.

A temporary controlled candidate added 0.25 to every non-padding dense-gradient row after the correct computation.
Its 24,192 non-padding program-to-task coordinates all excluded zero; the 384 padding coordinates remained zero. The
program-to-task mean absolute difference was `0.24609375` over the complete tensor and both fixed and aligned semantic
rule classes confirmed a nonzero average effect. Temporary source and binding files were removed; the report remains
at `.cache/spec_trial_20261019_embedding/triton_embedding/final_dtype_checked_controlled_dweight_plus_quarter_report.json`.

This binding does not cover another vocabulary size, width, index distribution, dtype, padding choice, frequency
choice, sparse layout, renormalization, `embedding_bag` or a kernel selected only by its name.
