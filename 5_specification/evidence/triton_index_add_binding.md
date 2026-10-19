# Direct Triton float32 index-add binding review — 2026-10-19

This review covers `corpus/calls/triton_index_add.py` and `corpus/call_bindings/triton_index_add.json`. It applies
only to one contiguous physical finite float32 self tensor of shape `[128, 32]`, one contiguous source of shape
`[64, 32]`, the exact index vector `[0, 2, ..., 126]`, dimension zero and alpha 0.5.

The wrapper accepts exactly self, source and index, checks the complete fixed index vector, allocates one complete
float32 output and returns no other value. One launch uses sixteen programs. Each program covers 256 consecutive
coordinates, so every public output coordinate is written once. Even output rows add the corresponding source row
times 0.5; odd rows retain self. The binding maps all three complete input pre-states and both scalar choices to the
reviewed INDEXREDUCE profile, maps the complete output, and requires all inputs to remain unchanged. A GPU 3 sanity
comparison with PyTorch 2.10 `torch.index_add` was bit identical, and all inputs remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all
98,304 task coordinates were complete and reached the declared resolution. Repeated launches were bitwise identical.
Program-to-task difference was exactly zero at every coordinate. Device-to-program mean absolute difference was about
`9.04e-9`, with maximum `2.38e-7`. The report is
`.cache/spec_trial_20261019_index_add/triton_index_add/baseline_report.json`.

A temporary controlled candidate added 0.25 after the same indexed addition. Device-to-program error remained at
ordinary float32 scale, with mean absolute difference about `1.33e-8`. All 98,304 program-to-task coordinates excluded
zero and had absolute difference exactly 0.25. Complete device-to-task mean absolute difference was about 0.25, and
both aligned semantic rules and fixed complete-output rules detected a nonzero average effect. The temporary candidate
and binding were removed after the run. Its report is
`.cache/spec_trial_20261019_index_add/triton_index_add/controlled_plus_quarter_report.json`.

This binding does not cover arbitrary index vectors, duplicate indices, another dimension or alpha, other shapes or
dtypes, mutation, index_reduce, scatter_reduce, backward, special values, layouts, aliasing or any other Triton kernel
with the same informal name.
