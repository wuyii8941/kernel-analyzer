# Direct Triton float32 dropout backward binding review — 2026-10-19

This review covers `corpus/calls/triton_dropout_backward.py` and
`corpus/call_bindings/triton_dropout_backward.json`. It applies only to one contiguous physical finite float32
gradient and one contiguous Boolean mask, both of shape `[1024]`, with the exact scale 1.25.

The wrapper accepts exactly `grad_output` and `mask`, allocates one complete float32 output and returns no other
value. One launch uses four programs. Program `p` loads and stores coordinates `256p` through `256p+255`, so every
public output coordinate is written once. The binding maps the two complete input pre-states and scale to the reviewed
RANDOM `finite_backward` profile, maps the complete output, and requires both inputs to remain unchanged. A GPU 3
sanity comparison with PyTorch 2.10 native dropout backward was bit identical, and both inputs remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all
24,576 task coordinates were complete and reached the declared resolution. Repeated launches were bitwise identical.
The program-to-task difference was exactly zero at every coordinate. Device-to-program mean absolute difference was
about `1.61e-8`, with maximum `2.38e-7`; the tested projections did not confirm a stable nonzero mean. The report is
`.cache/spec_trial_20261019_random/triton_dropout_backward/baseline_report.json`.

A temporary controlled candidate added 0.25 after the same masked multiplication. Device-to-program error remained
at ordinary float32 scale, with mean absolute difference about `1.63e-8`. All 24,576 program-to-task coordinates
excluded zero and had absolute difference exactly 0.25. Complete device-to-task mean absolute difference was about
0.25, and both aligned semantic rules and fixed complete-output rules detected a nonzero average effect. The temporary
candidate and binding were removed after the run. Its report is
`.cache/spec_trial_20261019_random/triton_dropout_backward/controlled_plus_quarter_report.json`.

This binding does not cover stochastic forward dropout, generation or quality of random masks, another scale, other
shapes or dtypes, noncontiguous inputs, special values, gradients with respect to the scale, or any other Triton
kernel with the same informal name.
