# Direct Triton float32-to-bfloat16 cast binding review — 2026-10-18

This review covers `corpus/calls/triton_cast_bf16.py` and
`corpus/call_bindings/triton_cast_bf16.json`. It applies only to the declared contiguous 1024-element call.

## Task correspondence

The wrapper accepts exactly one finite contiguous CUDA float32 tensor of shape `[1024]`, allocates one bfloat16
output of the same shape, and returns that entire output. One launch uses four programs. Program `p` loads and stores
coordinates `256p` through `256p+255`; all 1024 public coordinates are therefore covered exactly once.

The conversion expression names `tl.bfloat16` and passes `fp_downcast_rounding="rtne"`. The retained installed
Triton 3.6 `language/core.py` defines `rtne` as round-to-nearest, ties-to-even and distinguishes it from `rtz` and
bitcast. This directly selects the CAST `finite_rne` task rather than assuming a rounding mode from the operation
name. The plan maps the complete input pre-state, literal target dtype, complete output and unchanged input state.
There are no other arguments, hidden buffers or public outputs.

A direct run on GPU 3 matched `torch.Tensor.to(torch.bfloat16)` bit for bit at all 1,024 random coordinates, returned
the declared shape and dtype, and left the input unchanged. This is a device sanity check; the source, explicit
rounding argument and index map establish the call correspondence.

## Formal measurement

The reviewed contract run used 8 development inputs and 16 confirmation inputs, for 24,576 measured coordinates.
All coordinates had a complete, resolved program reference and an exact task value. The report records
`program_reference_mode: rounding_check`, because the declared task is the explicit RTNE conversion itself.

The maximum absolute difference was zero for all three comparisons: candidate minus program value, program value
minus CAST, and candidate minus CAST. The complete public output was covered. The nonzero-mean tests report that an
all-zero sample is degenerate; this is not a claim of equivalence, but the exact coordinate comparisons contain no
observed discrepancy. The retained report is
`.cache/spec_trial_20261018_diverse/triton_cast_bf16_controlled_20261018/production_rounding_reference_v2_20261018.json`.

An earlier run used the default real-operation program reference. It put the required bfloat16 rounding in the first
difference and its negative in the second, while their sum remained exactly zero. That run exposed an attribution
error rather than a candidate or specification error. The binding now makes the already-supported rounding-aware
program reference explicit; this choice is recorded in every result and does not change candidate minus CAST.

As a controlled detection check, a temporary candidate added `0.25` before the same RTNE conversion while the task
remained unchanged. Candidate minus program value was still exactly zero. Program value minus CAST excluded zero at
all 24,576 coordinates, with mean absolute difference `0.2499980705785371` and maximum `0.265625`; the complete
candidate-minus-CAST comparison showed the same discrepancy, and the fixed-mean rules confirmed a nonzero effect.
The temporary call and binding were removed after the run. Its report is
`.cache/spec_trial_20261018_diverse/triton_cast_bf16_controlled_20261018/controlled_plus_quarter_rounding_reference_20261018.json`.

## Scope

This binding does not cover other shapes, strides, source or target dtypes, nonfinite values, implicit rounding
defaults, `rtz`, float8, integer conversion, bitcasts or fused arithmetic before the conversion.
